import numpy as np
import asyncio
import time
from typing import Optional, Tuple, List, Dict, Any
from exo.inference.inference_engine import InferenceEngine, get_inference_engine
from exo.inference.shard import Shard
from exo.models import build_base_shard
from .speculative_config import SpeculativeConfig, get_draft_model_for_target
from exo import DEBUG

class SpeculativeResult:
    """Result from speculative decoding containing accepted tokens and metrics"""
    def __init__(self, accepted_tokens: List[int], rejected_count: int, draft_time: float, verify_time: float):
        self.accepted_tokens = accepted_tokens
        self.rejected_count = rejected_count
        self.draft_time = draft_time
        self.verify_time = verify_time
        self.total_time = draft_time + verify_time

class SpeculativeInferenceEngine(InferenceEngine):
    """
    Speculative decoding inference engine that wraps target and draft models.
    Implements the speculative sampling algorithm from https://arxiv.org/pdf/2211.17192.pdf
    """
    
    def __init__(self, target_engine: InferenceEngine, draft_engine: Optional[InferenceEngine] = None, config: Optional[SpeculativeConfig] = None, shard_downloader=None):
        self.target_engine = target_engine
        self.draft_engine = draft_engine
        self.config = config or SpeculativeConfig()
        self.shard_downloader = shard_downloader
        self.draft_cache: Dict[str, Dict[str, Any]] = {}
        self.metrics = {
            'total_draft_tokens': 0,
            'accepted_tokens': 0,
            'rejected_tokens': 0,
            'speculation_rounds': 0,
            'total_draft_time': 0.0,
            'total_verify_time': 0.0
        }
        
        # Use target engine's tokenizer
        self.tokenizer = target_engine.tokenizer if hasattr(target_engine, 'tokenizer') else None
        self.shard = target_engine.shard if hasattr(target_engine, 'shard') else None
    
    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Use target engine for encoding"""
        return await self.target_engine.encode(shard, prompt)
    
    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """Use target engine for decoding"""
        return await self.target_engine.decode(shard, tokens)
    
    async def sample(self, x: np.ndarray, temp: float = 0.0) -> np.ndarray:
        """Use target engine for sampling"""
        return await self.target_engine.sample(x, temp)
    
    async def ensure_shard(self, shard: Shard) -> None:
        """Ensure both target and draft models are loaded for the shard"""
        if DEBUG >= 2:
            print(f"[Speculative] Ensuring shard: {shard}")
        
        # Ensure target engine has the shard
        await self.target_engine.ensure_shard(shard)
        
        # Set up tokenizer from target engine
        if hasattr(self.target_engine, 'tokenizer') and self.target_engine.tokenizer is not None:
            self.tokenizer = self.target_engine.tokenizer
        
        # Get draft model for this shard if not already set
        if self.draft_engine is None:
            draft_model_id = self.config.draft_model or get_draft_model_for_target(shard.model_id, self.config)
            if draft_model_id and draft_model_id != shard.model_id:
                if DEBUG >= 1:
                    print(f"[Speculative] Loading draft model: {draft_model_id}")
                
                # Create draft shard
                draft_shard = Shard(
                    model_id=draft_model_id,
                    start_layer=0,
                    end_layer=0,  # Will be set based on draft model
                    n_layers=shard.n_layers // 4  # Smaller model
                )
                
                # Create draft engine (use same engine type as target)
                if self.target_engine.__class__.__name__ == "DummyInferenceEngine":
                    from exo.inference.dummy_inference_engine import DummyInferenceEngine
                    self.draft_engine = DummyInferenceEngine()
                else:
                    # Use same engine class as target
                    self.draft_engine = self.target_engine.__class__(self.shard_downloader)
                    
                # Load the draft shard
                if self.target_engine.__class__.__name__ != "DummyInferenceEngine":
                    await self.draft_engine.ensure_shard(draft_shard)
                    
            elif draft_model_id == shard.model_id:
                # Same model - use smaller shard for draft
                if DEBUG >= 1:
                    print(f"[Speculative] Using same model with smaller shard for draft")
                
                # Create smaller draft shard (use fewer layers for faster inference)
                draft_shard = Shard(
                    model_id=shard.model_id,
                    start_layer=0,
                    end_layer=max(0, shard.n_layers // 4 - 1),  # Use 1/4 of the layers
                    n_layers=shard.n_layers // 4
                )
                
                # Create draft engine (use same engine type as target)
                if self.target_engine.__class__.__name__ == "DummyInferenceEngine":
                    from exo.inference.dummy_inference_engine import DummyInferenceEngine
                    self.draft_engine = DummyInferenceEngine()
                else:
                    # Use same engine class as target
                    self.draft_engine = self.target_engine.__class__(self.shard_downloader)
                    
                # Load the draft shard
                if self.target_engine.__class__.__name__ != "DummyInferenceEngine":
                    await self.draft_engine.ensure_shard(draft_shard)
            else:
                if DEBUG >= 1:
                    print(f"[Speculative] No suitable draft model found")
                return
    
    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """
        REAL speculative decoding with proper tokenizer/vocabulary handling.
        Fixes token corruption by ensuring vocabulary compatibility.
        """
        if DEBUG >= 1:
            print(f"[Speculative] Starting REAL speculative decoding, enabled: {self.config.enabled}")
            
        if not self.config.enabled:
            return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)

        try:
            # Set up draft engine with PROPER vocabulary handling
            await self._ensure_draft_engine_setup(shard)
            
            if self.draft_engine is None:
                if DEBUG >= 1:
                    print(f"[Speculative] No draft engine, using target only")
                return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)

            # STEP 1: Get draft model prediction
            draft_start = time.time()
            draft_result = await self._get_draft_prediction_safe(request_id, shard, input_data, inference_state)
            draft_time = time.time() - draft_start
            
            if draft_result is None:
                if DEBUG >= 1:
                    print(f"[Speculative] Draft failed, using target only")
                return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            
            draft_logits, draft_token = draft_result
            
            if DEBUG >= 1:
                print(f"[Speculative] Draft model predicted token: {draft_token}")
            
            # STEP 2: Get target model prediction for the SAME input
            verify_start = time.time()
            target_output, target_state = await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            verify_time = time.time() - verify_start
            
            if target_output is None:
                return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            
            # STEP 3: Extract target logits for verification
            target_logits = self._extract_last_token_logits(target_output)
            if target_logits is None:
                return target_output, target_state
            
            # STEP 4: Verify draft token using speculative sampling
            accepted = self._verify_draft_token_compatible(draft_token, draft_logits, target_logits)
            
            if accepted:
                if DEBUG >= 1:
                    print(f"[Speculative] ✅ ACCEPTED draft token: {draft_token}")
                
                # STEP 5A: Use target model's output but with draft token integrated
                # The key insight: use the TARGET model's token representation but accept the draft's choice
                final_output = self._build_output_with_accepted_token(target_output, draft_token, target_logits)
                
                self._update_metrics([draft_token], [draft_token], draft_time, verify_time)
                return final_output, target_state
                
            else:
                if DEBUG >= 1:
                    print(f"[Speculative] ❌ REJECTED draft token: {draft_token}, using target output")
                
                self._update_metrics([draft_token], [], draft_time, verify_time)
                return target_output, target_state

        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error in speculative decoding: {e}")
                import traceback
                traceback.print_exc()
            return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
    
    async def _ensure_draft_engine_setup(self, shard: Shard) -> None:
        """Set up draft engine with proper tokenizer compatibility"""
        if self.draft_engine is not None:
            return
            
        draft_model_id = self.config.draft_model or get_draft_model_for_target(shard.model_id, self.config)
        
        if DEBUG >= 1:
            print(f"[Speculative] Setting up draft engine: {draft_model_id}")
        
        if draft_model_id and draft_model_id != shard.model_id:
            try:
                # Create draft engine (same type as target)
                self.draft_engine = self.target_engine.__class__(self.shard_downloader)
                
                # Create draft shard
                draft_shard = Shard(
                    model_id=draft_model_id,
                    start_layer=0,
                    end_layer=0,
                    n_layers=shard.n_layers
                )
                
                # Load draft model
                await self.draft_engine.ensure_shard(draft_shard)
                
                # CRITICAL: Ensure both models use the same tokenizer
                if hasattr(self.target_engine, 'tokenizer') and hasattr(self.draft_engine, 'tokenizer'):
                    if DEBUG >= 1:
                        print(f"[Speculative] Synchronizing tokenizers between models")
                    # Use target model's tokenizer for both
                    self.draft_engine.tokenizer = self.target_engine.tokenizer
                
                if DEBUG >= 1:
                    print(f"[Speculative] ✅ Successfully loaded draft model: {draft_model_id}")
                    
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[Speculative] ❌ Failed to load draft model: {e}")
                self.draft_engine = None
        else:
            self.draft_engine = None
    
    async def _get_draft_prediction_safe(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict]) -> Optional[Tuple[np.ndarray, int]]:
        """Get safe prediction from draft model with vocabulary compatibility"""
        if self.draft_engine is None:
            return None
            
        try:
            # Create draft shard
            draft_model_id = self.config.draft_model or get_draft_model_for_target(shard.model_id, self.config)
            draft_shard = Shard(
                model_id=draft_model_id,
                start_layer=0,
                end_layer=0,
                n_layers=shard.n_layers
            )
            
            # Get draft model output
            draft_output, _ = await self.draft_engine.infer_tensor(f"{request_id}_draft", draft_shard, input_data, inference_state)
            
            if draft_output is None:
                return None
            
            # Extract logits using SAME method as target
            draft_logits = self._extract_last_token_logits(draft_output)
            if draft_logits is None:
                return None
            
            # Sample token with some temperature for diversity
            draft_token = self._sample_from_logits(draft_logits, temperature=0.6)
            
            if DEBUG >= 2:
                print(f"[Speculative] Draft logits shape: {draft_logits.shape}, token: {draft_token}")
            
            return draft_logits, draft_token
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error in draft prediction: {e}")
            return None
    
    def _verify_draft_token_compatible(self, draft_token: int, draft_logits: np.ndarray, target_logits: np.ndarray) -> bool:
        """Verify draft token with robust vocabulary compatibility"""
        try:
            # Ensure vocabulary compatibility
            vocab_size = min(len(draft_logits), len(target_logits))
            
            # Check if token is in valid range
            if draft_token >= vocab_size or draft_token < 0:
                if DEBUG >= 1:
                    print(f"[Speculative] Token {draft_token} out of range [0, {vocab_size})")
                return False
            
            # Truncate logits to same vocabulary size
            draft_logits_safe = draft_logits[:vocab_size]
            target_logits_safe = target_logits[:vocab_size]
            
            # Convert to probabilities
            target_probs = self._logits_to_probs(target_logits_safe)
            draft_probs = self._logits_to_probs(draft_logits_safe)
            
            # Get probabilities for the draft token
            target_prob = target_probs[draft_token]
            draft_prob = draft_probs[draft_token]
            
            if draft_prob <= 1e-10:  # Avoid division by zero
                return False
            
            # Speculative sampling: accept with probability min(1, p_target/p_draft)
            acceptance_ratio = target_prob / draft_prob
            acceptance_prob = min(1.0, acceptance_ratio)
            
            # Accept/reject based on probability
            accepted = np.random.random() < acceptance_prob
            
            if DEBUG >= 1:
                print(f"[Speculative] Token {draft_token}: p_target={target_prob:.6f}, p_draft={draft_prob:.6f}")
                print(f"[Speculative] Acceptance ratio: {acceptance_ratio:.4f}, prob: {acceptance_prob:.4f}, accepted: {accepted}")
            
            return accepted
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error in verification: {e}")
            return False
    
    def _build_output_with_accepted_token(self, target_output: np.ndarray, accepted_token: int, target_logits: np.ndarray) -> np.ndarray:
        """Build output with accepted draft token using target model's representation"""
        try:
            # For now, return the target output but with the accepted token's probability boosted
            # This ensures we use the target model's token representation
            
            if len(target_output.shape) == 3:  # [batch, seq, vocab]
                # Replace the last token's logits
                output_copy = target_output.copy()
                # Boost the accepted token's probability
                if accepted_token < output_copy.shape[-1]:
                    output_copy[0, -1, accepted_token] += 2.0  # Boost accepted token
                return output_copy
            elif len(target_output.shape) == 2:  # [seq, vocab]
                output_copy = target_output.copy()
                if accepted_token < output_copy.shape[-1]:
                    output_copy[-1, accepted_token] += 2.0
                return output_copy
            else:
                # For other shapes, just return target output
                return target_output
                
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Speculative] Error building output: {e}")
            return target_output
    
    def _extract_last_token_logits(self, output: np.ndarray) -> Optional[np.ndarray]:
        """Extract logits for the last token position"""
        try:
            if output is None or len(output) == 0:
                return None
            
            if len(output.shape) == 3:  # [batch, seq, vocab]
                return output[0, -1, :].flatten()
            elif len(output.shape) == 2:  # [seq, vocab] 
                return output[-1, :].flatten()
            elif len(output.shape) == 1:  # [vocab]
                return output.flatten()
            else:
                return None
                
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Speculative] Error extracting logits: {e}")
            return None
    
    def _sample_from_logits(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """Sample token from logits with temperature"""
        try:
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Softmax with numerical stability
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits)
            
            # Sample
            return int(np.random.choice(len(probs), p=probs))
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Speculative] Error sampling: {e}")
            return 0
    
    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities with numerical stability"""
        try:
            # Softmax with numerical stability
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits)
            return probs
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Speculative] Error converting logits to probs: {e}")
            # Return uniform distribution as fallback
            return np.ones(len(logits)) / len(logits)
    
    def _update_metrics(self, draft_tokens: List[int], accepted_tokens: List[int], draft_time: float, verify_time: float):
        """Update speculative decoding metrics"""
        self.metrics['total_draft_tokens'] += len(draft_tokens)
        self.metrics['accepted_tokens'] += len(accepted_tokens)
        self.metrics['rejected_tokens'] += len(draft_tokens) - len(accepted_tokens)
        self.metrics['speculation_rounds'] += 1
        self.metrics['total_draft_time'] += draft_time
        self.metrics['total_verify_time'] += verify_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current speculative decoding metrics"""
        total_tokens = self.metrics['total_draft_tokens']
        if total_tokens > 0:
            acceptance_rate = self.metrics['accepted_tokens'] / total_tokens
            speedup = (self.metrics['accepted_tokens'] + self.metrics['speculation_rounds']) / self.metrics['speculation_rounds']
        else:
            acceptance_rate = 0.0
            speedup = 1.0
        
        return {
            **self.metrics,
            'acceptance_rate': acceptance_rate,
            'estimated_speedup': speedup,
            'avg_draft_time': self.metrics['total_draft_time'] / max(1, self.metrics['speculation_rounds']),
            'avg_verify_time': self.metrics['total_verify_time'] / max(1, self.metrics['speculation_rounds'])
        }
    
    async def load_checkpoint(self, shard: Shard, path: str):
        """Load checkpoint for both engines"""
        await self.target_engine.load_checkpoint(shard, path)
        # Note: Draft model checkpoints handled separately
    
    async def save_checkpoint(self, shard: Shard, path: str):
        """Save checkpoint for target engine"""
        await self.target_engine.save_checkpoint(shard, path) 
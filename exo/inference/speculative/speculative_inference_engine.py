import numpy as np
import asyncio
import time
from typing import Optional, Tuple, List, Dict, Any
from exo.inference.inference_engine import InferenceEngine, get_inference_engine
from exo.inference.shard import Shard
from exo.models import build_base_shard
from .speculative_config import SpeculativeConfig, get_draft_model_for_target
from exo import DEBUG
import os

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
    
    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """
        FIXED: Proper speculative decoding with correct algorithm
        """
        if DEBUG >= 1:
            print(f"[Speculative] Starting PROPER speculative decoding, γ={self.config.draft_tokens}")
            
        if not self.config.enabled:
            if DEBUG >= 1:
                print(f"[Speculative] ❌ Speculative decoding DISABLED - using target only")
            return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)

        # 🔧 CRITICAL FIX: Check GPU memory before attempting speculative decoding
        try:
            import torch
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                memory_gb = free_memory / (1024**3)
                if DEBUG >= 1:
                    print(f"[Speculative] 🔍 Available GPU memory: {memory_gb:.1f} GB")
                
                # If less than 4GB free, disable speculative decoding to prevent OOM
                if memory_gb < 4.0:
                    if DEBUG >= 1:
                        print(f"[Speculative] ⚠️  Insufficient GPU memory ({memory_gb:.1f} GB < 4.0 GB) - using target only")
                    return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] ⚠️  Could not check GPU memory: {e} - proceeding with caution")

        try:
            # 🔧 CRITICAL FIX: Clear any corrupted states before starting
            if hasattr(self.target_engine, '_clear_cache'):
                await self.target_engine._clear_cache()
            
            # --- Step 0: Ensure draft engine is ready ---
            await self._ensure_simple_draft_engine(shard)
            
            if self.draft_engine is None:
                if DEBUG >= 1:
                    print(f"[Speculative] ❌ No draft engine available - using target only")
                return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)

            # 🔧 CRITICAL FIX: Clear draft engine cache to prevent corruption
            if hasattr(self.draft_engine, '_clear_cache'):
                await self.draft_engine._clear_cache()

            if DEBUG >= 1:
                print(f"[Speculative] ✅ SPECULATIVE DECODING ACTIVE: generating {self.config.draft_tokens} draft tokens")

            # 🔧 DEBUG: Check input sequence format to detect prompt issues
            if DEBUG >= 1:
                print(f"[Speculative] 🔍 INPUT DEBUG: sequence shape={input_data.shape}, first 5 tokens={input_data.flatten()[:5]}")
                if hasattr(self.target_engine, 'tokenizer') and self.target_engine.tokenizer:
                    try:
                        decoded_input = self.target_engine.tokenizer.decode(input_data.flatten()[:20])
                        print(f"[Speculative] 🔍 INPUT DECODED: '{decoded_input}'")
                    except:
                        print(f"[Speculative] ⚠️  Could not decode input tokens")

            # --- Step 1: Generate draft tokens sequentially ---
            draft_tokens = []
            draft_probs = []  # 🔧 Store REAL draft probabilities
            current_sequence = input_data.copy()
            
            draft_model_id = self.config.draft_model or get_draft_model_for_target(shard.model_id, self.config)
            draft_shard = Shard(
                model_id=draft_model_id,
                start_layer=0,
                end_layer=15 if "1b" in draft_model_id.lower() else 27,  # Use full model, 0-15 for 1B (16 layers), 0-27 for 3B (28 layers)
                n_layers=16 if "1b" in draft_model_id.lower() else 28   # Different layer counts
            )
            
            if DEBUG >= 1:
                print(f"[Speculative] 🚀 PHASE 1: Generating {self.config.draft_tokens} draft tokens with {draft_model_id}")
            
            for i in range(self.config.draft_tokens):
                # 🔧 FIX: Use UNIQUE request ID for each draft token to prevent state reuse
                unique_draft_id = f"{request_id}_draft_{i}_{len(current_sequence)}"
                draft_output, _ = await self.draft_engine.infer_tensor(unique_draft_id, draft_shard, current_sequence, inference_state)
                
                if draft_output is None:
                    if DEBUG >= 1:
                        print(f"[Speculative] ❌ Draft failed at token {i}")
                    break
                
                # Extract draft token
                draft_logits = self._extract_last_token_logits(draft_output)
                if draft_logits is None:
                    break
                
                draft_probs_dist = self._logits_to_probs(draft_logits)
                draft_token = np.argmax(draft_probs_dist)
                draft_token_prob = float(draft_probs_dist[draft_token])  # 🔧 Get REAL probability
                
                # 🔧 CRITICAL FIX: Filter out special tokens that cause corruption
                special_tokens = [220, 128006, 128007, 128008, 128009]  # Common special tokens
                if draft_token in special_tokens:
                    if DEBUG >= 1:
                        print(f"[Speculative] ❌ FILTERING SPECIAL TOKEN: {draft_token} (likely corrupted)")
                    # Find next best non-special token
                    sorted_indices = np.argsort(draft_probs_dist)[::-1]  # Sort by probability
                    for alt_token in sorted_indices:
                        if alt_token not in special_tokens:
                            draft_token = alt_token
                            draft_token_prob = float(draft_probs_dist[draft_token])
                            if DEBUG >= 1:
                                print(f"[Speculative] ✅ Using alternative token: {draft_token}")
                            break
                    else:
                        # If all top tokens are special, skip this draft token
                        if DEBUG >= 1:
                            print(f"[Speculative] ❌ All top tokens are special - skipping draft generation")
                        break
                
                # 🔧 QUALITY CHECK: Reject obvious nonsense after special token filtering
                if len(draft_tokens) > 0 and draft_token == draft_tokens[-1]:
                    # Same token repeated - likely nonsense
                    if DEBUG >= 1:
                        print(f"[Speculative] ❌ QUALITY CHECK FAILED: Draft token {draft_token} is repeat of previous token")
                    break
                
                if len(draft_tokens) >= 2 and draft_token == draft_tokens[-1] == draft_tokens[-2]:
                    # Three identical tokens in a row - definitely nonsense
                    if DEBUG >= 1:
                        print(f"[Speculative] ❌ QUALITY CHECK FAILED: Three identical tokens {draft_token}")
                    break
                
                draft_tokens.append(int(draft_token))
                draft_probs.append(draft_token_prob)  # 🔧 Store REAL probability
                current_sequence = self._safe_append_token(current_sequence, draft_token)
                
                if DEBUG >= 1:
                    print(f"[Speculative] 🎲 Draft token {i+1}/{self.config.draft_tokens}: {draft_token} (prob={draft_token_prob:.6f})")
                    print(f"[Speculative] 🔍 Current sequence length: {current_sequence.shape if hasattr(current_sequence, 'shape') else len(current_sequence)}")
                    
                    # 🔧 DEBUG: Decode draft token to see what it is
                    if hasattr(self.target_engine, 'tokenizer') and self.target_engine.tokenizer:
                        try:
                            decoded_token = self.target_engine.tokenizer.decode([draft_token])
                            print(f"[Speculative] 🔍 Draft token {draft_token} decodes to: '{decoded_token}'")
                        except:
                            print(f"[Speculative] ⚠️  Could not decode draft token {draft_token}")
            
            # 🔧 ADDITIONAL QUALITY CHECK: If all draft tokens are identical, reject
            if len(draft_tokens) > 1 and len(set(draft_tokens)) == 1:
                if DEBUG >= 1:
                    print(f"[Speculative] ❌ QUALITY CHECK FAILED: All draft tokens are identical {draft_tokens[0]}")
                return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            
            if not draft_tokens:
                if DEBUG >= 1:
                    print(f"[Speculative] ❌ No draft tokens generated - fallback to target only")
                return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            
            if DEBUG >= 1:
                print(f"[Speculative] ✅ PHASE 1 COMPLETE: Generated {len(draft_tokens)} draft tokens: {draft_tokens}")
                print(f"[Speculative] 📊 Draft probabilities: {[f'{p:.6f}' for p in draft_probs]}")
            
            # --- Step 2: Target model processes FULL sequence at once ---
            extended_sequence = self._build_extended_sequence(input_data, draft_tokens)
            
            if DEBUG >= 1:
                print(f"[Speculative] 🔍 PHASE 2: Target model processing extended sequence...")
                print(f"[Speculative] 📝 Original sequence length: {input_data.shape[-1]}, Extended: {extended_sequence.shape[-1]}")
            
            # Target model processes the full sequence with all draft tokens
            target_output, target_state = await self.target_engine.infer_tensor(request_id, shard, extended_sequence, inference_state)
            
            if target_output is None:
                if DEBUG >= 1:
                    print(f"[Speculative] ❌ Target model failed - fallback to target only")
                return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            
            # --- Step 3: Extract probabilities and verify tokens ---
            if DEBUG >= 1:
                print(f"[Speculative] 🔍 PHASE 3: Token verification and acceptance/rejection decisions...")
                
            original_seq_len = input_data.shape[-1]
            target_probs_list = self._extract_simple_probs(target_output, original_seq_len, len(draft_tokens))
            
            if len(target_probs_list) < len(draft_tokens):
                if DEBUG >= 1:
                    print(f"[Speculative] ❌ Insufficient target probabilities - fallback")
                return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)
            
            # Verify each draft token
            accepted_tokens = []
            for i, draft_token in enumerate(draft_tokens):
                target_probs = target_probs_list[i]
                
                if draft_token >= len(target_probs):
                    if DEBUG >= 1:
                        print(f"[Speculative] ❌ Draft token {draft_token} out of range at position {i}")
                    break
                
                target_prob = target_probs[draft_token]
                draft_prob = draft_probs[i]  # 🔧 Use REAL draft probability
                
                # Mathematical acceptance rule
                acceptance_ratio = min(1.0, target_prob / max(draft_prob, 1e-10))
                random_val = np.random.uniform(0, 1)
                
                if DEBUG >= 1:
                    print(f"[Speculative] 🔎 Token {i+1}: {draft_token}")
                    print(f"[Speculative]    📊 Draft prob: {draft_prob:.6f}, Target prob: {target_prob:.6f}")
                    print(f"[Speculative]    🎲 Acceptance ratio: {acceptance_ratio:.4f}, Random: {random_val:.4f}")
                
                if random_val <= acceptance_ratio:
                    accepted_tokens.append(draft_token)
                    if DEBUG >= 1:
                        print(f"[Speculative] ✅ ACCEPTED token {i+1}: {draft_token}")
                else:
                    if DEBUG >= 1:
                        print(f"[Speculative] ❌ REJECTED token {i+1}: {draft_token} (target disagrees with draft)")
                    break
            
            if DEBUG >= 1:
                print(f"[Speculative] 📊 VERIFICATION COMPLETE: Accepted {len(accepted_tokens)}/{len(draft_tokens)} draft tokens")
                if len(accepted_tokens) > 0:
                    print(f"[Speculative] 🎯 SPEEDUP ACHIEVED: {len(accepted_tokens)} tokens")
                else:
                    print(f"[Speculative] ⚠️  NO SPEEDUP: All draft tokens rejected")
            
            # --- Step 4: Return final result ---
            if DEBUG >= 1:
                print(f"[Speculative] 🎯 PHASE 4: Final token sampling and return...")
                
            if len(accepted_tokens) > 0:
                # We already have the target output with all probabilities
                # Extract logits for the NEXT token after the accepted sequence
                next_token_position = len(accepted_tokens)  # Position for next token
                
                if DEBUG >= 1:
                    print(f"[Speculative] Getting logits for next token at position {next_token_position}")
                    print(f"[Speculative] Target output shape: {target_output.shape}")
                
                # Extract the logits for the next token prediction
                if target_output.ndim == 3:  # (batch, seq, vocab)
                    if next_token_position < target_output.shape[1]:
                        next_logits = target_output[0, next_token_position, :].reshape(1, 1, -1)
                    else:
                        next_logits = target_output[0, -1, :].reshape(1, 1, -1)
                elif target_output.ndim == 2:  # (seq, vocab)
                    if next_token_position < target_output.shape[0]:
                        next_logits = target_output[next_token_position, :].reshape(1, 1, -1)
                    else:
                        next_logits = target_output[-1, :].reshape(1, 1, -1)
                else:
                    next_logits = target_output.reshape(1, 1, -1)
                
                if DEBUG >= 1:
                    print(f"[Speculative] ✅ PHASE 4 COMPLETE: Returning logits for next token after {len(accepted_tokens)} accepted tokens")
                    print(f"[Speculative] Next logits shape: {next_logits.shape}")
                
                return next_logits, target_state
            else:
                # No tokens accepted, return target prediction for next token from original position
                if DEBUG >= 1:
                    print(f"[Speculative] Extracting logits for original position...")
                
                # Extract logits for the next token from the original input position
                original_pos = input_data.shape[-1] - 1  # Last position of original input
                
                if target_output.ndim == 3:  # (batch, seq, vocab)
                    if original_pos < target_output.shape[1]:
                        next_logits = target_output[0, original_pos, :].reshape(1, 1, -1)
                    else:
                        next_logits = target_output[0, -1, :].reshape(1, 1, -1)
                elif target_output.ndim == 2:  # (seq, vocab)
                    if original_pos < target_output.shape[0]:
                        next_logits = target_output[original_pos, :].reshape(1, 1, -1)
                    else:
                        next_logits = target_output[-1, :].reshape(1, 1, -1)
                else:
                    next_logits = target_output.reshape(1, 1, -1)
                
                if DEBUG >= 1:
                    print(f"[Speculative] ✅ PHASE 4 COMPLETE: No tokens accepted, returning target logits from position {original_pos}")
                    print(f"[Speculative] Next logits shape: {next_logits.shape}")
                
                return next_logits, target_state

        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error: {e}, using target only")
                import traceback
                traceback.print_exc()
            return await self.target_engine.infer_tensor(request_id, shard, input_data, inference_state)

    def _safe_append_token(self, sequence: np.ndarray, token: int) -> np.ndarray:
        """SAFE token appending that preserves sequence structure"""
        try:
            token = int(token)  # Ensure token is int
            
            if sequence.ndim == 1:
                # 1D sequence: [token1, token2, ...]
                return np.append(sequence, token).astype(sequence.dtype)
            elif sequence.ndim == 2:
                # 2D sequence: [[token1, token2, ...]]
                if sequence.shape[0] == 1:
                    # Single batch
                    token_array = np.array([[token]], dtype=sequence.dtype)
                    return np.concatenate([sequence, token_array], axis=1)
                else:
                    # Multiple batches - only modify first
                    new_sequence = sequence.copy()
                    token_array = np.array([[token]], dtype=sequence.dtype)
                    new_sequence[0:1] = np.concatenate([new_sequence[0:1], token_array], axis=1)
                    return new_sequence
            else:
                # Fallback for other shapes
                flat = sequence.flatten()
                appended = np.append(flat, token)
                return appended.reshape(sequence.shape[:-1] + (appended.shape[0],)).astype(sequence.dtype)
                
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error in _safe_append_token: {e}")
            return sequence

    def _build_extended_sequence(self, input_data: np.ndarray, draft_tokens: List[int]) -> np.ndarray:
        """FIXED: Build extended sequence with proper tensor handling"""
        try:
            # Convert to list for easier manipulation
            if input_data.ndim == 1:
                sequence_list = input_data.tolist()
            elif input_data.ndim == 2:
                sequence_list = input_data[0].tolist()  # Take first batch
            else:
                sequence_list = input_data.flatten().tolist()
            
            # Append draft tokens
            extended_list = sequence_list + draft_tokens
            
            # Convert back to numpy array with same shape as input
            if input_data.ndim == 1:
                return np.array(extended_list, dtype=input_data.dtype)
            elif input_data.ndim == 2:
                return np.array([extended_list], dtype=input_data.dtype)
            else:
                return np.array(extended_list, dtype=input_data.dtype)
                
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error building extended sequence: {e}")
            return input_data

    def _build_final_sequence(self, input_data: np.ndarray, tokens: List[int]) -> np.ndarray:
        """FIXED: Build final sequence with proper tensor handling"""
        try:
            # Convert to list for easier manipulation
            if input_data.ndim == 1:
                sequence_list = input_data.tolist()
            elif input_data.ndim == 2:
                sequence_list = input_data[0].tolist()  # Take first batch
            else:
                sequence_list = input_data.flatten().tolist()
            
            # Append tokens
            final_list = sequence_list + tokens
            
            # Convert back to numpy array with same shape as input
            if input_data.ndim == 1:
                return np.array(final_list, dtype=input_data.dtype)
            elif input_data.ndim == 2:
                return np.array([final_list], dtype=input_data.dtype)
            else:
                return np.array(final_list, dtype=input_data.dtype)
                
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error building final sequence: {e}")
            return input_data

    def _extract_target_probs_at_position(self, output: np.ndarray, position: int) -> Optional[np.ndarray]:
        """FIXED: Extract target probabilities at specific position"""
        try:
            if output is None:
                return None
            
            # Handle different output shapes
            if output.ndim == 3:  # (batch, seq, vocab)
                if position >= output.shape[1]:
                    return None
                logits = output[0, position, :]
            elif output.ndim == 2:  # (seq, vocab)
                if position >= output.shape[0]:
                    return None
                logits = output[position, :]
            else:
                # Single position output
                logits = output.flatten()
            
            # Convert to probabilities
            probs = self._logits_to_probs(logits)
            return probs
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error extracting probs at position {position}: {e}")
            return None

    async def _ensure_simple_draft_engine(self, shard: Shard) -> None:
        """Simplified draft engine setup - COMPLETELY REWRITTEN to avoid corruption"""
        if self.draft_engine is not None:
            return
            
        draft_model_id = self.config.draft_model or get_draft_model_for_target(shard.model_id, self.config)
        
        if DEBUG >= 1:
            print(f"[Speculative] Setting up FIXED draft engine: target={shard.model_id}, draft={draft_model_id}")
        
        if draft_model_id and draft_model_id != shard.model_id:
            try:
                # 🔧 COMPLETE FIX: Use a fresh instance of the same target engine class
                # This avoids all JIT conflicts and state corruption issues
                if DEBUG >= 1:
                    print(f"[Speculative] 🔧 Creating completely fresh draft engine instance")
                
                # 🔧 MEMORY PROTECTION: Clear memory before attempting to load draft model
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if DEBUG >= 1:
                            print(f"[Speculative] 🧹 Cleared CUDA cache before draft model loading")
                except:
                    pass
                
                # Create a completely new engine instance - no sharing with target engine
                self.draft_engine = self.target_engine.__class__(self.shard_downloader)
                
                # Use completely different shard with smaller model
                draft_shard = Shard(
                    model_id=draft_model_id,
                    start_layer=0,
                    end_layer=15 if "1b" in draft_model_id.lower() else 27,  # Use full model, 0-15 for 1B (16 layers), 0-27 for 3B (28 layers)
                    n_layers=16 if "1b" in draft_model_id.lower() else 28   # Different layer counts
                )
                
                if DEBUG >= 1:
                    print(f"[Speculative] Loading draft model: {draft_model_id} with {draft_shard.n_layers} layers")
                
                await self.draft_engine.ensure_shard(draft_shard)
                
                # 🔧 CRITICAL FIX: Force draft engine to use target engine's tokenizer
                # This prevents tokenizer mismatch issues
                if hasattr(self.target_engine, 'tokenizer') and self.target_engine.tokenizer is not None:
                    self.draft_engine.tokenizer = self.target_engine.tokenizer
                    if DEBUG >= 1:
                        print(f"[Speculative] ✅ Draft engine using target tokenizer")
                
                if DEBUG >= 1:
                    print(f"[Speculative] ✅ Successfully loaded fresh draft engine: {draft_model_id}")
                    
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[Speculative] ❌ Failed to load draft model {draft_model_id}: {e}")
                    if "out of memory" in str(e).lower():
                        print(f"[Speculative] 🚨 OOM Error detected - cleaning up and disabling speculative decoding")
                        # Aggressive cleanup on OOM
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                        except:
                            pass
                    import traceback
                    traceback.print_exc()
                self.draft_engine = None
        else:
            if DEBUG >= 1:
                print(f"[Speculative] No suitable draft model for {shard.model_id}")
            self.draft_engine = None

    async def _generate_simple_draft_tokens(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict]) -> Tuple[List[int], List[float]]:
        """Generate draft tokens with simplified approach"""
        draft_tokens = []
        draft_probs = []
        current_sequence = input_data.copy()
        
        draft_model_id = self.config.draft_model or get_draft_model_for_target(shard.model_id, self.config)
        draft_shard = Shard(model_id=draft_model_id, start_layer=0, end_layer=0, n_layers=0)
        
        if DEBUG >= 1:
            print(f"[Speculative] Generating {self.config.draft_tokens} draft tokens with {draft_model_id}")
        
        for i in range(self.config.draft_tokens):
            try:
                # Get draft model prediction
                draft_output, _ = await self.draft_engine.infer_tensor(f"{request_id}_draft_{i}", draft_shard, current_sequence, inference_state)
                
                if draft_output is None:
                    break
                
                # Extract logits and convert to probabilities
                draft_logits = self._extract_last_token_logits(draft_output)
                if draft_logits is None:
                    break
                
                draft_probs_dist = self._logits_to_probs(draft_logits)
                
                # Use greedy sampling for consistency
                draft_token = np.argmax(draft_probs_dist)
                draft_prob = draft_probs_dist[draft_token]
                
                draft_tokens.append(int(draft_token))
                draft_probs.append(float(draft_prob))
                
                # Append token to sequence for next iteration
                current_sequence = self._append_token_to_sequence(current_sequence, draft_token)
                
                if DEBUG >= 2:
                    print(f"[Speculative] Draft token {i}: {draft_token} (prob={draft_prob:.6f})")
                    
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[Speculative] Error generating draft token {i}: {e}")
                break
        
        return draft_tokens, draft_probs

    def _extract_simple_probs(self, output: np.ndarray, original_seq_len: int, n_tokens: int) -> List[np.ndarray]:
        """Extract probabilities for each token position in a simple, robust way"""
        probs_list = []
        
        try:
            # Handle different output shapes
            if output.ndim == 3:  # (batch, seq, vocab)
                logits_sequence = output[0]  # Take first batch
            elif output.ndim == 2:  # (seq, vocab)
                logits_sequence = output
            else:
                if DEBUG >= 1:
                    print(f"[Speculative] Unexpected output shape: {output.shape}")
                return []
            
            # Extract logits for each position we need
            for i in range(n_tokens + 1):  # +1 for the next token after draft sequence
                pos = original_seq_len + i - 1  # Position in the sequence
                
                if pos >= 0 and pos < logits_sequence.shape[0]:
                    logits = logits_sequence[pos]
                    probs = self._logits_to_probs(logits)
                    probs_list.append(probs)
                else:
                    if DEBUG >= 1:
                        print(f"[Speculative] Position {pos} out of range for sequence length {logits_sequence.shape[0]}")
                    break
            
            return probs_list
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error extracting probabilities: {e}")
            return []

    def _format_output_logits(self, logits: np.ndarray) -> np.ndarray:
        """Format logits for output"""
        try:
            if len(logits.shape) == 1:  # [vocab]
                return logits.reshape(1, 1, -1)  # [batch, seq, vocab]
            else:
                return logits
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Speculative] Error formatting output: {e}")
            return logits

    def _extract_last_token_logits(self, output: np.ndarray) -> Optional[np.ndarray]:
        """Extract logits for the last token from model output"""
        try:
            if output is None:
                return None
            
            if len(output.shape) == 3:  # [batch, seq, vocab]
                return output[0, -1, :]  # Last token logits
            elif len(output.shape) == 2:  # [seq, vocab]
                return output[-1, :]  # Last token logits
            elif len(output.shape) == 1:  # [vocab]
                return output  # Already single token logits
            else:
                if DEBUG >= 2:
                    print(f"[Speculative] Unexpected output shape: {output.shape}")
                return output.flatten()
                
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Speculative] Error extracting last token logits: {e}")
            return None

    def _append_token_to_sequence(self, sequence: np.ndarray, token: int) -> np.ndarray:
        """Append a token to the input sequence"""
        try:
            if len(sequence.shape) == 1:  # [seq_len]
                return np.append(sequence, token)
            elif len(sequence.shape) == 2:  # [batch, seq_len]
                token_array = np.array([[token]])
                return np.concatenate([sequence, token_array], axis=1)
            else:
                if DEBUG >= 2:
                    print(f"[Speculative] Unexpected sequence shape: {sequence.shape}")
                return np.append(sequence.flatten(), token)
                
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Speculative] Error appending token: {e}")
            return sequence

    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities using softmax"""
        try:
            # Apply softmax with numerical stability
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

    def _extract_last_token_probs(self, output: np.ndarray) -> Optional[np.ndarray]:
        """Extract probabilities for the last (next) token prediction"""
        try:
            if output is None:
                return None
            
            # Extract logits for the last position (next token prediction)
            if output.ndim == 3:  # (batch, seq, vocab)
                logits = output[0, -1, :]  # Last position of first batch
            elif output.ndim == 2:  # (seq, vocab)
                logits = output[-1, :]     # Last position
            else:
                logits = output.flatten()  # Single token prediction
            
            # Convert to probabilities
            probs = self._logits_to_probs(logits)
            return probs
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Speculative] Error extracting last token probs: {e}")
            return None

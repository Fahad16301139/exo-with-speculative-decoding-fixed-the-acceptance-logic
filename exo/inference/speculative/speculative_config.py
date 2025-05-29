from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding"""
    # Whether to enable speculative decoding
    enabled: bool = False
    # Number of tokens to draft at once
    draft_tokens: int = 4
    # Name of the draft (smaller) model to use
    draft_model: Optional[str] = None
    # Probability threshold for accepting draft tokens
    acceptance_threshold: float = 0.9
    # Maximum number of speculation rounds per inference
    max_speculation_depth: int = 4
    # Whether to adaptively adjust draft tokens
    adaptive_speculation: bool = True
    
    # Enhanced configuration options
    # Sampling temperature for randomness
    temperature: float = 0.7
    # Top-k sampling (None means disabled)
    top_k: Optional[int] = 50
    # Top-p (nucleus) sampling (None means disabled)
    top_p: Optional[float] = 0.9
    # Disable speculative decoding if acceptance rate drops below this
    min_acceptance_rate: float = 0.4
    # Max number of draft tokens to generate in parallel
    max_draft_batch_size: int = 8
    # Enable tree attention for better parallelization
    use_tree_attention: bool = False
    
    # Same-family model optimizations
    # Share embeddings between target and draft models
    use_shared_embeddings: bool = True
    # Use layer skipping if models are from the same family
    use_layer_skipping: bool = True
    # Ratio of layers to use for draft model
    draft_layer_ratio: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'draft_tokens': self.draft_tokens,
            'draft_model': self.draft_model,
            'acceptance_threshold': self.acceptance_threshold,
            'max_speculation_depth': self.max_speculation_depth,
            'adaptive_speculation': self.adaptive_speculation,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'min_acceptance_rate': self.min_acceptance_rate,
            'max_draft_batch_size': self.max_draft_batch_size,
            'use_tree_attention': self.use_tree_attention,
            'use_shared_embeddings': self.use_shared_embeddings,
            'use_layer_skipping': self.use_layer_skipping,
            'draft_layer_ratio': self.draft_layer_ratio
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpeculativeConfig':
        return cls(
            enabled=data.get('enabled', False),
            draft_tokens=data.get('draft_tokens', 4),
            draft_model=data.get('draft_model'),
            acceptance_threshold=data.get('acceptance_threshold', 0.9),
            max_speculation_depth=data.get('max_speculation_depth', 4),
            adaptive_speculation=data.get('adaptive_speculation', True),
            temperature=data.get('temperature', 0.7),
            top_k=data.get('top_k'),
            top_p=data.get('top_p'),
            min_acceptance_rate=data.get('min_acceptance_rate', 0.4),
            max_draft_batch_size=data.get('max_draft_batch_size', 8),
            use_tree_attention=data.get('use_tree_attention', False),
            use_shared_embeddings=data.get('use_shared_embeddings', True),
            use_layer_skipping=data.get('use_layer_skipping', True),
            draft_layer_ratio=data.get('draft_layer_ratio', 0.25)
        )

def get_draft_model_for_target(target_model: str, config: SpeculativeConfig) -> Optional[str]:
    """Get appropriate draft model for target model"""
    
    # If draft model explicitly set, use it
    if config.draft_model:
        return config.draft_model
    
    # 🔧 FIXED: Use completely different models to avoid JIT conflicts
    draft_mapping = {
        "llama-3.1-8b": "llama-3.2-1b",  # Much smaller, different architecture
        "llama-3.2-8b": "llama-3.2-1b", 
        "llama-3.2-3b": "llama-3.2-1b",
        "llama-3.1-70b": "llama-3.2-3b",
        "llama-3.2-70b": "llama-3.2-3b",
        # Add more mappings as needed
    }
    
    draft_model = draft_mapping.get(target_model)
    
    if not draft_model:
        # Fallback: use smallest available model
        return "llama-3.2-1b"
    
    return draft_model

def is_same_family_model(model_id: str) -> bool:
    """Check if model belongs to a family that supports same-family optimization"""
    families = ["llama", "qwen", "mistral", "gemma", "phi"]  # Supported families
    return any(family in model_id.lower() for family in families)

def has_shared_vocabulary(model_id: str) -> bool:
    """Check if model family typically shares vocabulary across sizes"""
    shared_vocab_families = ["llama", "qwen", "mistral", "gemma"]  # Families with shared vocab
    return any(family in model_id.lower() for family in shared_vocab_families)

def get_optimal_layer_ratio(model_id: str) -> float:
    """Get optimal layer ratio for draft model based on target model"""
    if "1b" in model_id.lower():
        return 0.5  # Use half layers for 1B models
    elif "3b" in model_id.lower():
        return 0.33  # Use 1/3 layers for 3B models
    elif "7b" in model_id.lower() or "8b" in model_id.lower():
        return 0.25  # Use 1/4 layers for 7-8B models
    elif "70b" in model_id.lower():
        return 0.1   # Use 1/10 layers for 70B models
    else:
        return 0.25  # Default ratio 
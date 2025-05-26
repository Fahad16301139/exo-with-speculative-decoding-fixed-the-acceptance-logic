from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding"""
    enabled: bool = False
    draft_tokens: int = 2
    draft_model: Optional[str] = None
    acceptance_threshold: float = 0.8
    max_speculation_depth: int = 4
    adaptive_speculation: bool = True
    
    # Enhanced configuration options
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_acceptance_rate: float = 0.3  # Disable if acceptance rate drops below this
    max_draft_batch_size: int = 8  # Maximum number of draft tokens to generate in parallel
    use_tree_attention: bool = False  # Enable tree attention for better parallelization
    
    # Same-family model optimizations
    use_shared_embeddings: bool = True  # Share embeddings between target and draft
    use_layer_skipping: bool = True     # Use layer skipping for same model
    draft_layer_ratio: float = 0.25     # Ratio of layers to use for draft model

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
            draft_tokens=data.get('draft_tokens', 2),
            draft_model=data.get('draft_model'),
            acceptance_threshold=data.get('acceptance_threshold', 0.8),
            max_speculation_depth=data.get('max_speculation_depth', 4),
            adaptive_speculation=data.get('adaptive_speculation', True),
            temperature=data.get('temperature', 1.0),
            top_k=data.get('top_k'),
            top_p=data.get('top_p'),
            min_acceptance_rate=data.get('min_acceptance_rate', 0.3),
            max_draft_batch_size=data.get('max_draft_batch_size', 8),
            use_tree_attention=data.get('use_tree_attention', False),
            use_shared_embeddings=data.get('use_shared_embeddings', True),
            use_layer_skipping=data.get('use_layer_skipping', True),
            draft_layer_ratio=data.get('draft_layer_ratio', 0.25)
        )

def get_draft_model_for_target(target_model: str, config: SpeculativeConfig) -> str:
    """
    Get the appropriate draft model for a target model.
    Prioritizes same-family models with shared vocabulary.
    ALWAYS returns a different/smaller model for proper speculative decoding.
    """
    if config.draft_model:
        return config.draft_model
    
    # Use real smaller models for proper speculative decoding
    model_families = {
        # Llama family - Use smaller models from same family
        "llama-3.2-1b": "dummy",  # Use dummy for 1B (smallest model)
        "llama-3.2-3b": "llama-3.2-1b",  # Use 1B as draft for 3B
        "llama-3.1-8b": "llama-3.2-1b",  # Use 1B as draft for 8B
        "llama-3.1-70b": "llama-3.1-8b",  # Use 8B as draft for 70B
        "llama-3.1-405b": "llama-3.1-8b",  # Use 8B as draft for 405B
        
        # Other families - use dummy for now to avoid similar issues
        "mistral-7b": "dummy",
        "mixtral-8x7b": "dummy",
        "gemma-2b": "dummy",
        "gemma-7b": "dummy",
        "deepseek-coder-1.3b": "dummy",
        "deepseek-coder-6.7b": "dummy",
        "qwen1.5-0.5b": "dummy",
        "qwen1.5-1.8b": "dummy",
        "qwen1.5-4b": "dummy",
        "qwen1.5-7b": "dummy",
        "qwen1.5-14b": "dummy",
        "qwen1.5-32b": "dummy",
        "qwen1.5-72b": "dummy",
        "qwen1.5-110b": "dummy",
        "qwen2-0.5b": "dummy",
        "qwen2-1.5b": "dummy",
        "qwen2-7b": "dummy",
        "qwen2-72b": "dummy",
        "llava-1.5-7b": "dummy",
        "llava-1.5-13b": "dummy",
    }
    
    # Find the best draft model
    for model_name, draft_model in model_families.items():
        if model_name in target_model.lower():
            print(f"[Speculative] Mapping {target_model} -> {draft_model}")
            return draft_model
    
    # Default fallback - always use dummy to avoid issues
    print(f"[Speculative] No specific mapping found for {target_model}, using dummy")
    return "dummy"

def is_same_family_model(model_id: str) -> bool:
    """Check if model belongs to a family that supports same-family optimization"""
    families = ["llama", "qwen", "mistral", "gemma", "phi"]
    return any(family in model_id.lower() for family in families)

def has_shared_vocabulary(model_id: str) -> bool:
    """Check if model family typically shares vocabulary across sizes"""
    shared_vocab_families = ["llama", "qwen", "mistral", "gemma"]
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
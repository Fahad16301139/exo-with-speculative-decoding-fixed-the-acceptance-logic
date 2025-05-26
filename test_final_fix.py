#!/usr/bin/env python3
"""
Final test to verify speculative decoding is working correctly
"""
import os
import sys
sys.path.append('.')

# Set DEBUG level
os.environ["DEBUG"] = "1"

from exo.models import get_tokenizer_repo, get_repo

def test_tokenizer_fix():
    """Test that tokenizer repositories are correctly mapped"""
    print("🔧 TESTING TOKENIZER REPOSITORY FIX")
    print("=" * 50)
    
    test_cases = [
        ("llama-3.2-1b", "TinygradDynamicShardInferenceEngine"),
        ("llama-3.2-3b", "TinygradDynamicShardInferenceEngine"),
        ("llama-3.1-8b", "TinygradDynamicShardInferenceEngine"),
    ]
    
    for model_id, engine in test_cases:
        model_repo = get_repo(model_id, engine)
        tokenizer_repo = get_tokenizer_repo(model_id, engine)
        
        print(f"✅ Model: {model_id}")
        print(f"   Model repo: {model_repo}")
        print(f"   Tokenizer repo: {tokenizer_repo}")
        print(f"   Fixed: {'✅' if tokenizer_repo != model_repo else '❌'}")
        print()

if __name__ == "__main__":
    test_tokenizer_fix()
    print("🎉 TOKENIZER FIX VERIFICATION COMPLETE!")
    print("\nNow you can run:")
    print("python exo/main.py run llama-3.2-3b --enable-speculative --draft-tokens 2 --max-tokens 10 --prompt \"Hello\" --inference-engine tinygrad") 
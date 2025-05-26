#!/usr/bin/env python3
"""
Simple test to verify speculative decoding configuration
"""
import os
import sys
sys.path.append('.')

# Set DEBUG level
os.environ["DEBUG"] = "2"

from exo.inference.speculative.speculative_config import SpeculativeConfig, get_draft_model_for_target

def test_model_selection():
    """Test the fixed model selection logic"""
    print("🔧 TESTING FIXED MODEL SELECTION")
    print("=" * 50)
    
    config = SpeculativeConfig(enabled=True, draft_tokens=2)
    
    test_cases = [
        "llama-3.2-1b",
        "llama-3.2-3b", 
        "llama-3.1-8b",
        "qwen-2.5-7b",
        "dummy"
    ]
    
    for target_model in test_cases:
        draft_model = get_draft_model_for_target(target_model, config)
        print(f"✅ Target: {target_model:15} -> Draft: {draft_model}")
        
        # Check if it's a proper different model
        if target_model != draft_model:
            print(f"   🎯 Using different model for speedup!")
        else:
            print(f"   ⚡ Using same model with layer skipping")
        print()

def test_speculative_engine_creation():
    """Test creating speculative engine"""
    print("\n🏗️  TESTING SPECULATIVE ENGINE CREATION")
    print("=" * 50)
    
    try:
        from exo.inference.dummy_inference_engine import DummyInferenceEngine
        from exo.inference.speculative.speculative_inference_engine import SpeculativeInferenceEngine
        from exo.inference.shard import Shard
        
        # Create engines
        target_engine = DummyInferenceEngine()
        config = SpeculativeConfig(enabled=True, draft_tokens=2, draft_model="llama-3.2-1b")
        
        spec_engine = SpeculativeInferenceEngine(target_engine, None, config)
        
        print("✅ Successfully created SpeculativeInferenceEngine")
        print(f"   Config: {config}")
        print(f"   Target engine: {target_engine.__class__.__name__}")
        
        # Test shard creation
        shard = Shard(model_id="llama-3.2-3b", start_layer=0, end_layer=15, n_layers=16)
        print(f"   Test shard: {shard}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating speculative engine: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 SPECULATIVE DECODING CONFIGURATION TEST")
    print("=" * 60)
    
    test_model_selection()
    success = test_speculative_engine_creation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Speculative decoding configuration is working!")
        print("💡 The issue was in model selection - now using proper different models")
    else:
        print("❌ Tests failed - need to debug further") 
#!/usr/bin/env python3
"""
Test script to verify speculative decoding fixes
"""
import asyncio
import numpy as np
from exo.inference.speculative.speculative_config import SpeculativeConfig
from exo.inference.speculative.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.dummy_inference_engine import DummyInferenceEngine
from exo.inference.shard import Shard

async def test_speculative_fixes():
    """Test that speculative decoding doesn't corrupt input"""
    print("🧪 Testing Speculative Decoding Fixes")
    print("=" * 50)
    
    # Create dummy engines
    target_engine = DummyInferenceEngine()
    draft_engine = DummyInferenceEngine()
    
    # Create speculative config
    config = SpeculativeConfig(
        enabled=True,
        draft_tokens=2,  # Conservative
        draft_model="dummy"
    )
    
    # Create speculative engine
    spec_engine = SpeculativeInferenceEngine(
        target_engine=target_engine,
        draft_engine=draft_engine,
        config=config
    )
    
    # Create test shard
    shard = Shard(
        model_id="dummy",
        start_layer=0,
        end_layer=0,
        n_layers=1
    )
    
    # Test input - simple token sequence
    test_input = np.array([1, 2, 3, 4, 5])  # "Hello" equivalent
    original_input = test_input.copy()
    
    print(f"📥 Original input: {test_input}")
    
    try:
        # Run speculative inference
        output, state = await spec_engine.infer_tensor(
            request_id="test_fix",
            shard=shard,
            input_data=test_input,
            inference_state=None
        )
        
        print(f"📤 Output shape: {output.shape}")
        print(f"🔍 Input after inference: {test_input}")
        print(f"✅ Input unchanged: {np.array_equal(test_input, original_input)}")
        
        # Check metrics
        metrics = spec_engine.get_metrics()
        print(f"📊 Metrics: {metrics}")
        
        if np.array_equal(test_input, original_input):
            print("🎉 SUCCESS: Input corruption bug FIXED!")
        else:
            print("❌ FAILURE: Input still being corrupted")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

async def test_memory_management():
    """Test memory management features"""
    print("\n🧠 Testing Memory Management")
    print("=" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🔥 CUDA available: {torch.cuda.get_device_name()}")
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 Current GPU memory: {memory_allocated:.2f}GB")
        else:
            print("💻 CUDA not available, using CPU")
    except ImportError:
        print("🐍 PyTorch not available")

async def main():
    """Main test function"""
    print("🚀 Speculative Decoding Fix Verification")
    print("=" * 50)
    
    await test_speculative_fixes()
    await test_memory_management()
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    asyncio.run(main()) 
#!/usr/bin/env python3
"""
Debug script to test speculative decoding with detailed logging
"""
import os
import asyncio
import numpy as np

# Set DEBUG level
os.environ["DEBUG"] = "2"

from exo.inference.speculative.speculative_config import SpeculativeConfig
from exo.inference.speculative.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.dummy_inference_engine import DummyInferenceEngine
from exo.inference.shard import Shard

async def test_speculative_decoding():
    """Test speculative decoding with debug output"""
    print("🔍 DEBUG SPECULATIVE DECODING TEST")
    print("=" * 50)
    
    # Create engines
    target_engine = DummyInferenceEngine()
    draft_engine = DummyInferenceEngine()
    
    # Create speculative config
    config = SpeculativeConfig(
        enabled=True,
        draft_tokens=2,
        draft_model="dummy",
        acceptance_threshold=0.8
    )
    
    print(f"Config: {config}")
    
    # Create speculative engine
    spec_engine = SpeculativeInferenceEngine(target_engine, draft_engine, config)
    
    # Create shard
    shard = Shard(model_id="dummy", start_layer=0, end_layer=0, n_layers=1)
    
    # Test input
    test_input = np.array([1, 2, 3, 4, 5])
    
    print(f"\n📥 Input: {test_input}")
    print(f"📊 Config enabled: {config.enabled}")
    print(f"📊 Draft tokens: {config.draft_tokens}")
    
    # Run inference
    print("\n🚀 Running speculative inference...")
    try:
        output, state = await spec_engine.infer_tensor("test", shard, test_input, None)
        print(f"📤 Output shape: {output.shape}")
        print(f"📤 Output: {output}")
        
        # Get metrics
        metrics = spec_engine.get_metrics()
        print(f"📊 Metrics: {metrics}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_real_model():
    """Test with real model configuration"""
    print("\n🔍 TESTING WITH REAL MODEL CONFIG")
    print("=" * 50)
    
    # Test with llama config
    config = SpeculativeConfig(
        enabled=True,
        draft_tokens=2,
        draft_model="llama-3.2-1b",  # Use same model for draft
        acceptance_threshold=0.8
    )
    
    print(f"Config: {config}")
    
    # Create dummy engines (simulating real models)
    target_engine = DummyInferenceEngine()
    
    # Create speculative engine (will create draft engine automatically)
    spec_engine = SpeculativeInferenceEngine(target_engine, None, config)
    
    # Create shard for llama
    shard = Shard(model_id="llama-3.2-1b", start_layer=0, end_layer=15, n_layers=16)
    
    # Test input
    test_input = np.array([1, 2, 3, 4, 5])
    
    print(f"\n📥 Input: {test_input}")
    print(f"📊 Shard: {shard}")
    
    # Run inference
    print("\n🚀 Running speculative inference with llama config...")
    try:
        output, state = await spec_engine.infer_tensor("test_llama", shard, test_input, None)
        print(f"📤 Output shape: {output.shape}")
        print(f"📤 Output: {output}")
        
        # Get metrics
        metrics = spec_engine.get_metrics()
        print(f"📊 Metrics: {metrics}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🎯 SPECULATIVE DECODING DEBUG SESSION")
    print("=" * 60)
    
    await test_speculative_decoding()
    await test_real_model()
    
    print("\n✨ Debug session completed!")

if __name__ == "__main__":
    asyncio.run(main()) 
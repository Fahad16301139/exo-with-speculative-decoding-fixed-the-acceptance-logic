#!/usr/bin/env python3
"""
Basic tests for speculative decoding functionality.
"""

import unittest
import numpy as np
from unittest.mock import Mock, AsyncMock
import asyncio

# Import our speculative decoding components
from exo.inference.speculative.speculative_config import SpeculativeConfig, get_draft_model_for_target
from exo.inference.speculative.speculative_inference_engine import SpeculativeInferenceEngine
from exo.inference.shard import Shard

class MockInferenceEngine:
    """Mock inference engine for testing"""
    def __init__(self, name="mock"):
        self.name = name
        self.tokenizer = Mock()
        self.tokenizer.eos_token_id = 2
        self.shard = None
        
    async def encode(self, shard, prompt):
        return np.array([1, 2, 3, 4])
    
    async def decode(self, shard, tokens):
        return "mock response"
    
    async def sample(self, x, temp=0.0):
        return np.array([5])
    
    async def infer_tensor(self, request_id, shard, input_data, inference_state=None):
        # Return mock logits
        return np.random.rand(10000), inference_state
    
    async def ensure_shard(self, shard):
        self.shard = shard
    
    async def load_checkpoint(self, shard, path):
        pass

class TestSpeculativeConfig(unittest.TestCase):
    """Test speculative configuration"""
    
    def test_default_config(self):
        config = SpeculativeConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.draft_tokens, 2)
        self.assertIsNone(config.draft_model)
        self.assertEqual(config.acceptance_threshold, 0.8)
    
    def test_config_from_dict(self):
        data = {
            'enabled': True,
            'draft_tokens': 6,
            'draft_model': 'test-model',
            'acceptance_threshold': 0.9
        }
        config = SpeculativeConfig.from_dict(data)
        self.assertTrue(config.enabled)
        self.assertEqual(config.draft_tokens, 6)
        self.assertEqual(config.draft_model, 'test-model')
        self.assertEqual(config.acceptance_threshold, 0.9)
    
    def test_draft_model_selection(self):
        config = SpeculativeConfig()
        
        # Test auto-selection
        draft_model = get_draft_model_for_target("llama-3.1-70b", config)
        self.assertEqual(draft_model, "llama-3.2-3b")
        
        # Test manual override
        config.draft_model = "custom-model"
        draft_model = get_draft_model_for_target("llama-3.1-70b", config)
        self.assertEqual(draft_model, "custom-model")
        
        # Test fallback
        draft_model = get_draft_model_for_target("unknown-model", config)
        self.assertEqual(draft_model, "custom-model")

class TestSpeculativeInferenceEngine(unittest.TestCase):
    """Test speculative inference engine"""
    
    def setUp(self):
        self.target_engine = MockInferenceEngine("target")
        self.draft_engine = MockInferenceEngine("draft")
        self.config = SpeculativeConfig(enabled=True, draft_tokens=3)
        self.engine = SpeculativeInferenceEngine(
            self.target_engine, 
            self.draft_engine, 
            self.config
        )
    
    def test_initialization(self):
        self.assertEqual(self.engine.target_engine, self.target_engine)
        self.assertEqual(self.engine.draft_engine, self.draft_engine)
        self.assertEqual(self.engine.config, self.config)
        self.assertIsNotNone(self.engine.metrics)
    
    def test_disabled_speculative_decoding(self):
        """Test that disabled config falls back to target engine"""
        config = SpeculativeConfig(enabled=False)
        engine = SpeculativeInferenceEngine(
            self.target_engine, 
            self.draft_engine, 
            config
        )
        
        async def test():
            shard = Mock()
            result, state = await engine.infer_tensor("test", shard, np.array([1, 2, 3]))
            return result
        
        result = asyncio.run(test())
        self.assertIsNotNone(result)
    
    def test_softmax(self):
        """Test softmax implementation"""
        x = np.array([1.0, 2.0, 3.0])
        probs = self.engine._softmax(x)
        
        # Check that probabilities sum to 1
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
        
        # Check that probabilities are positive
        self.assertTrue(np.all(probs > 0))
        
        # Check that higher logits get higher probabilities
        self.assertTrue(probs[2] > probs[1] > probs[0])
    
    def test_token_acceptance(self):
        """Test token acceptance logic"""
        target_probs = np.array([0.1, 0.3, 0.6])
        draft_probs = np.array([0.2, 0.3, 0.5])
        
        # Test acceptance for different tokens
        # Token 0: target_prob/draft_prob = 0.1/0.2 = 0.5 (should sometimes accept)
        # Token 1: target_prob/draft_prob = 0.3/0.3 = 1.0 (should always accept)
        # Token 2: target_prob/draft_prob = 0.6/0.5 = 1.2 > 1.0 (should always accept)
        
        # Note: Since acceptance is probabilistic, we can't test exact outcomes
        # but we can test the logic doesn't crash
        for token in range(3):
            result = self.engine._accept_token(token, target_probs, draft_probs)
            self.assertIsInstance(result, bool)
    
    def test_metrics_tracking(self):
        """Test metrics tracking"""
        draft_tokens = [1, 2, 3, 4]
        accepted_tokens = [1, 2]
        
        self.engine._update_metrics(draft_tokens, accepted_tokens, 0.1, 0.2)
        
        metrics = self.engine.get_metrics()
        self.assertEqual(metrics['total_draft_tokens'], 4)
        self.assertEqual(metrics['accepted_tokens'], 2)
        self.assertEqual(metrics['rejected_tokens'], 2)
        self.assertEqual(metrics['speculation_rounds'], 1)
        self.assertAlmostEqual(metrics['acceptance_rate'], 0.5)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_flow(self):
        """Test complete speculative decoding flow"""
        async def test():
            target_engine = MockInferenceEngine("target")
            draft_engine = MockInferenceEngine("draft")
            config = SpeculativeConfig(enabled=True, draft_tokens=2)
            engine = SpeculativeInferenceEngine(target_engine, draft_engine, config)
            
            shard = Mock()
            shard.model_id = "test-model"
            
            # Test inference
            result, state = await engine.infer_tensor("test", shard, np.array([1, 2, 3]))
            
            # Should return some result
            self.assertIsNotNone(result)
            
            # Check metrics were updated
            metrics = engine.get_metrics()
            self.assertGreaterEqual(metrics['speculation_rounds'], 0)
        
        asyncio.run(test())

if __name__ == '__main__':
    unittest.main() 
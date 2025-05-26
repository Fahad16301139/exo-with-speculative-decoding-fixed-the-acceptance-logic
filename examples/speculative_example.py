#!/usr/bin/env python3
"""
Example script demonstrating speculative decoding with Exo.

This script shows how to use speculative decoding to potentially speed up inference
by using a smaller draft model to generate candidate tokens that are then verified
by the target model.
"""

import asyncio
import requests
import json
import time

def test_normal_inference():
    """Test normal inference without speculative decoding"""
    print("Testing normal inference...")
    
    url = "http://localhost:52415/v1/chat/completions"
    data = {
        "model": "llama-3.1-8b",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"Normal inference completed in {end_time - start_time:.2f}s")
        print(f"Response: {content[:100]}...")
        return end_time - start_time
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def test_speculative_inference():
    """Test speculative decoding inference"""
    print("\nTesting speculative decoding...")
    
    url = "http://localhost:52415/v1/chat/completions"
    data = {
        "model": "llama-3.1-8b",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "speculative": {
            "enabled": True,
            "draft_tokens": 4,
            "draft_model": "llama-3.2-1b"
        }
    }
    
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"Speculative inference completed in {end_time - start_time:.2f}s")
        print(f"Response: {content[:100]}...")
        return end_time - start_time
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def main():
    """Run comparison between normal and speculative inference"""
    print("Speculative Decoding Example")
    print("=" * 40)
    print("Make sure Exo is running with speculative decoding enabled:")
    print("exo --enable-speculative --draft-tokens 4")
    print()
    
    # Test normal inference
    normal_time = test_normal_inference()
    
    # Wait a bit between tests
    time.sleep(2)
    
    # Test speculative inference
    speculative_time = test_speculative_inference()
    
    # Compare results
    if normal_time and speculative_time:
        speedup = normal_time / speculative_time
        print(f"\nResults:")
        print(f"Normal inference: {normal_time:.2f}s")
        print(f"Speculative inference: {speculative_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print("✅ Speculative decoding provided a speedup!")
        else:
            print("⚠️  Speculative decoding was slower (this can happen with small models or short sequences)")

if __name__ == "__main__":
    main() 
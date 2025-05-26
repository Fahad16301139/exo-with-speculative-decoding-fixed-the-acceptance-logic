#!/usr/bin/env python3
"""
Test real speculative decoding with actual models in exo
"""

import subprocess
import sys
import time

def test_speculative_decoding():
    """Test speculative decoding with real models"""
    
    print("=== Exo Speculative Decoding Test ===")
    print()
    
    # Test 1: Small Llama models (good for testing)
    print("🔥 Test 1: Llama 3.2 3B (target) + Llama 3.2 1B (draft)")
    print("Command: python exo/main.py run llama-3.2-3b --enable-speculative --draft-model llama-3.2-1b --draft-tokens 3 --max-tokens 20 --prompt 'Hello, how are you today?' --inference-engine mlx")
    print()
    
    # Test 2: Qwen models
    print("🔥 Test 2: Qwen 2.5 7B (target) + Qwen 2.5 0.5B (draft)")  
    print("Command: python exo/main.py run qwen-2.5-7b --enable-speculative --draft-model qwen-2.5-0.5b --draft-tokens 4 --max-tokens 15 --prompt 'What is Python?' --inference-engine mlx")
    print()
    
    # Test 3: Without speculative decoding for comparison
    print("🔥 Test 3: Normal inference (no speculative)")
    print("Command: python exo/main.py run llama-3.2-3b --max-tokens 20 --prompt 'Hello, how are you today?' --inference-engine mlx")
    print()
    
    # Test 4: Show available models
    print("🔥 Available models for speculative decoding:")
    print("TARGET MODELS (larger):")
    print("- llama-3.1-8b (32 layers)")
    print("- llama-3.2-3b (28 layers)")
    print("- qwen-2.5-7b (28 layers)")
    print("- qwen-2.5-3b (36 layers)")
    print()
    print("DRAFT MODELS (smaller/faster):")
    print("- llama-3.2-1b (16 layers)")
    print("- qwen-2.5-0.5b (28 layers)")
    print("- qwen-2.5-1.5b (28 layers)")
    print()
    
    print("📝 How speculative decoding works:")
    print("1. Draft model generates 3-4 candidate tokens quickly")
    print("2. Target model verifies all candidates in parallel")
    print("3. Accepted tokens are kept, rejected ones trigger new sampling")
    print("4. This can speed up inference by 1.5-3x!")
    print()
    
    user_choice = input("Do you want to run a test? (1/2/3/n): ").strip()
    
    if user_choice == "1":
        cmd = [
            "python", "exo/main.py", "run", "llama-3.2-3b",
            "--enable-speculative", 
            "--draft-model", "llama-3.2-1b",
            "--draft-tokens", "3",
            "--max-tokens", "20", 
            "--prompt", "Hello, how are you today?",
            "--inference-engine", "mlx"
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif user_choice == "2":
        cmd = [
            "python", "exo/main.py", "run", "qwen-2.5-7b",
            "--enable-speculative",
            "--draft-model", "qwen-2.5-0.5b", 
            "--draft-tokens", "4",
            "--max-tokens", "15",
            "--prompt", "What is Python?",
            "--inference-engine", "mlx"
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif user_choice == "3":
        cmd = [
            "python", "exo/main.py", "run", "llama-3.2-3b",
            "--max-tokens", "20",
            "--prompt", "Hello, how are you today?", 
            "--inference-engine", "mlx"
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    else:
        print("No test run. You can copy-paste the commands above to test manually!")

if __name__ == "__main__":
    test_speculative_decoding() 
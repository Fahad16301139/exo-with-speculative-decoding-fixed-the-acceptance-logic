#!/usr/bin/env python3
"""
Test proper speculative decoding with different models
"""
import os
import asyncio

# Set DEBUG level
os.environ["DEBUG"] = "1"

async def test_with_different_models():
    """Test with different target and draft models"""
    print("🎯 TESTING PROPER SPECULATIVE DECODING")
    print("=" * 50)
    
    # Test with a simple prompt first
    cmd = """python exo/main.py run llama-3.2-3b --enable-speculative --draft-tokens 2 --max-tokens 20 --prompt "Hello, how are you?" --inference-engine mlx"""
    
    print(f"Running: {cmd}")
    
    # Run the command
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_with_different_models()) 
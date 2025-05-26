#!/usr/bin/env python3
"""
Demo script for Exo with Speculative Decoding

This script demonstrates how to use the new speculative decoding feature
that has been implemented in Exo.
"""

import subprocess
import sys
import time

def print_banner():
    print("=" * 60)
    print("🚀 EXO WITH SPECULATIVE DECODING DEMO")
    print("=" * 60)
    print()

def print_section(title):
    print(f"\n📋 {title}")
    print("-" * 40)

def run_command_demo(description, command):
    print(f"\n💡 {description}")
    print(f"Command: {command}")
    print("\nPress Enter to run this command (or Ctrl+C to skip)...")
    try:
        input()
        print("Running...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print("❌ Error!")
            if result.stderr:
                print("Error:", result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr)
    except KeyboardInterrupt:
        print("⏭️  Skipped")

def main():
    print_banner()
    
    print("This demo shows the new speculative decoding feature implemented in Exo.")
    print("Speculative decoding can speed up inference by using a smaller draft model")
    print("to generate candidate tokens that are verified by the target model.")
    print()
    
    print_section("1. Basic Speculative Decoding")
    run_command_demo(
        "Start Exo with speculative decoding enabled (4 draft tokens)",
        "python exo/main.py --enable-speculative --draft-tokens 4 --inference-engine dummy --help"
    )
    
    print_section("2. Custom Draft Model")
    run_command_demo(
        "Start Exo with a custom draft model",
        "python exo/main.py --enable-speculative --draft-model llama-3.2-1b --draft-tokens 3 --inference-engine dummy --help"
    )
    
    print_section("3. Advanced Configuration")
    run_command_demo(
        "Start Exo with advanced speculative settings",
        "python exo/main.py --enable-speculative --draft-tokens 6 --speculative-threshold 0.9 --max-speculation-depth 10 --inference-engine dummy --help"
    )
    
    print_section("4. Run Tests")
    run_command_demo(
        "Run the speculative decoding test suite",
        "python -m pytest test/test_speculative_decoding.py -v"
    )
    
    print_section("5. Example Usage")
    print("\n💡 Example: Running a model with speculative decoding")
    print("Command: python exo/main.py run llama-3.1-8b --enable-speculative --draft-tokens 4")
    print("\nThis would:")
    print("• Load llama-3.1-8b as the target model")
    print("• Auto-select llama-3.2-1b as the draft model")
    print("• Generate 4 draft tokens per speculation round")
    print("• Potentially provide 1.5-3x speedup depending on the sequence")
    
    print_section("6. API Usage")
    print("\n💡 You can also use speculative decoding via the ChatGPT API:")
    print("""
import requests

response = requests.post("http://localhost:52415/v1/chat/completions", json={
    "model": "llama-3.1-70b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "speculative": {
        "enabled": True,
        "draft_tokens": 4,
        "draft_model": "llama-3.2-3b"
    }
})
""")
    
    print_section("Summary")
    print("\n✅ Speculative decoding has been successfully implemented!")
    print("\nKey features:")
    print("• 🎯 Automatic draft model selection")
    print("• ⚙️  Configurable parameters")
    print("• 📊 Real-time metrics tracking")
    print("• 🔄 Fallback to normal inference if needed")
    print("• 🌐 API integration")
    print("• 🧪 Comprehensive test suite")
    
    print("\nTo use speculative decoding:")
    print("1. Add --enable-speculative to your exo command")
    print("2. Optionally configure --draft-tokens, --draft-model, etc.")
    print("3. Enjoy potentially faster inference! 🚀")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 
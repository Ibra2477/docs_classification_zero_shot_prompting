#!/usr/bin/env python3
"""
Fix for the DeepSeek model loading error:
"The current `device_map` had weights offloaded to the disk. 
Please provide an `offload_folder` for them."

This script demonstrates multiple solutions to the problem.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

def solution_1_offload_folder():
    """Solution 1: Add offload_folder parameter"""
    print("=== Solution 1: Using offload_folder ===")
    try:
        # Create offload directory if it doesn't exist
        os.makedirs("./model_offload", exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
            offload_folder="./model_offload"  # This fixes the error
        )
        print("✅ Model loaded successfully with offload_folder!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None, None

def solution_2_no_device_map():
    """Solution 2: Load without device_map"""
    print("\n=== Solution 2: Without device_map ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
            # No device_map parameter
        )
        print("✅ Model loaded successfully without device_map!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None, None

def solution_3_cpu_only():
    """Solution 3: Force CPU loading"""
    print("\n=== Solution 3: CPU-only loading ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True,
            device_map="cpu"  # Force CPU
        )
        print("✅ Model loaded successfully on CPU!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None, None

def solution_4_low_memory():
    """Solution 4: Load with low memory settings"""
    print("\n=== Solution 4: Low memory loading ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
            offload_folder="./model_offload",
            low_cpu_mem_usage=True,
            load_in_8bit=True  # Use 8-bit quantization
        )
        print("✅ Model loaded successfully with low memory settings!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None, None

def test_model(model, tokenizer):
    """Test the loaded model with a simple generation"""
    if model is None or tokenizer is None:
        return False
    
    try:
        from transformers import pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        test_prompt = "def hello_world():"
        result = generator(
            test_prompt,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.3,
            return_full_text=False
        )
        
        print(f"🧪 Test successful! Generated: {result[0]['generated_text'][:50]}...")
        return True
    except Exception as e:
        print(f"🧪 Test failed: {e}")
        return False

def main():
    print(f"Attempting to load model: {MODEL_NAME}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Try solutions in order
    solutions = [
        solution_1_offload_folder,
        solution_2_no_device_map,
        solution_3_cpu_only,
        solution_4_low_memory
    ]
    
    for solution in solutions:
        model, tokenizer = solution()
        if model is not None and tokenizer is not None:
            if test_model(model, tokenizer):
                print(f"\n🎉 Success! Use the code from {solution.__name__}")
                break
            else:
                print(f"Model loaded but test failed for {solution.__name__}")
        print("-" * 50)
    else:
        print("\n❌ All solutions failed. Consider:")
        print("1. Installing safetensors: pip install safetensors")
        print("2. Using a smaller model")
        print("3. Checking your internet connection")
        print("4. Verifying your Hugging Face token")

if __name__ == "__main__":
    main()
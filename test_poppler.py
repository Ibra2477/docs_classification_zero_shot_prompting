import os
import json
import re
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

# Get token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN = "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"
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face token.")
login(token=HF_TOKEN)

# Model name from the error message
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

print(f"Loading model: {MODEL_NAME}")

# Load tokenizer first
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")

print("Loading model...")
try:
    # Fix 1: Add offload_folder parameter
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
        offload_folder="./model_offload"  # Added offload folder
    )
    print("Model loaded successfully with offload_folder!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Try updating transformers or using a different model")
    
    # Alternative fix: Try without device_map
    try:
        print("Trying without device_map...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("Model loaded successfully without device_map!")
    except Exception as e2:
        print(f"Still failed: {e2}")
        print("Consider installing safetensors: pip install safetensors")
        exit(1)

# Test the model with a simple generation
try:
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    test_prompt = "def calculate_sum(a, b):"
    result = generator(
        test_prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.3,
        return_full_text=False
    )
    
    print("Test generation successful!")
    print(f"Input: {test_prompt}")
    print(f"Output: {result[0]['generated_text']}")
    
except Exception as e:
    print(f"Error during text generation: {e}")
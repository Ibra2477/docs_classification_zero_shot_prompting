# Fix for DeepSeek Model Loading Error

## Problem
You're getting this error when trying to load the DeepSeek model:
```
ValueError: The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder` for them. Alternatively, make sure you have `safetensors` installed if the model you are using offers the weights in this format.
```

## Solutions

### Solution 1: Add `offload_folder` (Recommended)
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='auto',
    offload_folder="./model_offload"  # Add this line
)
```

### Solution 2: Install safetensors
In your virtual environment:
```bash
pip install safetensors
```

### Solution 3: Remove device_map
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
    # Remove device_map='auto'
)
```

### Solution 4: Use CPU-only
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map="cpu"
)
```

## Files Fixed
- `doc_classication_deepseek.py` - Updated with offload_folder
- `test_poppler.py` - Created with proper error handling
- `fix_model_loading.py` - Comprehensive testing script

## Testing
Run the fix testing script:
```bash
python fix_model_loading.py
```

This will try all solutions and tell you which one works best for your setup.

## Additional Requirements
Install the updated requirements:
```bash
pip install -r requirements_fix.txt
```
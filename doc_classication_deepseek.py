import os
import json
import re
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Get token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN = "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"
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face token.")
login(token=HF_TOKEN)

# CORRECTED MODEL IDENTIFIER
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"

# Initialize model with better settings
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def deepseek_classify_and_extract(text: str) -> dict:
    """Classify document and extract data using DeepSeek LLM zero-shot"""
    truncated_text = text[:10000]  # More conservative truncation
    
    classification_prompt = f"""
      [SYSTEM] You are a financial document expert. 
    1. Classify the document type as either "Bank Statement" or "Invoice".
    2. Extract relevant data into structured JSON format:
    Use these exact structures:
    
    Bank Statement:
    {{
      "document_type": "Bank Statement",
      "transactions": [
        {{
          "date_operation": "DD/MM/YYYY",
          "date_valeur": "DD/MM/YYYY",
          "libelle_operation": "...",
          "reference_operation": "...",
          "credit": 0.00,
          "debit": 0.00
        }}
      ]
    }}
    
    Invoice:
    {{
      "document_type": "Invoice",
      "date": "DD/MM/YYYY",
      "due_date": "DD/MM/YYYY",
      "invoice_number": "...",
      "total_amount": 100.00,
      "items": [
        {{
          "description": "...",
          "quantity": 1,
          "unit_price": 50.00,
          "total": 50.00
        }}
      ]
    }}

    """
    
    try:
        response = generator(
            classification_prompt,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            return_full_text=False
        )
        
        # Extract JSON from response
        json_str = response[0]['generated_text'].strip()
        json_match = re.search(r'\{[\s\S]*\}', json_str)
        
        if json_match:
            return json.loads(json_match.group())
        return {"error": "No JSON found in response", "raw": json_str}
    
    except Exception as e:
        return {"error": str(e)}

def process_file(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        result = {
            "file_name": os.path.basename(file_path),
            "content_length": len(text)
        }
        
        # Process only if content is not empty
        if text.strip():
            result.update(deepseek_classify_and_extract(text))
        else:
            result["error"] = "Empty file"
            
        return result
    
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "error": str(e)
        }

if __name__ == "__main__":
    # Example usage
    result = process_file("output/0dbef502-RELEVES_0082348808_20240906_page1.txt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
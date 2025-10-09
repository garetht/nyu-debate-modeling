#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
MODEL_BASE = "openai/gpt-oss-20b"
OUTPUT_DIR = "outputs/trained_models/gpt_oss_20b_lora_11_09"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")

def load_prompt_from_file(file_path="prompt.txt"):
    """Load prompt from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt_text = file.read()
        print(f"Loaded prompt from '{file_path}' ({len(prompt_text)} characters)")
        return prompt_text
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found in current directory")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def load_finetuned_model():
    """Load the fine-tuned GPT-OSS model with LoRA adapters"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, input_text, max_new_tokens=512):
    """Generate response using the harmony format"""
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")
    
    # Generate response with appropriate parameters for GPT-OSS
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (response)
    input_length = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response

def main():
    # Load prompt from file
    prompt_text = load_prompt_from_file("/home/ubuntu/mars-arnesen-gh/leonidtsyplenkov/sft_data/debater/prompt.txt")
    
    if prompt_text is None:
        print("Failed to load prompt. Exiting...")
        return
    
    try:
        # Load the fine-tuned model
        model, tokenizer = load_finetuned_model()
        
        # Generate response using the loaded prompt
        response = generate_response(model, tokenizer, prompt_text)
        
        # Print results
        print("\n" + "="*60)
        print("GENERATED RESPONSE:")
        print("="*60)
        print(response)
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your model paths are correct")
        print("2. Check that PEFT library is installed: pip install peft")
        print("3. Verify you have sufficient GPU memory")
        print("4. Make sure 'prompt.txt' exists in the same directory")

if __name__ == "__main__":
    main()

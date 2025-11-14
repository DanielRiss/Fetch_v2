import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# FIX 1: Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Padding token set to EOS token: {tokenizer.pad_token}")

print("Tokenizer loaded successfully.")

# For Mistral, we need to use the base model and extract logits manually
# Or use a verifier-specific variant if available
value_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

value_model.eval()
print("Verifier B model loaded successfully.")

app = FastAPI()

class InputText(BaseModel):
    texts: List[str]

class OutputPrediction(BaseModel):
    values: List[float]

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    max_seq_length = 1024
    inputs = tokenizer(
        input_text.texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_seq_length
    )
    print("Length of tokenized sequences:", inputs["input_ids"].shape[1])
    
    inputs = {name: tensor.to(value_model.device) for name, tensor in inputs.items()}
    
    with torch.no_grad():
        outputs = value_model(**inputs, output_hidden_states=True)
        # For causal LM, use the last hidden state as representation
        hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, seq_length, hidden_dim)
        
        # Get the hidden state at the last token position for each sequence
        batch_size = hidden_states.shape[0]
        last_token_indices = torch.sum(inputs["attention_mask"], dim=-1) - 1
        
        # Extract value from last token's hidden state
        # Simple approach: take mean of last hidden state features as score
        scores = []
        for i in range(batch_size):
            last_idx = last_token_indices[i].item()
            last_hidden = hidden_states[i, last_idx, :]  # Shape: (hidden_dim,)
            # Use mean of hidden state as a value estimate
            score = float(last_hidden.mean().item())
            scores.append(score)
    
    print(f"Extracted scores: {scores}")
    return {"values": scores}
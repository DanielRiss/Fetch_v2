import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, LlamaForTokenClassification

model_name_or_path = "xmu-nlp/Llama-3-8b-gsm8k-value-A"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# FIX 1: Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Padding token set to EOS token: {tokenizer.pad_token}")

print("Tokenizer loaded successfully.")

value_model = LlamaForTokenClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

value_model.eval()
print("Verifier A model loaded successfully.")

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
        outputs = value_model(**inputs)
        # outputs.logits has shape (batch_size, seq_length, num_labels)
        logits = outputs.logits  # Shape: (batch_size, seq_length, 1) for token classification
        
        # Get the logits at the last token position for each sequence
        # This is the standard approach for sequence-level scoring
        batch_size = logits.shape[0]
        last_token_indices = torch.sum(inputs["attention_mask"], dim=-1) - 1
        
        # Extract score from last valid token for each sequence
        scores = []
        for i in range(batch_size):
            last_idx = last_token_indices[i].item()
            score = logits[i, last_idx, 0].item()  # Get single float value
            scores.append(float(score))
    
    print(f"Extracted scores: {scores}")
    return {"values": scores}
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, LlamaForTokenClassification

model_name_or_path = ["xmu-nlp/Llama-3-8b-gsm8k-value-A",
                      "xmu-nlp/Llama-3-8b-gsm8k-value-B"]

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path[0], use_fast=True)
print("Tokenizer loaded successfully.")

value_models = []
for i in range(len(model_name_or_path)):
    print(f"Loading model {i} on cuda:{i}")
    value_model = LlamaForTokenClassification.from_pretrained(
        model_name_or_path[i], 
        torch_dtype=torch.float16,
        device_map=f"cuda:{i}"  # Use device_map for proper GPU assignment
    )
    value_model.eval()
    value_models.append(value_model)
    print(f"Model {i} loaded successfully on {value_model.device}")

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
    
    batch_size = inputs["input_ids"].shape[0]
    print(f"Processing batch of size: {batch_size}, sequence length: {inputs['input_ids'].shape[1]}")
    
    # Collect outputs from both models
    all_outputs = []
    for i, model in enumerate(value_models):
        # Move inputs to model's device
        device_inputs = {k: v.to(model.device) for k, v in inputs.items()}
        indices = torch.sum(device_inputs["attention_mask"], dim=-1) - 1
        
        with torch.no_grad():
            logits = model(**device_inputs).logits.squeeze(-1)
            # Get the logit at the last valid token for each sequence
            outputs = logits[torch.arange(batch_size), indices]
            # Clamp and move to CPU
            return torch.clamp(outputs, min=-1, max=1).cpu().numpy().tolist()

            all_outputs.append(clamped)
            print(f"Model {i} predictions: {clamped}")
    
    # Average across models (axis=0 averages the two model outputs)
    avg_outputs = np.mean(all_outputs, axis=0).tolist()
    print(f"Ensemble averaged outputs: {avg_outputs}")
    
    return {"values": avg_outputs}

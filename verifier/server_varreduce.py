import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, LlamaForTokenClassification

# Read model path from environment
model_name_or_path = os.environ["MODEL_PATH"]

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
value_model = LlamaForTokenClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
value_model.eval()

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
    inputs = {k: v.to(value_model.device) for k, v in inputs.items()}
    indices = torch.sum(inputs["attention_mask"], dim=-1) - 1
    with torch.no_grad():
        logits = value_model(**inputs).logits.squeeze(-1)
        # clamp per‐example logits to [–1,1]
        clamped = torch.clamp(logits, min=-1.0, max=1.0)
        outputs = clamped[torch.arange(len(indices)), indices].cpu().numpy().tolist()

    return {"values": outputs}

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering

model_fpath = "xmu-nlp/simcse-large-gsm8k"

# Load tokenizer and model once at startup
tokenizer = AutoTokenizer.from_pretrained(model_fpath)
model = AutoModel.from_pretrained(model_fpath).cuda()
max_seq_length = 256

def compute_emb(texts: List[str]):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Get embeddings - use pooler_output from model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        embeddings = outputs.pooler_output
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy()

def cluster(texts: List[str], distance_threshold: float = 0.1):
    if len(texts) < 2:
        # Only one element, assign to one cluster
        return [0]
    embeddings = compute_emb(texts)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",  # Note: "affinity" vs "metric" depends on sklearn version
        linkage="average",
        distance_threshold=distance_threshold
    )
    clustering.fit(embeddings)
    return clustering.labels_.tolist()

app = FastAPI()

class InputText(BaseModel):
    texts: List[str]
    d: float = 0.1  # default distance threshold

class OutputPrediction(BaseModel):
    labels: List[int]

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    labels = cluster(input_text.texts, input_text.d)
    return {"labels": labels}

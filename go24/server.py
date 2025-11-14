from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class InputText(BaseModel):
    texts: List[str]

class OutputPrediction(BaseModel):
    values: List[float]

def score_move(move_text: str) -> float:
    try:
        result = float(move_text.split("=")[1].split("(")[0].strip())
        
        if abs(result - 24) < 0.01: return 1.0
        if abs(result - 12) < 0.1: return 0.85
        if abs(result - 8) < 0.1: return 0.80
        if abs(result - 6) < 0.1: return 0.80
        if 1 <= result < 24: return 0.60
        if result <= 0: return 0.1
        return 0.5
    except:
        return 0.5

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    values = [score_move(text) for text in input_text.texts]
    return {"values": values}


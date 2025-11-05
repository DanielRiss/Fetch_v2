import os
import re
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", 3500))  # leave room for generation
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 16))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))  # deterministic
TOP_P = float(os.environ.get("TOP_P", 1.0))

# Instruction to elicit a single numeric score
SYSTEM_PROMPT = (
    "You are a strict verifier. Read the provided text (which may include a question, options, and a thought). "
    "Output a single line in the exact format: SCORE: <number between -1 and 1>. "
    "Do not output anything else after the number."
)

# Optional few-shot examples (kept minimal). You can extend these if needed.
FEW_SHOTS = [
    {
        "user": "Question: 2+2=?\nA. 3\nB. 4\nC. 5\nD. 6\nThought: The sum of two and two is four.\nAnswer:",
        "assistant": "SCORE: 1.0",
    },
    {
        "user": "Question: 2+2=?\nA. 3\nB. 4\nC. 5\nD. 6\nThought: The sum of two and two is six.\nAnswer:",
        "assistant": "SCORE: -0.8",
    },
    {
        "user": "Question: A planet orbits a star of twice the Sun's mass at a semi-major axis of 1 AU. Compared to Earth's orbital period, what is its period?\nA. Shorter than 1 year\nB. Equal to 1 year\nC. Longer than 1 year\nD. Cannot be determined from the information given\nThought: By Kepler's third law in Newtonian form, T ∝ a^{3/2} / sqrt(M_star). With a = 1 AU and M_star = 2 M_sun, T ≈ 1 / sqrt(2) years, which is less than 1 year.\nAnswer:",
        "assistant": "SCORE: 0.9"
    },
    {
        "user": "Question: A 0.010 M solution of acetic acid (Ka ~ 1.8*10^-5) is compared to a 0.010 M solution of HCl. Which has the higher pH?\nA. They have approximately the same pH (~2)\nB. The acetic acid solution has the higher pH\nC. The acetic acid solution has the lower pH\nD. It cannot be determined from the information given\nThought: HCl is a strong acid and fully dissociates; acetic acid is weak and only partially dissociates. At equal formal concentration, the weak acid yields fewer H3O+ ions, so its pH is higher (less acidic).\nAnswer:",
        "assistant": "SCORE: 0.8"
    },
    {
        "user": "Question: In Michaelis-Menten enzyme kinetics, what is the effect of a competitive inhibitor on Km and Vmax?\nA. Increases Vmax; Km unchanged\nB. Decreases Vmax; Km unchanged\nC. Increases Km; Vmax unchanged\nD. Decreases Km; Vmax unchanged\nThought: Competitive inhibitors compete for the active site, effectively reducing apparent affinity. They increase apparent Km but do not change Vmax at saturating substrate.\nAnswer:",
        "assistant": "SCORE: 0.9"
    },
    {
        "user": "Question: An exoplanet transit shows a fractional flux drop (transit depth) of 1%. What is the planet-to-star radius ratio Rp/Rs?\nA. 0.01\nB. 0.10\nC. 0.316\nD. 0.50\nThought: For small planets, transit depth ≈ (Rp/Rs)^2. If depth = 0.01, then Rp/Rs = sqrt(0.01) = 0.1.\nAnswer:",
        "assistant": "SCORE: 0.85"
    },
    {
        "user": "Question: During an El Niño event, what typically happens along the Peruvian coast?\nA. Coastal upwelling strengthens, leading to cooler sea surface temperatures\nB. Coastal upwelling weakens, leading to warmer sea surface temperatures\nC. Trade winds strengthen, enhancing the Walker circulation\nD. Global mean sea level drops significantly\nThought: El Niño weakens the trade winds and the Walker circulation, reducing upwelling along the eastern Pacific coast. Reduced upwelling yields warmer coastal sea surface temperatures.\nAnswer:",
        "assistant": "SCORE: 0.8"
    },
    {
        "user": "Question: In the photoelectric effect, what primarily determines the maximum kinetic energy of emitted electrons?\nA. Increasing light intensity always increases the electrons' maximum kinetic energy\nB. Increasing light frequency increases the electrons' maximum kinetic energy\nC. Decreasing light frequency increases the electrons' maximum kinetic energy\nD. The maximum kinetic energy is independent of frequency\nThought: Brighter light (higher intensity) means more energy per electron, so the electrons should be ejected with greater kinetic energy as intensity rises.\nAnswer:",
        "assistant": "SCORE: -0.8"
    },
    {
        "user": "Question: For an exothermic reaction at equilibrium, what is the effect of increasing temperature?\nA. Shifts equilibrium to products (right)\nB. Shifts equilibrium to reactants (left)\nC. No change in equilibrium position\nD. Insufficient information to decide\nThought: Raising temperature speeds up the forward reaction, so more products should form and the equilibrium should shift to the right.\nAnswer:",
        "assistant": "SCORE: -0.7"
    },
    {
        "user": "Question: Two genes show a recombination frequency of 20 percent in test crosses. What is their approximate genetic map distance?\nA. 5 centiMorgans (cM)\nB. 20 centiMorgans (cM)\nC. 40 centiMorgans (cM)\nD. 80 centiMorgans (cM)\nThought: In classical mapping for moderate distances, recombination frequency (as a percent) approximates map distance in centiMorgans. 20 percent recombination ≈ 20 cM.\nAnswer:",
        "assistant": "SCORE: 0.75"
    }
]

# Load model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Use chat template if available (Llama-3.1 Instruct supports it)
uses_chat = hasattr(tokenizer, "apply_chat_template")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

app = FastAPI()

class InputText(BaseModel):
    texts: List[str]

class OutputPrediction(BaseModel):
    values: List[float]

score_pat = re.compile(r"SCORE\s*:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)

def clamp_score(x: float) -> float:
    return max(-1.0, min(1.0, x))

def build_inputs(text: str):
    """
    Build inputs either via chat template (preferred) or plain prompt.
    """
    if uses_chat:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in FEW_SHOTS:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": f"{text}\nPlease respond with exactly one line: SCORE: <float>."})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS)
        return enc, prompt
    else:
        # Fallback single-turn prompt
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"TEXT:\n{text}\n\n"
            f"Respond with exactly one line: SCORE: <float>.\n"
        )
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS)
        return enc, prompt

def generate_score(text: str) -> float:
    enc, prompt = build_inputs(text)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0.0),
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        )
    # Decode only the newly generated part
    gen_text = tokenizer.decode(out_ids[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # Try to parse SCORE: <float>
    m = score_pat.search(gen_text)
    if m:
        try:
            return clamp_score(float(m.group(1)))
        except Exception:
            pass
    # Fallback: grab any float in the output
    any_num = re.search(r"[-+]?\d*\.?\d+", gen_text)
    if any_num:
        try:
            return clamp_score(float(any_num.group(0)))
        except Exception:
            pass
    # If parsing fails, return 0.0
    return 0.0

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    values: List[float] = []
    for t in input_text.texts:
        score = generate_score(t)
        values.append(score)
    return {"values": values}
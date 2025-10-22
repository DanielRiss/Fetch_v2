#experimental

#!/usr/bin/env python3
"""
sft_pretrain_llama31.py

Instruction/SFT-style fine-tuning of meta-llama/Llama-3.1-8B-Instruct
on a CSV with columns:
    Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3, Explanation

Requirements (install before running):
    pip install -U "transformers>=4.34" accelerate datasets peft trl bitsandbytes sentencepiece pandas

You MUST have accepted the model license on Hugging Face and be logged in (huggingface-cli login).
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTTrainingArguments

# -------------------------
# Utilities: prompt formatting
# -------------------------
def build_prompt(question: str, choices: List[str]) -> str:
    """
    Build a concise instruction-style prompt that the model should follow.
    We'll present choices labeled A/B/C/... and ask for the answer and explanation.
    """
    labels = ["A", "B", "C", "D", "E"]
    # Trim choices to available labels
    choices = choices[: len(labels)]
    choice_lines = "\n".join(f"{labels[i]}) {c.strip()}" for i, c in enumerate(choices))
    prompt = (
        f"Question: {question.strip()}\n"
        f"Choices:\n{choice_lines}\n\n"
        f"Please provide (1) the correct choice letter and (2) the correct answer text, "
        f"and (3) a short explanation for why it is correct.\n\nAnswer:"
    )
    return prompt

def build_target(correct_answer: str, explanation: Optional[str]) -> str:
    """
    Target the model should produce: letter + answer text + explanation.
    We'll keep a consistent structure so trainer can learn the format.
    """
    expl = (explanation.strip() if isinstance(explanation, str) else "")
    target = f" {correct_answer.strip()}\n\nExplanation: {expl}\n"
    return target

# -------------------------
# Data preparation
# -------------------------
def csv_to_dataset(csv_path: str, question_col="Question", correct_col="Correct Answer",
                   inc_cols=None, explanation_col="Explanation") -> Dataset:
    df = pd.read_csv(csv_path)
    if inc_cols is None:
        # try to detect up to 3 incorrect columns by name pattern
        inc_cols = [c for c in df.columns if "Incorrect" in c][:3]

    records = []
    for _, row in df.iterrows():
        question = str(row[question_col])
        correct = str(row[correct_col])
        explanation = row[explanation_col] if explanation_col in row.index else ""
        choices = [correct]
        # add incorrects if present
        for c in inc_cols:
            if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
                choices.append(str(row[c]))
        # shuffle? We keep correct at position A for clarity; you may randomize if you want.
        prompt = build_prompt(question, choices)
        # For training target, include the text of the correct answer (not just letter), to teach model to produce both
        target = build_target(correct, explanation)
        records.append({"prompt": prompt, "response": target})

    ds = Dataset.from_pandas(pd.DataFrame(records))
    return ds

# -------------------------
# Tokenization helper
# -------------------------
def tokenize_function(examples, tokenizer, max_length=1024):
    # For SFTTrainer we typically concatenate prompt + response and create labels for the full sequence,
    # but mask the prompt tokens with -100 so loss is only computed on response tokens.
    inputs = [p + r for p, r in zip(examples["prompt"], examples["response"])]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    # build labels: set prompt tokens to -100
    with tokenizer.as_target_tokenizer():
        resp_tok = tokenizer(examples["response"], max_length=max_length, truncation=True, padding="max_length")
    # compute number of prompt tokens per example to mask them
    prompt_tok = tokenizer(examples["prompt"], max_length=max_length, truncation=True, padding="max_length")
    labels = []
    for i in range(len(inputs)):
        # label = token ids for full input, but mask prompt tokens with -100
        input_ids = model_inputs["input_ids"][i].copy()
        # find length of prompt tokens by comparing prompt_tok attention mask sum
        prompt_len = sum(prompt_tok["attention_mask"][i])
        label = [-100] * prompt_len + input_ids[prompt_len:]
        # ensure label length equals input_ids length
        label = label[: len(input_ids)]
        if len(label) < len(input_ids):
            label += [-100] * (len(input_ids) - len(label))
        labels.append(label)
    model_inputs["labels"] = labels
    return model_inputs

# -------------------------
# Main training flow
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="SFT fine-tune Llama 3.1-8B on GPQA data")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--output_dir", default="./sft_llama31_finetuned", help="Where to save adapters / model")
    parser.add_argument("--do_lora", action="store_true", help="Use LoRA (recommended)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true", help="Load quantized 4-bit model via bitsandbytes (if supported)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    args = parser.parse_args()

    # -------------------------
    # Load tokenizer & model
    # -------------------------
    print("Loading tokenizer and model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # AutoModelForCausalLM supports these models; use device_map="auto" for multi-GPU or accelerate
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16 if args.fp16 else None,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # If using k-bit (4-bit) + PEFT, prepare model for k-bit training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # -------------------------
    # Optional: attach LoRA (PEFT)
    # -------------------------
    if args.do_lora:
        print("Preparing LoRA (PEFT) adapter...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # -------------------------
    # Load dataset from CSV and tokenize
    # -------------------------
    print("Loading CSV and building dataset...")
    inc_cols = [c for c in ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Incorrect Answer"] if c]
    ds = csv_to_dataset(args.csv, inc_cols=inc_cols)
    # tokenize dataset
    tokenized = ds.map(lambda x: tokenize_function(x, tokenizer, max_length=args.max_length), batched=True, remove_columns=ds.column_names)

    # -------------------------
    # Training arguments & SFTTrainer
    # -------------------------
    print("Configuring training...")
    sft_args = SFTTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=20,
        save_strategy="epoch",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Saving final model/adapters...")
    # If LoRA used, save adapter only; otherwise save full model
    if args.do_lora:
        print("Saving LoRA adapter to", args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)

    print("Done.")

if __name__ == "__main__":
    main()

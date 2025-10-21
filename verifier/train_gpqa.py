import sys
import torch
import random
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, Trainer
from transformers import LlamaForTokenClassification
from datasets import Dataset, DatasetDict
from torch.nn import MSELoss
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

# Set up training arguments
training_args = TrainingArguments(
    output_dir=sys.argv[2],
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=10,
    learning_rate=2e-6,
    fp16=True,
    fp16_full_eval=True,
    weight_decay=0.01,
    warmup_ratio=0.05,
    max_steps=0,
    deepspeed="ds_config.json",
    save_only_model=True,
    report_to="none",
    seed=42
)
HEAD_LR = 1e-5

# Load model, tokenizer
model_name = sys.argv[1]
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded successfully")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token}")
except Exception as e:
    print(f"ERROR loading tokenizer: {e}")
    raise
config = AutoConfig.from_pretrained(model_name)
if config.num_labels != 1:
    config.num_labels = 1

class MyLlamaForTokenClassification(LlamaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, input_ids, attention_mask, labels, **kwargs):
        outputs = super().forward(input_ids, attention_mask, labels=None, return_dict=True, **kwargs)
        label_mask = torch.logical_and(attention_mask != 0, labels != 0)
        logits = outputs.logits.squeeze()
        loss_fct = MSELoss()
        outputs["loss"] = loss_fct(logits[label_mask], labels[label_mask].to(torch.float16))
        return outputs

model = MyLlamaForTokenClassification.from_pretrained(model_name, config=config, torch_dtype=torch.float16)

if config.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    config.pad_token = "[PAD]"
    config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = "[PAD]"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    print("Added [PAD] token, set pad token id", tokenizer.pad_token_id)

# Load and preprocess GPQA CSV
def load_gpqa_csv(path):
    df = pd.read_csv(path)
    # Remove rows with missing data
    df = df.dropna(subset=["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"])
    # Build input/label pairs
    rows = []
    for _, row in df.iterrows():
        answers = [
            ("A", row["Correct Answer"], 1),
            ("B", row["Incorrect Answer 1"], 0),
            ("C", row["Incorrect Answer 2"], 0),
            ("D", row["Incorrect Answer 3"], 0),
        ]
        random.shuffle(answers)  # Shuffle choices per row
        input_text = f"Question: {row['Question']}\n"
        # Re-assign A, B, C, D after shuffle
        for i in range(len(answers)):
            answers[i] = (chr(ord('A') + i), answers[i][1], answers[i][2])
        # Add answer options to input text
        for letter, ans, _ in answers:
            input_text += f"{letter}. {ans}\n"
        # The label is a list: 1 for correct, 0 for incorrect, in the order presented
        label = [lbl for _, _, lbl in answers]
        rows.append({"text": input_text.strip(), "label": label})
    return rows

train_rows = load_gpqa_csv(sys.argv[3])
valid_rows = load_gpqa_csv(sys.argv[4])

dataset = DatasetDict({
    "train": Dataset.from_list(train_rows),
    "valid": Dataset.from_list(valid_rows),
})

eos_token = tokenizer.eos_token
def add_eos_token(text):
    if not text.endswith(eos_token):
        text += eos_token
    return text

def preprocess_function(examples):
    inputs = [add_eos_token(t) for t in examples["text"]]
    # For each example, expand label to match input length (simple: label for each answer, rest 0)
    model_inputs = tokenizer(inputs, padding="max_length", max_length=512, truncation=True, add_special_tokens=True)
    new_labels = []
    for idx, input_text in enumerate(inputs):
        # Find answer lines
        lines = input_text.splitlines()
        label = examples["label"][idx]
        labels = [0] * len(model_inputs["input_ids"][idx])
        # Assign label to the first token of each answer line
        for i, line in enumerate(lines):
            if line and line[0] in "ABCD" and line[1] == ".":
                # Find token index for this line
                ans_idx = input_text.find(line)
                token_idx = None
                for j, tok in enumerate(model_inputs["input_ids"][idx]):
                    if tok == tokenizer.eos_token_id:
                        break
                    # crude: assign label to first non-pad token after previous answer
                    if token_idx is None and model_inputs["attention_mask"][idx][j]:
                        token_idx = j
                        break
                if token_idx is not None and i-1 < len(label):
                    labels[token_idx] = label[i-1]
        new_labels.append(labels)
    model_inputs["labels"] = new_labels
    return model_inputs

with training_args.main_process_first(desc="dataset map tokenizer"):
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=["text", "label"],
        desc="Running tokenizer on train dataset",
    )

with training_args.main_process_first(desc="dataset map tokenizer"):
    valid_dataset = dataset["valid"].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=["text", "label"],
        desc="Running tokenizer on valid dataset",
    )

@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        return batch

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
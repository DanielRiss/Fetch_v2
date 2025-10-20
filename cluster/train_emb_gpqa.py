import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import RobertaTokenizer, RobertaModel

model_name_or_path ="xmu-nlp/simcse-large-gsm8k" # sup-simcse-roberta-large ?

tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
model = RobertaModel.from_pretrained(model_name_or_path, device_map="auto")
model.train()

data_fpath = "/home/tzeinstra/projects/Fetch/gpqa/dataset/gpqa_training_split.csv"
output_model_path = f"cluster/"

df = pd.read_csv(data_fpath)
# Remove rows with missing data
df = df.dropna(subset=["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"])

# Build pairs for contrastive training: correct vs incorrect answers for each question
pairs = []
for _, row in df.iterrows():
    question = row["Question"]
    correct = row["Correct Answer"]
    incorrects = [
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"]
    ]
    # Each correct/incorrect pair gets label 1/0
    for inc in incorrects:
        pairs.append({
            "text1": f"Question: {question}\nAnswer: {correct}",
            "text2": f"Question: {question}\nAnswer: {inc}",
            "label": 1
        })
        pairs.append({
            "text1": f"Question: {question}\nAnswer: {inc}",
            "text2": f"Question: {question}\nAnswer: {correct}",
            "label": 0
        })


import random
random.seed(42)

epoch = 1
batch_size = 128
mini_batch_size = 16

import torch
from torch import nn
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=0.1)

# add scheduler of transformers
from transformers import get_linear_schedule_with_warmup
step_num = epoch * len(pairs) // batch_size
warmup_ratio = 0.1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(step_num * warmup_ratio), num_training_steps=step_num)

from tqdm import tqdm

for _ in range(epoch):
    random.shuffle(pairs)
    pbar = tqdm(total=len(pairs) // batch_size)
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        _loss = 0
        for j in range(0, len(batch), mini_batch_size):
            mini_batch = batch[j:j+mini_batch_size]

            text1 = [item["text1"] for item in mini_batch]
            text2 = [item["text2"] for item in mini_batch]
            labels = [item["label"] for item in mini_batch]

            text1_inputs = tokenizer(text1, padding=True, truncation=True, return_tensors="pt", max_length=256)
            text2_inputs = tokenizer(text2, padding=True, truncation=True, return_tensors="pt", max_length=256)

            text1_inputs = {k: v.to(model.device) for k, v in text1_inputs.items()}
            text2_inputs = {k: v.to(model.device) for k, v in text2_inputs.items()}
            labels = torch.tensor(labels).to(model.device).float()

            text1_outputs = model(**text1_inputs, output_hidden_states=True, return_dict=True).pooler_output
            text2_outputs = model(**text2_inputs, output_hidden_states=True, return_dict=True).pooler_output

            # cosine similarity
            cos_sim = nn.CosineSimilarity()(text1_outputs, text2_outputs)
            loss = nn.BCEWithLogitsLoss()(cos_sim, labels)

            loss.backward()
            _loss += loss.detach().item()

        _loss /= batch_size // mini_batch_size
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        pbar.set_description(f"loss: {_loss:.4f}")
        pbar.update(1)
    pbar.close()

tokenizer.save_pretrained(output_model_path)
model.save_pretrained(output_model_path)





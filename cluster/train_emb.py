import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import RobertaTokenizer, RobertaModel

model_name_or_path ="xmu-nlp/simcse-large-gsm8k"

tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
model = RobertaModel.from_pretrained(model_name_or_path, device_map="auto")
model.train()

import jsonlines
data_fpath = "/home/dris/projects/Fetch/gsm8k/test_main.jsonl"
output_model_path = f"cluster/"

with jsonlines.open(data_fpath) as reader:
    dataset = list(reader)

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
step_num = epoch * len(dataset)//batch_size
warmup_ratio = 0.1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=step_num * warmup_ratio, num_training_steps=step_num)

from tqdm import tqdm

for _ in range(epoch):
    random.shuffle(dataset)
    pbar = tqdm(total=len(dataset)//batch_size)
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        _loss = 0
        for j in range(0, len(batch), mini_batch_size):
            mini_batch = batch[j:j+mini_batch_size]

            text1, text2, labels = [], [], []
            for item in mini_batch:
                text1.append(item["text1"])
                text2.append(item["text2"])
                labels.append(item["label"])

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
            # print(cos_sim, labels, loss)

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






import os
import json
import pickle
import numpy as np
import jsonlines
import requests
import time
from tqdm import tqdm
import itertools

# Toggle between a single ensemble host or two separate hosts
USE_ENSEMBLE_HOST = False

# Endpoint configurations
ENSEMBLE_URL = "http://127.0.0.1:8002/predict"
SINGLE_HOST_URLS = [
    "http://127.0.0.1:8002/predict",
    "http://127.0.0.1:8003/predict"
]
single_cycle = itertools.cycle(SINGLE_HOST_URLS)

# Fetch the job ID (or fallback to timestamp)
JOBID = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOBID") or datetime.now().strftime("%Y%m%d%H%M%S")
LOGDIR = os.environ.get("LOGDIR", ".")


LIMIT = 50
BUDGET = 5
BEAM = 5
TEMPERATURE = 0.8

data_fpath = "/gpfs/home6/dris/projects/Fetch/gsm8k/test_main.jsonl"
filename = f"{JOBID}_test_gsm8k_beamsearch_b{BUDGET}_t{TEMPERATURE}.pkl"
output_fpath = os.path.join(LOGDIR, filename)
policy_fpath = "xmu-nlp/Llama-3-8b-gsm8k"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(policy_fpath)
MAX_LEN_PER_STEP = 256

def assert_end(text):
    return "The answer is" in text and text.endswith(tokenizer.eos_token)

def fix_value(state):
    if state.parent and state.parent.content == state.content:
        state.value = -1
    if not state.content or len(tokenizer.tokenize(state.content)) > MAX_LEN_PER_STEP:
        state.value = -1
    if state.content.endswith(tokenizer.eos_token) and not assert_end(state.content):
        state.value = -1
    return state

def call_policy(question, path):
    url = "http://127.0.0.1:8000/v1/completions"
    prompt = f"Question: {question}\nAnswer:{path}"
    pload = {
        "prompt": prompt,
        "model": policy_fpath,
        "temperature": TEMPERATURE,
        "max_tokens": 512,
        "stop": ["\n"],
        "include_stop_str_in_output": True,
        "skip_special_tokens": False
    }
    resp = requests.post(url, json=pload); resp.raise_for_status()
    text = resp.json()["choices"][0]["text"]
    return text, len(tokenizer.tokenize(text))

def call_value(question, path):
    query_text = f"Question: {question}\nAnswer:{path}"
    if USE_ENSEMBLE_HOST:
        # Single ensemble endpoint returns averaged score already
        resp = requests.post(ENSEMBLE_URL, json={"texts": [query_text]})
        resp.raise_for_status()
        val = resp.json()["values"][0]
        return (min(max(val, -1.), 1.) + 1.) / 2
    else:
        # Query both single-model hosts and average client-side
        scores = []
        for _ in SINGLE_HOST_URLS:
            url = next(single_cycle)
            resp = requests.post(url, json={"texts": [query_text]})
            resp.raise_for_status()
            scores.append(resp.json()["values"][0])
        vals = [(min(max(s, -1.), 1.) + 1.) / 2 for s in scores]
        return float(np.mean(vals))

#### Search Tree ####
class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content, self.value, self.parent = content, value, parent
        self.children, self.timestep, self.tree = [], timestep, tree
        self.is_leaf = is_leaf

    def return_path(self):
        if self.content is None: return []
        return (self.parent.return_path() if self.parent else []) + [self.content]

    def print_path(self):
        return "".join(self.return_path())

class Tree:
    def __init__(self, question, answer):
        self.question, self.answer = question, answer
        self.all_nodes = [Node(None, 0, None, 0, self)]
    def return_timestep(self):
        return max(node.timestep for node in self.all_nodes)
    def add_node(self, content, value, parent, is_leaf=False):
        node = Node(content, value, parent, parent.timestep+1, self, is_leaf)
        parent.children.append(node); self.all_nodes.append(node)
        return node
    def get_beam_to_expand(self, beam_size):
        ts = self.return_timestep()
        latest = [n for n in self.all_nodes if n.is_leaf or n.timestep==ts]
        beam = sorted(latest, key=lambda x: x.value, reverse=True)[:beam_size]
        return [n for n in beam if not n.is_leaf]

# Load dataset
dataset = [json.loads(l) for l in open(data_fpath)]
problems = [Tree(d["question"], d["answer"]) for d in dataset]

from multiprocessing import Pool
def worker(tree):
    q = tree.question
    policy_times, value_times, total_tokens = [], [], 0
    for _ in range(LIMIT):
        actions = tree.get_beam_to_expand(BEAM)
        if not actions: break
        for a in actions:
            for _ in range(BUDGET):
                path = a.print_path()
                try:
                    t0 = time.time()
                    nxt, toks = call_policy(q, path)
                    policy_times.append(time.time()-t0); total_tokens+=toks
                except:
                    continue
                try:
                    t1 = time.time()
                    val = call_value(q, path+nxt)
                    value_times.append(time.time()-t1)
                except:
                    continue
                state = tree.add_node(nxt, val, a, nxt.endswith(tokenizer.eos_token) and "The answer is" in nxt)
                fix_value(state)
    return {
        "tree": tree,
        "total_generated_tokens": total_tokens,
        "policy_times": policy_times,
        "value_times": value_times,
        "num_policy_calls": len(policy_times),
        "num_value_calls": len(value_times)
    }

# Run
pool = Pool(80)
results = list(tqdm(pool.imap_unordered(worker, problems), total=len(problems)))
pool.close()

# Aggregate and print
total_tokens = sum(r["total_generated_tokens"] for r in results)
all_pt = sum((r["policy_times"] for r in results), [])
all_vt = sum((r["value_times"] for r in results), [])
tp, tv = sum(r["num_policy_calls"] for r in results), sum(r["num_value_calls"] for r in results)

avg_tokens = total_tokens/len(problems) if problems else 0
avg_pt = sum(all_pt)/len(all_pt) if all_pt else 0
avg_vt = sum(all_vt)/len(all_vt) if all_vt else 0

print("\n=== OVERALL EXPERIMENT SUMMARY ===")
print(f"Total problems: {len(problems)}")
print(f"Total generated tokens: {total_tokens:,}")
print(f"Avg tokens/problem: {avg_tokens:.1f}")
print(f"Total policy calls: {tp:,}, avg time: {avg_pt:.3f}s")
print(f"Total value calls: {tv:,}, avg time: {avg_vt:.3f}s")

pickle.dump([r["tree"] for r in results], open(output_fpath, "wb"))
print(f"Saved output to: {output_fpath}")

import os
import json
import pickle
import csv
import numpy as np
import requests
import time
from tqdm import tqdm
import itertools
from transformers import AutoTokenizer

# Configuration flag
USE_ENSEMBLE_HOST = False

# Endpoints
ENSEMBLE_URL = "http://127.0.0.1:8002/predict"
SINGLE_HOST_URLS = [
    "http://127.0.0.1:8002/predict",
    "http://127.0.0.1:8003/predict"
]
single_cycle = itertools.cycle(SINGLE_HOST_URLS)

# Fetch the job ID (or fallback to timestamp)
JOBID = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOBID") or datetime.now().strftime("%Y%m%d%H%M%S")
LOGDIR = os.environ.get("LOGDIR", ".")

# Hyperparameters
LIMIT = 50
BUDGET = 5
BEAM = 5
TEMPERATURE = 0.8
DISTANCE = 0.15

# File paths
data_fpath = "/gpfs/home6/dris/projects/Fetch/gsm8k/test_main_small.jsonl"
# Build a filename with the ID embedded
filename = f"{JOBID}_test_gsm8k_beamsearch_merge_b{BUDGET}_t{TEMPERATURE}.pkl"
output_fpath = os.path.join(LOGDIR, filename)
policy_fpath = "xmu-nlp/Llama-3-8b-gsm8k"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(policy_fpath)
MAX_LEN_PER_STEP = 256

# Helper functions
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
    payload = {
        "prompt": prompt,
        "model": policy_fpath,
        "temperature": TEMPERATURE,
        "max_tokens": 512,
        "stop": ["\n"],
        "include_stop_str_in_output": True,
        "skip_special_tokens": False
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["text"]
    tokens = len(tokenizer.tokenize(text))
    return text, tokens

def call_value(question, path):
    query = f"Question: {question}\nAnswer:{path}"
    if USE_ENSEMBLE_HOST:
        resp = requests.post(ENSEMBLE_URL, json={"texts": [query]})
        resp.raise_for_status()
        val = resp.json()["values"][0]
        return (min(max(val, -1.), 1.) + 1.) / 2
    else:
        scores = []
        for _ in SINGLE_HOST_URLS:
            url = next(single_cycle)
            resp = requests.post(url, json={"texts": [query]})
            resp.raise_for_status()
            scores.append(resp.json()["values"][0])
        vals = [(min(max(s, -1.), 1.) + 1.) / 2 for s in scores]
        return float(np.mean(vals))

def clean_text(text):
    if text.endswith(tokenizer.eos_token):
        text = text[:-len(tokenizer.eos_token)]
    return text.strip()

def call_esm(texts):
    url = "http://127.0.0.1:8004/predict"
    texts = [clean_text(t) for t in texts]
    resp = requests.post(url, json={"texts": texts, "d": DISTANCE})
    resp.raise_for_status()
    return resp.json()["labels"]

# Tree and merge logic (unchanged except call_value integration)
class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content, self.value = content, value
        self.parent, self.timestep = parent, timestep
        self.tree = tree; self.children = []; self.is_leaf = is_leaf
        self.cache = []

    def print_path(self):
        return "".join(self.return_path())

    def return_path(self):
        if self.content is None: return []
        return (self.parent.return_path() if self.parent else []) + [self.content]

class VirtualNode:
    def __init__(self, nodes, parent=None):
        self.nodes = sorted(nodes, key=lambda x: x.value, reverse=True)
        self.tree = self.nodes[0].tree
        self.value = self.nodes[0].value
        self.visited = False
        self.children = []
        self.cache = []
        self.parent = parent
        self.is_leaf = self.nodes[0].is_leaf
        self.timestep = self.nodes[0].timestep

    def merge_nodes(self):
        groups = [
            [n for n in self.cache if n.is_leaf],
            [n for n in self.cache if not n.is_leaf]
        ]
        clusters = {}
        for gid, grp in enumerate(groups):
            if grp:
                labels = call_esm([n.content for n in grp])
                for n, lbl in zip(grp, labels):
                    key = (gid, lbl)
                    clusters.setdefault(key, []).append(n)
        for cluster in clusters.values():
            vnode = VirtualNode(cluster, self)
            self.children.append(vnode)
            self.tree.virtual_nodes.append(vnode)
        self.cache.clear()

class Tree:
    def __init__(self, question, answer):
        self.question, self.answer = question, answer
        root = Node(None, 0, None, 0, self)
        self.all_nodes = [root]
        self.virtual_nodes = [VirtualNode([root])]

    def return_timestep(self):
        return max(n.timestep for n in self.all_nodes)

    def add_node(self, content, value, parent, timestep, is_leaf=False):
        n = Node(content, value, parent, timestep, self, is_leaf)
        parent.children.append(n)
        self.all_nodes.append(n)
        return n

    def get_beam_to_expand(self, beam_size):
        ts = self.return_timestep()
        candidates = [vn for vn in self.virtual_nodes if vn.is_leaf or vn.timestep == ts]
        beam = sorted(candidates, key=lambda v: v.value, reverse=True)[:beam_size]
        return [v for v in beam if not v.is_leaf]

def load_csv_data(csv_fpath):
    problems = []
    with open(csv_fpath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create Tree with question and answer
            tree_obj = Tree(row['Question'], row['Correct Answer'])
            
            # Store additional metadata as attributes
            tree_obj.incorrect_answers = [
                row['Incorrect Answer 1'],
                row['Incorrect Answer 2'],
                row['Incorrect Answer 3']
            ]
            tree_obj.explanation = row['Explanation']
            
            problems.append(tree_obj)
    return problems

# Usage
problems = load_csv_data('test_data.csv')

# Worker function
def worker(tree):
    q = tree.question
    policy_times, value_times, esm_times = [], [], []
    total_tokens = 0

    for step in range(LIMIT):
        clusters = tree.get_beam_to_expand(BEAM)
        if not clusters: break
        for cl in clusters:
            for b in range(BUDGET):
                node = cl.nodes[b % len(cl.nodes)]
                path = node.print_path()
                try:
                    t0 = time.time()
                    nxt, toks = call_policy(q, path)
                    policy_times.append(time.time()-t0)
                    total_tokens += toks
                except:
                    continue
                try:
                    t1 = time.time()
                    val = call_value(q, path+nxt)
                    value_times.append(time.time()-t1)
                except:
                    continue
                new_node = tree.add_node(nxt, val, node, step+1, assert_end(nxt))
                fix_value(new_node)
                if new_node.value > -1:
                    cl.cache.append(new_node)
            if cl.cache:
                t2 = time.time()
                cl.merge_nodes()
                esm_times.append(time.time()-t2)

    return {
        "tree": tree,
        "total_generated_tokens": total_tokens,
        "policy_times": policy_times,
        "value_times": value_times,
        "esm_times": esm_times,
        "num_policy_calls": len(policy_times),
        "num_value_calls": len(value_times),
        "num_esm_calls": len(esm_times)
    }

# Run in parallel
from multiprocessing import Pool
pool = Pool(80)
results = list(tqdm(pool.imap_unordered(worker, problems), total=len(problems)))
pool.close()

# Aggregate & print summary
total_tokens = sum(r["total_generated_tokens"] for r in results)
all_pt = sum((r["policy_times"] for r in results), [])
all_vt = sum((r["value_times"] for r in results), [])
all_et = sum((r["esm_times"] for r in results), [])
tp = sum(r["num_policy_calls"] for r in results)
tv = sum(r["num_value_calls"] for r in results)
te = sum(r["num_esm_calls"] for r in results)

print("\n=== MERGE SEARCH SUMMARY ===")
print(f"Problems: {len(problems)}")
print(f"Total tokens: {total_tokens:,}")
print(f"Avg policy calls: {tp/len(problems):.2f}, time: {np.mean(all_pt):.3f}s")
print(f"Avg value calls: {tv/len(problems):.2f}, time: {np.mean(all_vt):.3f}s")
print(f"Avg ESM calls: {te/len(problems):.2f}, time: {np.mean(all_et):.3f}s")

pickle.dump([r["tree"] for r in results], open(output_fpath, "wb"))
print(f"Saved output to: {output_fpath}")

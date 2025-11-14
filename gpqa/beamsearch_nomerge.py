import os
import json
import pickle
import numpy as np
import requests
import time
import csv
import datetime
from tqdm import tqdm
import itertools
from transformers import AutoTokenizer
from inference_analytics import InferenceAnalytics

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
JOBID = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOBID") or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
LOGDIR = os.environ.get("LOGDIR", ".")

# Hyperparameters
LIMIT = 15
BUDGET = 3
BEAM = 4
TEMPERATURE = 0.8
DISTANCE = 0.15

# File paths
data_fpath = "Fetch_git/gpqa/gpqa_eval_split.csv"

# Build a filename with the ID embedded
filename = f"{JOBID}_test_gpqa_beamsearch_nomerge_b{BUDGET}_t{TEMPERATURE}.pkl"
output_fpath = os.path.join(LOGDIR, filename)

policy_fpath = "meta-llama/Llama-3.1-8B-Instruct"

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

def call_policy(question, answer_choices, path):
    url = "http://127.0.0.1:8000/v1/completions"
    
    prompt = f"Question: {question}\n"
    for idx, choice in enumerate(answer_choices):
        letter = chr(ord('A') + idx)
        prompt += f"{letter}. {choice}\n"
    prompt += f"Answer:{path}"
    
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

def call_value(question, answer_choices, path):
    prompt = f"Question: {question}\n"
    for idx, choice in enumerate(answer_choices):
        letter = chr(ord('A') + idx)
        prompt += f"{letter}. {choice}\n"
    prompt += f"Answer:{path}"
    
    if USE_ENSEMBLE_HOST:
        resp = requests.post(ENSEMBLE_URL, json={"texts": [prompt]})
        resp.raise_for_status()
        val = resp.json()["values"][0]
        return (min(max(val, -1.), 1.) + 1.) / 2
    else:
        scores = []
        for _ in SINGLE_HOST_URLS:
            url = next(single_cycle)
            resp = requests.post(url, json={"texts": [prompt]})
            resp.raise_for_status()
            scores.append(resp.json()["values"][0])
        
        vals = [(min(max(s, -1.), 1.) + 1.) / 2 for s in scores]
        return float(np.mean(vals))

# Tree logic WITHOUT merging
class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content, self.value = content, value
        self.parent, self.timestep = parent, timestep
        self.tree = tree
        self.children = []
        self.is_leaf = is_leaf
    
    def print_path(self):
        return "".join(self.return_path())
    
    def return_path(self):
        if self.content is None:
            return []
        return (self.parent.return_path() if self.parent else []) + [self.content]

class Tree:
    def __init__(self, question, answer, answer_choices):
        self.question = question
        self.answer = answer
        self.answer_choices = answer_choices
        root = Node(None, 0, None, 0, self)
        self.all_nodes = [root]
        self.leaf_nodes = []  # Track leaf nodes separately
    
    def return_timestep(self):
        return max(n.timestep for n in self.all_nodes)
    
    def add_node(self, content, value, parent, timestep, is_leaf=False):
        n = Node(content, value, parent, timestep, self, is_leaf)
        parent.children.append(n)
        self.all_nodes.append(n)
        if is_leaf:
            self.leaf_nodes.append(n)
        return n
    
    def get_beam_to_expand(self, beam_size):
        """
        WITHOUT MERGING: Simply select top BEAM candidates by value
        from all non-leaf nodes at current timestep.
        """
        ts = self.return_timestep()
        # Get all non-leaf nodes (can be expanded further)
        expandable = [n for n in self.all_nodes if not n.is_leaf and n.timestep <= ts]
        
        # Sort by value and take top BEAM
        beam = sorted(expandable, key=lambda n: n.value, reverse=True)[:beam_size]
        return beam

def load_csv_data(csv_fpath):
    """
    Load CSV data with format:
    Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3, Explanation
    """
    problems = []
    with open(csv_fpath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Combine all answer choices: Correct Answer + 3 Incorrect Answers
            answer_choices = [
                row['Correct Answer'],
                row['Incorrect Answer 1'],
                row['Incorrect Answer 2'],
                row['Incorrect Answer 3']
            ]
            
            # Create Tree with question, answer, and answer choices
            tree_obj = Tree(
                row['Question'],
                row['Correct Answer'],
                answer_choices
            )
            
            # Store additional metadata as attributes
            tree_obj.explanation = row['Explanation']
            problems.append(tree_obj)
    
    return problems

# Usage
problems = load_csv_data(data_fpath)
problems = problems[:100]  # Keep only first 3

# Worker function
def worker(tree):
    q = tree.question
    answer_choices = tree.answer_choices
    
    policy_times, value_times = [], []
    total_tokens = 0
    
    for step in range(LIMIT):
        # Get beam candidates WITHOUT any merging
        beam = tree.get_beam_to_expand(BEAM)
        if not beam:
            break
        
        for node in beam:
            for b in range(BUDGET):
                path = node.print_path()
                
                try:
                    t0 = time.time()
                    nxt, toks = call_policy(q, answer_choices, path)
                    policy_times.append(time.time() - t0)
                    total_tokens += toks
                except Exception as e:
                    print(f"Policy call error: {e}")
                    continue
                
                try:
                    t1 = time.time()
                    val = call_value(q, answer_choices, path + nxt)
                    value_times.append(time.time() - t1)
                except Exception as e:
                    print(f"Value call error: {e}")
                    continue
                
                # Add node directly to tree
                new_node = tree.add_node(nxt, val, node, step + 1, assert_end(nxt))
                fix_value(new_node)
    
    return {
        "tree": tree,
        "total_generated_tokens": total_tokens,
        "policy_times": policy_times,
        "value_times": value_times,
        "num_policy_calls": len(policy_times),
        "num_value_calls": len(value_times),
        "num_leaf_nodes": len(tree.leaf_nodes),
        "num_total_nodes": len(tree.all_nodes)
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

tp = sum(r["num_policy_calls"] for r in results)
tv = sum(r["num_value_calls"] for r in results)
total_leaf = sum(r["num_leaf_nodes"] for r in results)
total_all = sum(r["num_total_nodes"] for r in results)

print("\n=== BASELINE: BEAMSEARCH WITHOUT MERGING ===")
print(f"Problems: {len(problems)}")
print(f"Total tokens: {total_tokens:,}")
print(f"Total nodes generated: {total_all:,}")
print(f"Leaf nodes (completed): {total_leaf:,}")
print(f"Avg policy calls per problem: {tp/len(problems):.2f}, time: {np.mean(all_pt):.3f}s")
print(f"Avg value calls per problem: {tv/len(problems):.2f}, time: {np.mean(all_vt):.3f}s")

analytics = InferenceAnalytics()
for result in results:
    analytics.add_result(result["tree"], result)

report = analytics.generate_summary(
    [(r["tree"], r) for r in results],
    mode="baseline"
)
print(report)

with open(os.path.join(LOGDIR, f"{JOBID}_baseline_report.txt"), "w") as f:
    f.write(report)

pickle.dump([r["tree"] for r in results], open(output_fpath, "wb"))

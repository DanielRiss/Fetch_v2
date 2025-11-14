import os
import json
import pickle
import numpy as np
import requests
import time
import csv
from tqdm import tqdm
import itertools
from transformers import AutoTokenizer
from datetime import datetime
from multiprocessing import Pool
import argparse


# ============================================================================
# CONFIGURATION
# ============================================================================

# Server endpoints
POLICY_SERVER = os.environ.get("POLICY_SERVER", "http://127.0.0.1:8000")
VERIFIER_URLS = [
    os.environ.get("VERIFIER_1", "http://127.0.0.1:8002/predict"),
    os.environ.get("VERIFIER_2", "http://127.0.0.1:8003/predict"),
]

# Get SLURM info for logging
JOBID = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d%H%M%S")
LOGDIR = os.environ.get("LOGDIR", f"./logs_{JOBID}")
os.makedirs(LOGDIR, exist_ok=True)

# ============================================================================
# ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description="Beam Search for Game of 24")
parser.add_argument("--policy_server", default="http://127.0.0.1:8000", help="Policy server URL")
parser.add_argument("--verifier_1", default="http://127.0.0.1:8002/predict", help="Verifier 1 URL")
parser.add_argument("--verifier_2", default="http://127.0.0.1:8003/predict", help="Verifier 2 URL")
parser.add_argument("--data_path", default="go24-benchmark.csv", help="Path to benchmark data")
parser.add_argument("--limit", type=int, default=50, help="Max search depth per problem")
parser.add_argument("--budget", type=int, default=5, help="Budget per node")
parser.add_argument("--beam", type=int, default=5, help="Beam width")
parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
parser.add_argument("--num_problems", type=int, default=0, help="Number of problems (0 = all)")
parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
parser.add_argument("--verbose", action="store_true", help="Verbose output")

args = parser.parse_args()

# Update global config from args
POLICY_SERVER = args.policy_server
VERIFIER_URLS = [args.verifier_1, args.verifier_2]
LIMIT = args.limit
BUDGET = args.budget
BEAM = args.beam
TEMPERATURE = args.temperature
NUM_PROBLEMS = args.num_problems
NUM_WORKERS = args.workers

POLICY_MODEL = "ftm_go24_policy_merged"
HELPER_PROMPT = """You are solving the Game of 24. Given 4 numbers, use +, -, *, / to reach 24.
Each number must be used exactly once. Provide step-by-step solution.

"""

MAX_LEN_PER_STEP = 256
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)

print(f"\n{'='*70}")
print(f"BEAM SEARCH - Game of 24")
print(f"{'='*70}")
print(f"Policy server: {POLICY_SERVER}")
print(f"Verifier 1: {VERIFIER_URLS[0]}")
print(f"Verifier 2: {VERIFIER_URLS[1]}")
print(f"Limit: {LIMIT}")
print(f"Budget: {BUDGET}")
print(f"Beam: {BEAM}")
print(f"Problems to run: {NUM_PROBLEMS if NUM_PROBLEMS > 0 else 'ALL'}")
print(f"Workers: {NUM_WORKERS}")
print(f"Job ID: {JOBID}")
print(f"Log dir: {LOGDIR}")


# ============================================================================
# POLICY SERVER CALLS
# ============================================================================

def call_policy(question: str, path: str) -> tuple:
    """Call policy server"""
    url = f"{POLICY_SERVER}/v1/completions"

    try:
        numbers = [float(x.strip()) for x in question.split()]
        input_str = " ".join(str(int(n) if n == int(n) else n) for n in numbers)
    except:
        input_str = question
    prompt = f"Input: {input_str}\nPossible next steps:{path}"



    payload = {
        "prompt": prompt,
        "model": POLICY_MODEL,
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


# ============================================================================
# VERIFIER SERVER CALLS
# ============================================================================

verifier_cycle = itertools.cycle(VERIFIER_URLS)


def call_verifier(question: str, path: str) -> float:
    """Call verifier servers (round-robin)"""
    query = f"{HELPER_PROMPT}Question: {question}\nAnswer:{path}"

    scores = []
    for url in VERIFIER_URLS:
        try:
            resp = requests.post(url, json={"texts": [query]})
            resp.raise_for_status()
            score = resp.json()["values"][0]
            scores.append((min(max(score, -1.), 1.) + 1.) / 2)
        except:
            continue

    return float(np.mean(scores)) if scores else 0.0


# ============================================================================
# TREE DATA STRUCTURES
# ============================================================================

class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content, self.value = content, value
        self.parent, self.timestep = parent, timestep
        self.tree = tree
        self.children = []
        self.cache = []
        self.is_leaf = is_leaf

    def print_path(self):
        return "".join(self.return_path())

    def return_path(self):
        if self.content is None:
            return []
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
        """Merge nodes (simple clustering)"""
        groups = [
            [n for n in self.cache if n.is_leaf],
            [n for n in self.cache if not n.is_leaf]
        ]
        clusters = {}
        for gid, grp in enumerate(groups):
            if grp:
                for n in grp:
                    key = (gid, 0)
                    clusters.setdefault(key, []).append(n)

        for cluster in clusters.values():
            vnode = VirtualNode(cluster, self)
            self.children.append(vnode)
            self.tree.virtual_nodes.append(vnode)
        self.cache.clear()


class Tree:
    def __init__(self, question):
        self.question = question
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


# ============================================================================
# BEAM SEARCH WORKER
# ============================================================================

def assert_end(text):
    """Check if text is valid final answer"""
    return "The answer is" in text and text.endswith(tokenizer.eos_token)


def fix_value(state):
    """Validate and fix node value"""
    if state.parent and state.parent.content == state.content:
        state.value = -1
    if not state.content or len(tokenizer.tokenize(state.content)) > MAX_LEN_PER_STEP:
        state.value = -1
    if state.content.endswith(tokenizer.eos_token) and not assert_end(state.content):
        state.value = -1
    return state


def worker(tree):
    """Worker function for one problem"""
    q = tree.question
    policy_times, verifier_times = [], []
    total_tokens = 0

    for step in range(LIMIT):
        clusters = tree.get_beam_to_expand(BEAM)
        if not clusters:
            break

        for cl in clusters:
            for b in range(BUDGET):
                node = cl.nodes[b % len(cl.nodes)]
                path = node.print_path()

                # Call policy
                try:
                    t0 = time.time()
                    nxt, toks = call_policy(q, path)
                    policy_times.append(time.time() - t0)
                    total_tokens += toks
                except Exception as e:
                    if args.verbose:
                        print(f"Policy error: {e}")
                    continue

                # Call verifier
                try:
                    t1 = time.time()
                    val = call_verifier(q, path + nxt)
                    verifier_times.append(time.time() - t1)
                except Exception as e:
                    if args.verbose:
                        print(f"Verifier error: {e}")
                    val = 0.0

                # Add node
                new_node = tree.add_node(nxt, val, node, step + 1, assert_end(nxt))
                fix_value(new_node)

                if new_node.value > -1:
                    cl.cache.append(new_node)

            # Merge
            if cl.cache:
                cl.merge_nodes()

    return {
        "tree": tree,
        "total_generated_tokens": total_tokens,
        "policy_times": policy_times,
        "verifier_times": verifier_times,
        "num_policy_calls": len(policy_times),
        "num_verifier_calls": len(verifier_times),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    problems = []

    try:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for game in row:
                    problems.append(Tree(game.strip()))
    except FileNotFoundError:
        print(f"Error: {args.data_path} not found")
        print("Creating sample problems...")
        problems = [
            Tree("1 1 4 6"),
            Tree("2 3 4 5"),
            Tree("1 2 3 4"),
        ]

    # LIMIT to N problems if specified
    if NUM_PROBLEMS > 0:
        problems = problems[:NUM_PROBLEMS]
        print(f"Limited to {NUM_PROBLEMS} problems")

    total_problems = len(problems)
    print(f"Running beam search on {total_problems} problems")

    # Run beam search in parallel
    print(f"\nStarting beam search (workers={NUM_WORKERS})...")
    pool = Pool(NUM_WORKERS)
    results = list(tqdm(pool.imap_unordered(worker, problems), total=total_problems))
    pool.close()
    pool.join()

    # Aggregate results
    print("\n" + "="*70)
    print("BEAM SEARCH RESULTS")
    print("="*70)

    total_tokens = sum(r["total_generated_tokens"] for r in results)
    all_pt = sum((r["policy_times"] for r in results), [])
    all_vt = sum((r["verifier_times"] for r in results), [])
    tp = sum(r["num_policy_calls"] for r in results)
    tv = sum(r["num_verifier_calls"] for r in results)

    print(f"Problems: {total_problems}")
    print(f"Total tokens: {total_tokens:,}")
    if all_pt:
        print(f"\nPolicy:")
        print(f"  Total calls: {tp}")
        print(f"  Avg time per call: {np.mean(all_pt):.3f}s")
        print(f"  Total time: {sum(all_pt):.1f}s")
    if all_vt:
        print(f"\nVerifier:")
        print(f"  Total calls: {tv}")
        print(f"  Avg time per call: {np.mean(all_vt):.3f}s")
        print(f"  Total time: {sum(all_vt):.1f}s")

    # Save results
    output_file = os.path.join(LOGDIR, f"beamsearch_results_{JOBID}.pkl")
    pickle.dump([r["tree"] for r in results], open(output_file, "wb"))
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = main()
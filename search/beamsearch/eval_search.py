import pickle
import argparse
import os

class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.value = value
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf

class VirtualNode:
    def __init__(self, nodes, parent=None):
        self.nodes = sorted(nodes, key=lambda x: x.value, reverse=True)
        self.tree = self.nodes[0].tree
        self.value = self.nodes[0].value
        self.visited = False
        self.children = []
        self.cache = []
        self.parent = parent

class Tree:
    def __init__(self, question, options, answer):
        self.question = question
        self.options = options
        self.answer = answer # provided, but will not used when searching
        self.all_nodes = []
        self.root = Node(None, 0, None, 0, self)
        self.all_nodes.append(self.root)

# 1) Command-line args
parser = argparse.ArgumentParser()
parser.add_argument("--jobid", required=True)
parser.add_argument("--budget", type=int, default=5)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--logdir", default=os.path.expanduser("~/projects/Fetch/search/beamsearch"))
parser.add_argument("--policy_fpath", default="xmu-nlp/Llama-3-8b-gsm8k")
parser.add_argument("--type", default="")
args = parser.parse_args()

# 2) Build paths
logs_dir = os.path.join(args.logdir, f"logs_{args.jobid}")
filename = f"{args.jobid}_test_gsm8k_beamsearch{args.type}_b{args.budget}_t{args.temperature}.pkl"
data_fpath = os.path.join(logs_dir, filename)

print(f"Loading pickle from: {data_fpath}")
if not os.path.isfile(data_fpath):
    raise FileNotFoundError(f"Cannot find {data_fpath}")

# 3) Load and sanity-check
with open(data_fpath, "rb") as f:
    problems = pickle.load(f)
print(f"Loaded {len(problems)} problem trees")
if len(problems) == 0:
    raise RuntimeError("Pickle contains zero problems")

# 4) Check first tree
first = problems[0]
print(f"First tree has {len(first.all_nodes)} total nodes; leaf nodes:",
      sum(1 for n in first.all_nodes if n.is_leaf),
      "; virtual nodes:",
      len(getattr(first, "virtual_nodes", [])))
policy_fpath = "xmu-nlp/Llama-3-8b-gsm8k"

with open(data_fpath, "rb") as f:
    problems = pickle.load(f)

import re

def extract_gold_answer(completion):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return "[INVALID]"
        return match_str
    else:
        return "[INVALID]"

def extract_pred_answer(text):
    PATTERN = re.compile(r"The answer is (.*)<|endoftext|>")
    match = PATTERN.search(text)
    if match is not None:
        return match.group(1).replace(",", "")
    else:
        return "[INVALID]"

def eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except:
        return False

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(policy_fpath)

def compute_cost(problem):
    return sum([len(tokenizer.tokenize(node.content)) for node in problem.all_nodes if node.content])

def select_node(problem):
    cnt = 0
    if hasattr(problem, "virtual_nodes"):
        for virtual_node in problem.virtual_nodes:
            if len(virtual_node.children) > 0:
                cnt += 1
    else:
        for node in problem.all_nodes:
            if len(node.children) > 0:
                cnt += 1
    return cnt

corr, total = 0, 0
not_finished = 0
tokens = 0
select_cnt = 0
for problem in problems:
    ref = extract_gold_answer(problem.answer) if "####" in problem.answer else problem.answer
    leaf_nodes = [node for node in problem.all_nodes if node.is_leaf]
    if leaf_nodes:
        best_node = max(leaf_nodes, key=lambda x: x.value)
        hyp = extract_pred_answer(best_node.content)
        if eq(hyp, ref):
            corr += 1
    else:
        not_finished += 1
    total += 1
    tokens += compute_cost(problem)
    select_cnt += select_node(problem)


print(f"{corr}/{total}={corr/total}")
print(f"{not_finished}/{total}={not_finished/total}")
print(f"{tokens/total}")
print(f"{select_cnt/total}")


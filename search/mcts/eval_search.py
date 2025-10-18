import pickle
import re
from mcts_tree import *
from mcts_tree_merge import *
# from grader import grade_answer
from tqdm import tqdm
import random

data_fpath = "path/to/pred"
MODEL_PATH = "path/to/policy"

with open(data_fpath, "rb") as f:
    problems = pickle.load(f)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

INVALID_ANS="[invalid]"
def extract_gold_answer(completion):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

def extract_pred_answer(completion):
    ANS_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion.strip().split("\n")[-1])
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

def read_step_num(tree):
    return len(tree.all_nodes) - 1 # -1 root

def read_token_num(tree):
    return sum([len(tokenizer.tokenize(node.content)) for node in tree.all_nodes if node.content])

def read_rollout_token_num(tree):
    rollout_token_num = 0
    for node in tree.all_nodes:
        for rollout in node.rollouts:
            if rollout and rollout["text"]:
                rollout_token_num += len(tokenizer.tokenize(rollout["text"]))
    return rollout_token_num

def read_skip_num(tree):
    return len(set([node.timestep for node in tree.all_nodes if node and node.parent and node.timestep - node.parent.timestep > 1]))

def return_max_depth(tree):
    return max([node.get_depth() for node in tree.all_nodes])

def return_best_path(tree):
    leaf_nodes = [node for node in tree.all_nodes if node.is_leaf]
    if leaf_nodes:
        best_node = max(leaf_nodes, key = lambda x: x.q() if len(x.rewards) else x.value)
        return "\n".join(best_node.return_path())
    else:
        try:
            best_node = max(tree.all_nodes, key = lambda x: x.q())
            best_rollout = max(best_node.rollouts, key = lambda x: x["value"])["text"]
            return "\n".join(best_node.return_path() + [best_rollout])
        except:
            return None

def return_greedy_path(tree):
    curr_node = tree.root
    while True:
        if curr_node.is_leaf:
            if not hasattr(curr_node, "sub_nodes"):
                return "\n".join(curr_node.return_path())
            else:
                return "\n".join(curr_node.sub_nodes[0].return_path())
        elif curr_node.actions:
            curr_node = max(curr_node.actions, key = lambda x: x.q())
        else:
            break
    if not hasattr(curr_node, "sub_nodes"):
        return max(curr_node.rollouts, key = lambda x: x["value"])["text"]
    else:
        return max(curr_node.sub_nodes[0].rollouts, key = lambda x: x["value"])["text"]

def select_node(problem):
    cnt = 0
    root = problem.root
    def get_actions(node):
        nodes = [node]
        for child in node.actions:
            nodes += get_actions(child)
        return nodes
    all_nodes = get_actions(root)
    for node in all_nodes:
        if len(node.actions) > 0:
            cnt += 1
    return cnt

def eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except:
        return False

correct = 0
total = 0
finished = 0
total_steps = 0
total_rollout_token_num = 0
total_token_nums = 0
skip_num = 0
depth = 0
select_cnt = 0
for problem in tqdm(problems):
    prediction = return_best_path(problem)
    if prediction is not None:
        finished += 1
        hyp = extract_pred_answer(prediction)
        ref = extract_gold_answer(problem.answer)
        if ref == "[invalid]":
            ref = problem.answer
        if eq(hyp, ref):
            correct += 1
    total += 1
    total_steps += read_step_num(problem)
    total_token_nums += read_token_num(problem)
    total_rollout_token_num += read_rollout_token_num(problem)
    skip_num += read_skip_num(problem)
    depth += return_max_depth(problem)
    select_cnt += select_node(problem)

print("Accuracy:", correct / total)
print("Finished:", finished / total, "Total:", total)
print("Avg Steps:", total_steps / total)
print("Avg Token Nums:", total_token_nums / total)
print("Avg Rollout Token Num:", total_rollout_token_num / total)
print("Sum of Token Nums:", (total_token_nums + total_rollout_token_num) / total)
print("Avg Skip Nums:", skip_num / total)
print("Avg Depth:", depth / total)
print("Select Node:", select_cnt / total)



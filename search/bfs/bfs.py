# this script is written in the early stage of this project. At that time, we use a slightly low-efficient implementation of multi-threading. Besides, we do not use the embedding model as a server.

import os
import sys
from call_joint_service import call
from search_tree import *
import json
import pickle
import numpy as np
import jsonlines


LIMIT=50
BUDGET=10
TEMPERATURE=0.8

print("BUDGET:", BUDGET)

# task dependent
def assert_end(text):
    return True if text.strip().split("\n")[-1].startswith("The answer is") else False

def fix_value(state):
    if state.parent is not None: # repeat
        if state.parent.content == state.content:
            state.value = 0
    if state.content is not None and (len(state.content) == 0 or len(state.content) > 1024): # too short or too long
        state.value = 0


SEQ_STOP_TOKENS = []
STEP_STOP_TOKENS = ["\n"]

CONTINUE = False
data_fpath = "gsm8k/test.jsonl" # path to the test set
output_fpath = f"test_gsm8k_bfs_b{BUDGET}_t{TEMPERATURE}.pkl" # path to the output file
policy_fpath = "path/to/llama/ckpt" # path to the policy model

if __name__ == '__main__':
    
    if CONTINUE:    
        problems = pickle.load(open(output_fpath, "rb"))
        start = max([problem.return_timestep() for problem in problems])
    else:
        dataset = []
        with open(data_fpath, "r") as f:
            for line in f.readlines():
                dataset.append(json.loads(line))

        problems = []
        for instance in dataset:
            question = instance["question"]
            answer = instance["answer"]
            problem = Tree(question, answer, additional_info={})
            problem.init_root_node(0)
            problems.append(problem)
        start = 0

    for i in range(start, LIMIT):
        questions = []
        anchors = []
        paths = []
        finished = 0
        for problem in problems:
            state = problem.select_best_node()
            if state is not None:
                anchors += [state] * BUDGET
                questions += [problem.question] * BUDGET
                paths += [state.print_path()] * BUDGET
            else:
                finished += 1

        print(f"iteration {i}")
        print(f"finished {finished} / {len(problems)}")

        if len(questions) == 0:
            break

        next_steps, next_values = call(questions, paths, [TEMPERATURE] * len(questions), [STEP_STOP_TOKENS] * len(questions))
        for state, next_step, next_value in zip(anchors, next_steps, next_values):
            child = state.tree.add_node(next_step, next_value, state, i + 1, assert_end(next_step))
            fix_value(child)

        pickle.dump(problems, open(output_fpath, "wb"))
            
# if unfinish, select 1 node and extend to the end
questions = []
anchors = []
paths = []
finished = 0
for problem in problems:
    state = problem.select_best_node()
    if state is not None:
        anchors += [state]
        questions += [problem.question]
        paths += [state.print_path()]
    else:
        finished += 1

if len(questions) != 0:
    print(f"iteration final")
    print(f"finished {finished} / {len(problems)}")

    next_steps, next_values = call(questions, paths, [0] * len(questions), [SEQ_STOP_TOKENS] * len(questions))
    for state, next_step, next_value in zip(anchors, next_steps, next_values):
        child = state.tree.add_node(next_step, next_value, state, LIMIT + 1, assert_end(next_step))
        fix_value(child)

    pickle.dump(problems, open(output_fpath, "wb"))


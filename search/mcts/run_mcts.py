import os
import sys
from gsm_config import *
from mcts_tree import *
import multiprocessing
import json
import pickle
import numpy as np
import math
import copy


CONTINUE = False

if __name__ == '__main__':
    
    if CONTINUE:    
        output_fpath = "path/to/prev/unfinish/file"
        problems = pickle.load(open(output_fpath, "rb"))
    else:
        config = GSMConfig()
        
        data_fpath = "gsm8k/test.json" # 
        output_fpath = f"test_mcts_rb{config.root_budget}_nb{config.node_budget}_c{config.c}_r{config.n_rollouts}_md{config.max_depth}_mt{config.min_terminals}_d{config.d}.pkl" #

        dataset = []
        with open(data_fpath, "r") as f:
            for line in f.readlines():
                dataset.append(json.loads(line))

        problems = []
        for instance in dataset:
            question = instance["question"]
            answer = instance["answer"]
            problem = MCTSTree(question, answer, config)
            problems.append(problem)

    def run_exp(problem):
        problem.run_mcts()
        return problem

    pool = multiprocessing.Pool(100)
    problems = list(tqdm(pool.imap_unordered(run_exp, problems), total=len(problems)))    
    pool.close()
    
    pickle.dump(problems, open(output_fpath, "wb"))


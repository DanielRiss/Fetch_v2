from tqdm import tqdm
import requests
import json
import multiprocessing
import re
import numpy as np
import random
import time
import copy
from string import Template

TEMPERATURE = 0.8
SEQ_STOP_TOKENS = []
STEP_STOP_TOKENS = ["\n"]

from transformers import AutoTokenizer
MODEL_PATH = "path/to/policy/ckpt"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class PolicyArgument:
    url='127.0.0.1'
    port=8000
    model_name=MODEL_PATH
    max_tokens=512

class ValueArgument:
    url='127.0.0.1'
    port=8002

class MergeArgument:
    url='127.0.0.1'
    port=8003

def assert_end(text):
    return True if text.strip().split("\n")[-1].startswith("The answer is") else False

POLICY_INSTRUCTION = Template("""Question: ${question}\nAnswer: ${path}""")

def wrap_query_for_policy(query, path):
    wrapped_query = POLICY_INSTRUCTION.substitute(question=query, path=path+"\n" if path else "")
    if wrapped_query[-1] != "\n":
        wrapped_query = wrapped_query.strip()
    return wrapped_query

def wrap_query_for_value(query, path):
    wrapped_query = POLICY_INSTRUCTION.substitute(question=query, path=path+"\n" if path else "")
    if wrapped_query[-1] != "\n":
        wrapped_query = wrapped_query.strip()
    if assert_end(wrapped_query):
        wrapped_query = wrapped_query.strip()
    return wrapped_query

class GSMConfig:
    def __init__(self):
        self.policy_args = PolicyArgument()
        self.policy_url = f"http://{self.policy_args.url}:{self.policy_args.port}/v1/completions"
        self.value_args = ValueArgument()
        self.value_url = f"http://{self.value_args.url}:{self.value_args.port}/predict"
        self.merge_args = MergeArgument()
        self.merge_url = f"http://{self.merge_args.url}:{self.merge_args.port}/predict"
        self.root_budget = 8
        self.node_budget = 4
        self.leaf_budget = 4
        self.c = 0.6
        self.alpha = 0.5 # weight
        self.n_rollouts = 2
        self.min_search_time = 40
        self.max_search_time = 400
        self.min_terminals = 10
        self.conf_high_value = 0.9
        self.max_depth = 5
        self.min_length = 1
        self.max_seq_len = 1024
        self.max_step_len = 256
        self.expand_strategy = "random" # "random" or "merge", no used now
        self.top_p = 0.25 # merge, no use now
        self.d = 0.15 # merge
    
    def call_policy(self, pload):
        try:
            response = requests.post(self.policy_url, json=pload)
            content = json.loads(response.content)["choices"][0]["text"]
        except Exception as e:
            print("Policy Server Error", e, response.content) # mostly because out of length
            content = ""
        return content 
        
    def call_value(self, pload):
        try:
            response = requests.post(self.value_url, json=pload)
            value = response.json()["values"][0]
            value = (max(min(value, 1), -1) + 1) / 2 # normalize
        except Exception as e:
            print("Value Server Error", e, response.content)
            value = 0 # usually because oom
        return value

    def extract_first_step(self, rollout):
        steps = rollout.split("\n")
        if len(steps) > 1:
            return steps[0], "\n".join(steps[1:])
        else:
            return rollout, None

    def concat_steps(self, steps):
        return "\n".join(steps)

    def get_next_step(self, question, steps, greedy=False):
        answer = self.concat_steps(steps)
        query = wrap_query_for_policy(question, answer)
        temperature = 0 if greedy else TEMPERATURE
        pload ={"prompt": query, "model": self.policy_args.model_name, "temperature": temperature,
                "max_tokens": self.policy_args.max_tokens, "stop": STEP_STOP_TOKENS,}
        return self.call_policy(pload).strip()

    def get_full_traj(self, question, steps=[], greedy=False):
        answer = self.concat_steps(steps)
        query = wrap_query_for_policy(question, answer)
        temperature = 0 if greedy else TEMPERATURE
        pload ={"prompt": query, "model": self.policy_args.model_name, "temperature": temperature,
                "max_tokens": self.policy_args.max_tokens, "stop": SEQ_STOP_TOKENS}
        return self.call_policy(pload).strip()

    def get_value(self, question, steps=[]):
        answer = self.concat_steps(steps)
        query = wrap_query_for_value(question, answer)
        if self.check_seq_length(query):
            pload = {"texts": [query]}
            return self.call_value(pload)
        else:
            return 0.

    def is_terminal(self, answer):
        return assert_end(answer)
    
    def compute_reward(self, question, steps, rollouts):
        rollout_values = [self.get_value(question, steps + [rollout]) for rollout in rollouts]
        new_rollouts = [{"text": rollout, "value": rollout_value} for rollout, rollout_value in zip(rollouts, rollout_values)]
        return new_rollouts

    def is_acceptable(self, new_action, actions):
        pload = {"text_A": [new_action], "text_B": actions}
        result = requests.post(self.merge_url, json=pload)
        scores = result.json()["scores"][0]
        min_score = min(scores)
        return min_score < self.sim_threshold

    def check_seq_length(self, text):
        text_len = len(tokenizer.tokenize(text))
        return text_len < self.max_seq_len

    def prior(self, text):
        text_len = len(tokenizer.tokenize(text))
        return 1 if text_len < self.max_step_len else 0 # too messy

    def cluster(self, texts):
        pload = {"texts": texts, "d": self.d}
        result = requests.post(self.merge_url, json=pload)
        return result.json()["labels"]

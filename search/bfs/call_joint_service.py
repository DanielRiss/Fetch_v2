from tqdm import tqdm
import requests
import json
import multiprocessing
import re
import numpy as np
import random
import time
import copy

class PolicyArgument:
    url='127.0.0.1'
    port=8000
    model_name="path/to/policy"
    max_tokens=512

class ValueArgument:
    url='127.0.0.1'
    port=8002

from string import Template

POLICY_INSTRUCTION = Template("""Question: ${question}\nAnswer: ${path}""")
VALUE_INSTRUCTION = Template("""Question: ${question}\nAnswer: ${path}""")

# task dependent
def assert_end(text):
    return True if text.split("\n")[-1].startswith("The answer is") else False

def wrap_query_for_policy(query, path):
    wrapped_query = POLICY_INSTRUCTION.substitute(question=query, path=path+"\n" if path else "")
    if wrapped_query[-1] != "\n":
        wrapped_query = wrapped_query.strip()
    return wrapped_query

def wrap_query_for_value(query, path):
    wrapped_query = VALUE_INSTRUCTION.substitute(question=query, path=path+"\n" if path else "")
    if wrapped_query[-1] != "\n":
        wrapped_query = wrapped_query.strip()
    if assert_end(wrapped_query):
        wrapped_query = wrapped_query.strip()
    return wrapped_query

class Worker(object):

    def __init__(self, policy_args, value_args):
        self.policy_args = policy_args
        self.policy_url = f"http://{policy_args.url}:{policy_args.port}/v1/completions"
        self.value_args = value_args
        self.value_url = f"http://{value_args.url}:{value_args.port}/predict"
    
    def encode_wrapper(self, args):
        return self.encode(*args)
    
    def encode(self, question, path, temp=1.0, stop=[]):
        headers ={"User-Agent":"Test Client"}

        if temp is not None: # None, only call value
            query = wrap_query_for_policy(question, path)
            pload ={"prompt": query, "model": self.policy_args.model_name, "temperature": temp,
                    "max_tokens": self.policy_args.max_tokens, "stop": stop}

            response =requests.post(self.policy_url, headers=headers, json=pload)
            
            try:
                next_step = json.loads(response.content)["choices"][0]["text"].strip()
            except Exception as e:
                print("Policy Server Error", e, response.content)
                return "", 0 # usually because over-length
        else:
            next_step = ""

        new_path = path + "\n" + next_step if path else next_step
        pload ={"texts": [wrap_query_for_value(question, new_path)]}
        
        response = requests.post(self.value_url, headers=headers, json=pload)
        
        try:
            value = (min(max(response.json()["values"][0], -1.), 1.) + 1.) / 2
        except Exception as e:
            print("Value Server Error", e, response.content)
            return "", 0 # usually because oom

        return next_step, value


def call(questions, paths, temperatures, stops):
    policy_args = PolicyArgument()
    value_args = ValueArgument()
    worker = Worker(policy_args, value_args)
    pool = multiprocessing.Pool(80)
    model_outputs = list(tqdm(pool.imap(worker.encode_wrapper, [(a, b, c, d) for a, b, c, d in zip(questions, paths, temperatures, stops)], 4), total=len(questions)))
    next_steps, values = zip(*model_outputs)
    return next_steps, values

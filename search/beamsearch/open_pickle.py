import pickle
import re
import csv
import numpy as np  # For quartiles and std

class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.value = value
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf

    def get_depth(self):
        return len(self.return_path()) + 1

    def return_path(self):
        if self.content is None:
            return []
        if self.parent is None:
            return [self.content]
        return self.parent.return_path() + [self.content]

    def print_path(self):
        return "".join(self.return_path())

class Tree:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.all_nodes = []
        self.root = Node(None, 0, None, 0, self)
        self.all_nodes.append(self.root)

    def return_timestep(self):
        return max([node.timestep for node in self.all_nodes])

    def add_node(self, content, value, parent, is_leaf=False):
        node = Node(content, value, parent, parent.timestep + 1, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        return node

    def get_beam_to_expand(self, beam_size=5):
        curr_timestep = self.return_timestep()
        latest_nodes = [node for node in self.all_nodes if node.is_leaf or node.timestep == curr_timestep]
        beam = sorted(latest_nodes, key=lambda x: x.value, reverse=True)[:beam_size]
        return [node for node in beam if not node.is_leaf]

    def get_best_leaf(self):
        leaves = [node for node in self.all_nodes if node.is_leaf]
        if not leaves:
            return None
        return max(leaves, key=lambda n: n.value)

with open('/home/dris/projects/Fetch/search/beamsearch/logs_15440383/15440383_test_gsm8k_beamsearch_b5_t0.8.pkl', 'rb') as f:
    problems = pickle.load(f)

def extract_numeric_answer(text):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        try:
            num = float(matches[-1])
            return int(num) if num.is_integer() else num
        except:
            return None
    return None

def compute_accuracy(problems, best_answers):
    correct = 0
    total = len(problems)

    for problem, best_answer in zip(problems, best_answers):
        true_answer_num = extract_numeric_answer(problem.answer)
        pred_answer_num = extract_numeric_answer(best_answer)
        if true_answer_num is not None and pred_answer_num is not None:
            if true_answer_num == pred_answer_num:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

def analyze_results(problems, best_answers, scores):
    accuracy = compute_accuracy(problems, best_answers)
    avg_score = np.mean(scores) if scores else 0
    min_score = np.min(scores) if scores else 0
    max_score = np.max(scores) if scores else 0
    std_score = np.std(scores) if scores else 0
    q1 = np.percentile(scores, 25) if scores else 0
    median = np.percentile(scores, 50) if scores else 0
    q3 = np.percentile(scores, 75) if scores else 0

    print(f"Total problems: {len(problems)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Score statistics (value estimates):")
    print(f"  Mean:   {avg_score:.3f}")
    print(f"  StdDev: {std_score:.3f}")
    print(f"  Min:    {min_score:.3f}")
    print(f"  Q1:     {q1:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  Q3:     {q3:.3f}")
    print(f"  Max:    {max_score:.3f}")

    # Save errors to CSV
    with open('beamsearch_errors.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Question', 'GroundTruth', 'Predicted', 'Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        errors_count = 0
        for prob, best_ans, score in zip(problems, best_answers, scores):
            true_answer_num = extract_numeric_answer(prob.answer)
            pred_answer_num = extract_numeric_answer(best_ans)
            if true_answer_num != pred_answer_num:
                writer.writerow({
                    'Question': prob.question,
                    'GroundTruth': prob.answer,
                    'Predicted': best_ans,
                    'Score': f"{score:.3f}",
                })
                errors_count += 1

    print(f"Errors saved to beamsearch_errors.csv (total {errors_count})")

best_answers = []
scores = []
for tree in problems:
    best_leaf = tree.get_best_leaf()
    if best_leaf:
        best_answers.append(best_leaf.print_path())
        scores.append(best_leaf.value)
    else:
        best_answers.append("")
        scores.append(0)

analyze_results(problems, best_answers, scores)

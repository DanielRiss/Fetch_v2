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


class VirtualNode:
    def __init__(self, nodes, parent=None):
        # Sort by descending value
        self.nodes = sorted(nodes, key=lambda x: x.value, reverse=True)
        self.tree = self.nodes[0].tree
        self.value = self.nodes[0].value
        self.visited = False
        self.children = []
        self.cache = []
        self.parent = parent
        self.is_leaf = self.nodes[0].is_leaf
        self.timestep = self.nodes[0].timestep

    def get_depth(self):
        return self.nodes[0].get_depth()


class Tree:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.all_nodes = []
        self.root = Node(None, 0, None, 0, self)
        # Initialize with a virtual node wrapping the root
        self.virtual_nodes = [VirtualNode([self.root])]
        self.all_nodes.append(self.root)

    def return_timestep(self):
        return max(node.timestep for node in self.all_nodes)

    def add_node(self, content, value, parent, timestep, is_leaf=False):
        node = Node(content, value, parent, timestep, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        return node

    def get_beam_to_expand(self, beam_size=5):
        curr_t = self.return_timestep()
        latest = [n for n in self.all_nodes if n.is_leaf or n.timestep == curr_t]
        beam = sorted(latest, key=lambda x: x.value, reverse=True)[:beam_size]
        return [n for n in beam if not n.is_leaf]

    def get_best_leaf(self):
        leaves = [n for n in self.all_nodes if n.is_leaf]
        return max(leaves, key=lambda n: n.value) if leaves else None

    def get_best_virtual_leaf(self):
        virtual_leaves = [v for v in self.virtual_nodes if v.is_leaf]
        return max(virtual_leaves, key=lambda v: v.value) if virtual_leaves else None


def extract_numeric_answer(text):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if not matches:
        return None
    try:
        num = float(matches[-1])
        return int(num) if num.is_integer() else num
    except:
        return None


def compute_accuracy(problems, answers):
    correct = 0
    for prob, ans in zip(problems, answers):
        true_num = extract_numeric_answer(prob.answer)
        pred_num = extract_numeric_answer(ans)
        if true_num is not None and pred_num is not None and true_num == pred_num:
            correct += 1
    return correct / len(problems) if problems else 0


def analyze_merge_search_stats(problems):
    total_vnodes = 0
    total_clusters = 0
    total_merged = 0
    sizes = []

    for tree in problems:
        total_vnodes += len(tree.virtual_nodes)
        for vnode in tree.virtual_nodes:
            size = len(vnode.nodes)
            sizes.append(size)
            if size > 1:
                total_clusters += 1
                total_merged += size

    print("\nMerge Search Statistics:")
    print(f"  Total virtual nodes: {total_vnodes}")
    print(f"  Total clusters merged: {total_clusters}")
    print(f"  Total nodes merged: {total_merged}")
    if sizes:
        print(f"  Avg cluster size: {np.mean(sizes):.2f}")
        print(f"  Max cluster size: {max(sizes)}")
        print(f"  Min cluster size: {min(sizes)}")


def analyze_results(problems, answers, scores, use_virtual=True):
    accuracy = compute_accuracy(problems, answers)
    avg = np.mean(scores) if scores else 0
    mn = np.min(scores) if scores else 0
    mx = np.max(scores) if scores else 0
    sd = np.std(scores) if scores else 0
    q1 = np.percentile(scores, 25) if scores else 0
    med = np.percentile(scores, 50) if scores else 0
    q3 = np.percentile(scores, 75) if scores else 0

    mode = "Virtual Node (Merge)" if use_virtual else "Regular Node"
    print(f"\nAnalysis using {mode} Search:")
    print(f"Total problems: {len(problems)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("Score statistics (value estimates):")
    print(f"  Mean:   {avg:.3f}")
    print(f"  StdDev: {sd:.3f}")
    print(f"  Min:    {mn:.3f}")
    print(f"  Q1:     {q1:.3f}")
    print(f"  Median: {med:.3f}")
    print(f"  Q3:     {q3:.3f}")
    print(f"  Max:    {mx:.3f}")

    if use_virtual:
        analyze_merge_search_stats(problems)

    csv_name = f"beamsearch_errors_{'virtual' if use_virtual else 'regular'}.csv"
    with open(csv_name, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["Question","GroundTruth","Predicted","Score","SearchType"])
        writer.writeheader()
        errors = 0
        for prob, ans, sc in zip(problems, answers, scores):
            tnum = extract_numeric_answer(prob.answer)
            pnum = extract_numeric_answer(ans)
            if tnum != pnum:
                writer.writerow({
                    "Question": prob.question,
                    "GroundTruth": prob.answer,
                    "Predicted": ans,
                    "Score": f"{sc:.3f}",
                    "SearchType": mode
                })
                errors += 1
    print(f"Errors saved to {csv_name} (total {errors})")


# Load results
with open("test_gsm8k_beamsearch_merge_b5_t0.8.pkl", "rb") as f:
    problems = pickle.load(f)

# Virtual node analysis
print("=== MERGE SEARCH ANALYSIS (Virtual Nodes) ===")
best_ans_v = []
scores_v = []
for tree in problems:
    bvn = tree.get_best_virtual_leaf()
    if bvn and bvn.nodes:
        best_ans_v.append(bvn.nodes[0].print_path())
        scores_v.append(bvn.value)
    else:
        bl = tree.get_best_leaf()
        if bl:
            best_ans_v.append(bl.print_path())
            scores_v.append(bl.value)
        else:
            best_ans_v.append("")
            scores_v.append(0)
analyze_results(problems, best_ans_v, scores_v, use_virtual=True)

print("\n" + "="*60)

# Regular node analysis
print("\n=== REGULAR SEARCH ANALYSIS (Individual Nodes) ===")
best_ans_r = []
scores_r = []
for tree in problems:
    bl = tree.get_best_leaf()
    if bl:
        best_ans_r.append(bl.print_path())
        scores_r.append(bl.value)
    else:
        best_ans_r.append("")
        scores_r.append(0)
analyze_results(problems, best_ans_r, scores_r, use_virtual=False)

print("\n" + "="*60)
print("\n=== COMPARISON ===")
acc_v = compute_accuracy(problems, best_ans_v)
acc_r = compute_accuracy(problems, best_ans_r)
print(f"Virtual Node Accuracy: {acc_v*100:.2f}%")
print(f"Regular Node Accuracy: {acc_r*100:.2f}%")
print(f"Improvement: {(acc_v-acc_r)*100:.2f} percentage points")
print(f"Virtual Avg Score: {np.mean(scores_v):.3f}")
print(f"Regular Avg Score: {np.mean(scores_r):.3f}")
print(f"Score Improvement: {(np.mean(scores_v)-np.mean(scores_r)):.3f}")


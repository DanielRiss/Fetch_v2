
import pickle
import os
import statistics
import csv

# Tree and merge logic (unchanged except call_value integration)
class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content, self.value = content, value
        self.parent, self.timestep = parent, timestep
        self.tree = tree; self.children = []; self.is_leaf = is_leaf
        self.cache = []

    def print_path(self):
        return "".join(self.return_path())

    def return_path(self):
        if self.content is None: return []
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
        groups = [
            [n for n in self.cache if n.is_leaf],
            [n for n in self.cache if not n.is_leaf]
        ]
        clusters = {}
        for gid, grp in enumerate(groups):
            if grp:
                labels = call_esm([n.content for n in grp])
                for n, lbl in zip(grp, labels):
                    key = (gid, lbl)
                    clusters.setdefault(key, []).append(n)
        for cluster in clusters.values():
            vnode = VirtualNode(cluster, self)
            self.children.append(vnode)
            self.tree.virtual_nodes.append(vnode)
        self.cache.clear()

class Tree:
    def __init__(self, question, answer, answer_choices):
        self.question = question
        self.answer = answer
        self.answer_choices = answer_choices  # NEW: Store answer choices
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

# ============================================
# CONFIGURATION
# ============================================

# Path to your pickle file
pkl_fpath = "/home/dris/projects/Fetch_git/search/beamsearch/15749110_test_gpqa_beamsearch_merge_b5_t0.8.pkl"

# Optional: filter results
VERBOSE = True  # Print detailed output


# ============================================
# 1. LOAD THE PICKLE FILE
# ============================================

def load_results(pkl_fpath):
    """Load trees from pickle file"""
    if not os.path.exists(pkl_fpath):
        raise FileNotFoundError(f"Pickle file not found: {pkl_fpath}")
    
    with open(pkl_fpath, "rb") as f:
        trees = pickle.load(f)
    
    print(f"✓ Loaded {len(trees)} trees from {pkl_fpath}\n")
    return trees


# ============================================
# 2. HELPER FUNCTIONS
# ============================================

def extract_answer_from_output(output_text):
    """
    Extract the multiple choice letter (A, B, C, D) from LLM output.
    """
    if not output_text:
        return None
    
    output_lower = output_text.lower()
    
    # Look for "answer is X" or "answer: X" patterns
    for letter in ['A', 'B', 'C', 'D']:
        patterns = [
            f"answer is {letter.lower()}",
            f"answer: {letter.lower()}",
            f"answer {letter.lower()}",
            f"the answer is {letter.lower()}",
            f"the answer: {letter.lower()}"
        ]
        for pattern in patterns:
            if pattern in output_lower:
                return letter
    
    # Fallback: just look for A, B, C, D in output
    for letter in ['A', 'B', 'C', 'D']:
        if letter in output_text:
            return letter
    
    return None


def get_leaf_nodes(node):
    """
    Recursively extract all leaf nodes from a tree node.
    """
    if node.is_leaf:
        return [node]
    
    leaves = []
    for child in node.children:
        leaves.extend(get_leaf_nodes(child))
    
    return leaves if leaves else [node]


def analyze_answer_quality(tree):
    """
    Analyze how well the LLM found the correct answer using ranking metrics.
    
    Returns dict with:
    - found_correct: Did correct answer appear anywhere?
    - top_1_correct: Is best answer the correct one?
    - rank_of_best_correct: Where does best-scoring correct answer rank?
    - score_ratio: How good is best-correct vs best-overall?
    - num_correct_outputs: How many leaves have correct answer?
    """
    # Get correct answer letter
    try:
        correct_choice = tree.answer_choices.index(tree.answer)
        correct_letter = chr(ord('A') + correct_choice)
    except (ValueError, AttributeError) as e:
        return {
            'error': str(e),
            'found_correct': False,
            'top_1_correct': False,
            'top_5_correct': False,
            'top_10_correct': False,
            'rank_of_best_correct': None,
            'score_of_best_correct': None,
            'score_of_best_overall': None,
            'score_ratio': 0.0,
            'num_correct_outputs': 0,
            'total_outputs': 0,
            'best_overall_answer': '?',
            'correct_letter': '?'
        }

    
    # Get all leaf nodes
    root_node = tree.all_nodes[0]
    leaf_nodes = get_leaf_nodes(root_node)
    
    # Extract all outputs with scores
    leaf_outputs = []
    for leaf in leaf_nodes:
        path = leaf.print_path()
        extracted_letter = extract_answer_from_output(path)
        leaf_outputs.append({
            'output': path,
            'extracted_letter': extracted_letter,
            'is_correct': extracted_letter == correct_letter,
            'value': leaf.value
        })
    
    # Sort by score (highest first)
    leaf_outputs_sorted = sorted(leaf_outputs, key=lambda x: x['value'], reverse=True)
    
    # Find where correct answer ranks
    correct_outputs = [lo for lo in leaf_outputs_sorted if lo['is_correct']]
    
    if not correct_outputs:
        # Correct answer not found at all
        return {
            'found_correct': False,
            'top_1_correct': False,
            'top_5_correct': False,
            'top_10_correct': False,
            'rank_of_best_correct': None,
            'score_of_best_correct': None,
            'score_of_best_overall': leaf_outputs_sorted[0]['value'] if leaf_outputs_sorted else None,
            'score_ratio': 0.0,
            'num_correct_outputs': 0,
            'total_outputs': len(leaf_outputs_sorted),
            'best_overall_answer': leaf_outputs_sorted[0]['extracted_letter'] if leaf_outputs_sorted else None
        }
    
    best_correct = correct_outputs[0]  # Best scoring correct answer
    best_overall = leaf_outputs_sorted[0]  # Best scoring output (any answer)
    
    # Find rank (position in sorted list, 1-indexed)
    rank = None
    for i, lo in enumerate(leaf_outputs_sorted):
        if lo is best_correct:
            rank = i + 1
            break
    
    # Score ratio: score of best correct / score of best overall
    score_ratio = best_correct['value'] / best_overall['value'] if best_overall['value'] != 0 else 0.0
    
    return {
        'found_correct': True,
        'top_1_correct': best_overall['is_correct'],
        'top_5_correct': any(lo['is_correct'] for lo in leaf_outputs_sorted[:5]),
        'top_10_correct': any(lo['is_correct'] for lo in leaf_outputs_sorted[:10]),
        'rank_of_best_correct': rank,
        'score_of_best_correct': best_correct['value'],
        'score_of_best_overall': best_overall['value'],
        'score_ratio': score_ratio,
        'num_correct_outputs': len(correct_outputs),
        'total_outputs': len(leaf_outputs_sorted),
        'best_overall_answer': best_overall['extracted_letter'],
        'correct_letter': correct_letter
    }


def evaluate_with_ranking(trees):
    """
    Evaluate all trees using ranking metrics.
    """
    metrics = {
        'found_correct': 0,
        'top_1_correct': 0,
        'top_5_correct': 0,
        'top_10_correct': 0,
        'rank_list': [],
        'score_ratio_list': [],
        'total_trees': len(trees)
    }
    
    detailed_results = []
    
    for idx, tree in enumerate(trees):
        analysis = analyze_answer_quality(tree)
        detailed_results.append(analysis)
        
        if analysis['found_correct']:
            metrics['found_correct'] += 1
        if analysis['top_1_correct']:
            metrics['top_1_correct'] += 1
        if analysis['top_5_correct']:
            metrics['top_5_correct'] += 1
        if analysis['top_10_correct']:
            metrics['top_10_correct'] += 1
        
        if analysis['rank_of_best_correct']:
            metrics['rank_list'].append(analysis['rank_of_best_correct'])
        if analysis['score_ratio'] > 0:
            metrics['score_ratio_list'].append(analysis['score_ratio'])
    
    return {
        'total_trees': len(trees),
        'found_anywhere': metrics['found_correct'],
        'found_anywhere_pct': metrics['found_correct'] / len(trees) * 100,
        'top_1_accuracy': metrics['top_1_correct'] / len(trees) * 100,
        'top_5_accuracy': metrics['top_5_correct'] / len(trees) * 100,
        'top_10_accuracy': metrics['top_10_correct'] / len(trees) * 100,
        'avg_rank': statistics.mean(metrics['rank_list']) if metrics['rank_list'] else None,
        'median_rank': statistics.median(metrics['rank_list']) if metrics['rank_list'] else None,
        'min_rank': min(metrics['rank_list']) if metrics['rank_list'] else None,
        'max_rank': max(metrics['rank_list']) if metrics['rank_list'] else None,
        'avg_score_ratio': statistics.mean(metrics['score_ratio_list']) if metrics['score_ratio_list'] else None,
        'median_score_ratio': statistics.median(metrics['score_ratio_list']) if metrics['score_ratio_list'] else None,
        'min_score_ratio': min(metrics['score_ratio_list']) if metrics['score_ratio_list'] else None,
        'max_score_ratio': max(metrics['score_ratio_list']) if metrics['score_ratio_list'] else None,
        'detailed_results': detailed_results
    }


def print_overall_statistics(stats):
    """Print overall statistics"""
    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS - RANKING-BASED ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Total Trees: {stats['total_trees']}")
    
    print(f"\n--- CORRECT ANSWER FOUND ---")
    print(f"Anywhere in outputs: {stats['found_anywhere']}/{stats['total_trees']} ({stats['found_anywhere_pct']:.1f}%)")
    
    print(f"\n--- TOP-K ACCURACY (is best answer correct?) ---")
    print(f"Top-1 (Best answer is correct): {stats['top_1_accuracy']:.1f}%")
    print(f"Top-5: {stats['top_5_accuracy']:.1f}%")
    print(f"Top-10: {stats['top_10_accuracy']:.1f}%")
    
    if stats['avg_rank'] is not None:
        print(f"\n--- RANKING OF BEST CORRECT ANSWER ---")
        print(f"Mean rank: {stats['avg_rank']:.1f} (out of {stats['total_trees']} total outputs avg)")
        print(f"Median rank: {stats['median_rank']:.0f}")
        print(f"Range: {stats['min_rank']:.0f} to {stats['max_rank']:.0f}")
        print(f"(Lower is better: rank=1 means best answer is correct)")
    
    if stats['avg_score_ratio'] is not None:
        print(f"\n--- SCORE RATIO (correct vs best overall) ---")
        print(f"Mean ratio: {stats['avg_score_ratio']:.3f}")
        print(f"Median ratio: {stats['median_score_ratio']:.3f}")
        print(f"Range: {stats['min_score_ratio']:.3f} to {stats['max_score_ratio']:.3f}")
        print(f"(1.0 = best answer is correct; 0.5 = correct is half as good as best)")
    
    print(f"\n{'='*80}\n")


def print_tree_summary(idx, tree, analysis, verbose_level=1):
    """Print summary for a single tree"""
    if verbose_level < 1:
        return
    
    # Safety check: skip if analysis has error
    if 'error' in analysis:
        print(f"\n{'='*80}")
        print(f"TREE {idx} - ERROR")
        print(f"{'='*80}")
        print(f"Question: {getattr(tree, 'question', 'UNKNOWN')[:120]}...")
        print(f"Error: {analysis['error']}")
        return
    
    print(f"\n{'='*80}")
    print(f"TREE {idx}")
    print(f"{'='*80}")
    
    print(f"\nQuestion: {tree.question[:120]}...")
    
    if verbose_level >= 2:
        print(f"\nAnswer Choices:")
        if hasattr(tree, 'answer_choices'):
            for i, choice in enumerate(tree.answer_choices):
                letter = chr(ord('A') + i)
                is_correct = choice == tree.answer
                marker = " ← CORRECT ANSWER" if is_correct else ""
                print(f"  {letter}. {choice[:100]}{marker}")
    
    # Safely get values with defaults
    correct_letter = analysis.get('correct_letter', '?')
    best_overall_answer = analysis.get('best_overall_answer', '?')
    found_correct = analysis.get('found_correct', False)
    
    print(f"\nCorrect Answer: {correct_letter}")
    print(f"Best Overall Answer: {best_overall_answer}")
    
    if found_correct:
        print(f"\n✓ Correct answer found in outputs")
        top_1 = analysis.get('top_1_correct', False)
        print(f"  Top-1 (best answer is correct): {'YES ✓' if top_1 else 'NO ✗'}")
        
        rank = analysis.get('rank_of_best_correct')
        if rank:
            print(f"  Rank of best correct: {rank}")
        
        score_correct = analysis.get('score_of_best_correct')
        score_overall = analysis.get('score_of_best_overall')
        if score_correct is not None and score_overall is not None:
            print(f"  Score of best correct: {score_correct:.4f}")
            print(f"  Score of best overall: {score_overall:.4f}")
        
        ratio = analysis.get('score_ratio', 0.0)
        print(f"  Score ratio: {ratio:.3f}")
        
        num_correct = analysis.get('num_correct_outputs')
        total = analysis.get('total_outputs')
        if num_correct and total:
            print(f"  Number of correct outputs: {num_correct}/{total}")
    else:
        print(f"\n✗ Correct answer NOT found in outputs")
        print(f"  Best answer found: {best_overall_answer}")
        total = analysis.get('total_outputs', 0)
        print(f"  Total outputs: {total}")


def save_detailed_csv(stats, pkl_fpath):
    """Save detailed per-tree results to CSV"""
    csv_output = pkl_fpath.replace('.pkl', '_ranking_analysis.csv')
    
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Tree Index',
            'Found Correct',
            'Top-1 Correct',
            'Top-5 Correct',
            'Top-10 Correct',
            'Rank of Best Correct',
            'Score of Best Correct',
            'Score of Best Overall',
            'Score Ratio',
            'Num Correct Outputs',
            'Total Outputs'
        ])
        
        for idx, result in enumerate(stats['detailed_results']):
            writer.writerow([
                idx,
                result['found_correct'],
                result['top_1_correct'],
                result['top_5_correct'],
                result['top_10_correct'],
                result['rank_of_best_correct'] if result['rank_of_best_correct'] else '',
                result['score_of_best_correct'] if result['score_of_best_correct'] else '',
                result['score_of_best_overall'] if result['score_of_best_overall'] else '',
                result['score_ratio'],
                result['num_correct_outputs'],
                result['total_outputs']
            ])
    
    print(f"✓ Saved detailed results to: {csv_output}\n")
    return csv_output


# ============================================
# 4. MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Load results
    trees = load_results(pkl_fpath)
    
    # Evaluate all trees with ranking metrics
    stats = evaluate_with_ranking(trees)
    
    # Print overall statistics
    print_overall_statistics(stats)
    
    # Print detailed analysis for first few trees
    print(f"DETAILED ANALYSIS OF INDIVIDUAL TREES\n")
    
    # Show all trees if small dataset, otherwise show first 10
    num_to_show = 10
    
    for idx in range(num_to_show):
        print_tree_summary(idx, trees[idx], stats['detailed_results'][idx], verbose_level=1)
    
    if len(trees) > num_to_show:
        print(f"\n... and {len(trees) - num_to_show} more trees")
    
    # Save detailed CSV
    save_detailed_csv(stats, pkl_fpath)
    
    # Print interpretation guide
    print(f"\n{'='*80}")
    print(f"METRIC INTERPRETATION GUIDE")
    print(f"{'='*80}\n")
    
    interpretation = """
WHAT THESE METRICS MEAN:

1. TOP-1 ACCURACY (PRIMARY METRIC)
   → Is the best-scoring answer the correct one?
   → This is your TRUE accuracy for multiple choice
   → What percentage of time does the LLM rank the correct answer as #1?

2. AVERAGE RANK (SECONDARY METRIC)
   → Where does the best-scoring CORRECT answer rank?
   → Rank 1 = best answer is correct (perfect)
   → Rank 50 = correct answer is ranked 50th (LLM doesn't understand)
   → Rank 300+ = correct answer is almost last (basically random)

3. SCORE RATIO (TERTIARY METRIC)
   → How good is the best-correct-answer vs the best-overall-answer?
   → 0.95 = correct answer is almost as good as best (LLM is confident)
   → 0.50 = correct answer is half as good as best (LLM is uncertain)
   → 0.05 = correct answer barely scores (LLM doesn't know)

4. TOP-K ACCURACY
   → Top-5: Would you accept if model picks from top 5 rankings?
   → Top-10: Same but more lenient
   → Usually not primary metric for multiple choice


RECOMMENDED REPORTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY:   Top-1 Accuracy (simple, directly answerable)
SECONDARY: Average Rank (shows LLM understanding depth)
TERTIARY:  Score Ratio (shows LLM confidence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BENCHMARKS FOR GOOD PERFORMANCE:
• Top-1 Accuracy: > 50% (depending on task difficulty)
• Average Rank: < 10 (correct is usually in top 10)
• Score Ratio: > 0.7 (correct answer scores highly)

BENCHMARKS FOR BAD PERFORMANCE:
• Top-1 Accuracy: < 25%
• Average Rank: > 100 (correct usually ranks low)
• Score Ratio: < 0.3 (correct answer barely scores)
    """
    
    print(interpretation)
    print(f"{'='*80}\n")
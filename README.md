# FETCH Framework: Cross-Domain Evaluation on Game of 24 and GPQA

This repository contains code for evaluating the FETCH (Faster Exploration and Tree-Cut Heuristics) framework on two reasoning benchmarks: Game of 24 and GPQA. The implementation demonstrates how variance reduction and semantic merging techniques can improve Tree-of-Thought reasoning across different domains.

---

## Part 1: Game of 24 - Policy Fine-Tuning and Beam Search

Game of 24 is a mathematical reasoning task requiring systematic exploration of a large solution space. This section evaluates a fine-tuned Llama 3.1-8B policy model.

### What is Game of 24?

Given four numbers, combine them using +, −, ×, ÷ to reach exactly 24. Each number must be used exactly once.

**Example**: Input `1 1 4 6` → Solution `(6 - 1) × (4 + 1) = 24`

### Results & Limitations

**Accuracy: 0/100 (0%)**

The policy model achieved 0% accuracy on the benchmark (0/100 problems solved) despite fine-tuning on 300K arithmetic examples. Analysis revealed two key failure modes:

1. **Incomplete Reasoning Chains**: The model succeeds at 4→3→2 reductions but fails at the final 2→1 consolidation step. Mean tree depth: 3.24 (expected: 3.0).
2. **Policy Collapse at 2-Number States**: When facing two numbers, the model often stops exploring after the first failed move.

**Capacity Gap**: GPT-4 with Tree-of-Thought achieves 74% accuracy (Yao et al., 2023), while our 8B model achieves 0%. This strongly suggests Game of 24 is **capacity-bound** and requires significantly larger models.

### Game of 24 Files

| File | Purpose |
|------|---------|
| `policy_inference.py` | Core inference engine (loads model, generates 18 candidate moves, caches results) |
| `vllm_policy_server.py` | HTTP server wrapping policy inference (vLLM-compatible API) |
| `server.py` | Simple heuristic verifier (scores states based on proximity to 24) |
| `beamsearch_full_infrastructure.py` | Main beam search orchestrator |
| `beamsearch_full_infrastructure.sbatch` | Batch script for Snellius (SLURM) |

### Quick Start - Game of 24

#### Step 1: Download Pre-trained Model

The fine-tuned policy model is available on HuggingFace:
- **Model**: [DanielRisss/Llama3-8b-go24](https://huggingface.co/DanielRisss/Llama3-8b-go24)
- **Training Data**: [DanielRisss/policytrainer_go24](https://huggingface.co/datasets/DanielRisss/policytrainer_go24)
  - Generated from 1262 games scraped from 4nums.com
  - 100 hardest problems reserved for testing
  - All possible moves from solutions extracted

**Setup**: Download the model and ensure data paths in batch/Python files point to the correct folders.

#### Step 2: Running on Snellius (Recommended)

**Option A: Using Batch File (Recommended)**

```bash
# Edit sbatch file and submit
sbatch beamsearch_full_infrastructure.sbatch
```

Ensure you're connected via SSH to Snellius. This will run on GPU nodes with proper resource allocation.

**Option B: Interactive tmux Session (Testing Only)**

For small runs (not suitable for full benchmarks):

```bash
# Create tmux session
tmux new-session -d -s game24

# Window 1: Policy server
tmux send-keys -t game24 "python vllm_policy_server.py" Enter

# Window 2: Verifier server (instance 1)
tmux new-window -t game24
tmux send-keys -t game24 "PORT=8002 python server.py" Enter

# Window 3: Verifier server (instance 2)
tmux new-window -t game24
tmux send-keys -t game24 "PORT=8003 python server.py" Enter

# Window 4: Beam search (test run)
tmux new-window -t game24
tmux send-keys -t game24 "python beamsearch_full_infrastructure.py --num_problems 5" Enter

# View progress
tmux attach -t game24
```

### Beam Search Parameters - Game of 24

```bash
python beamsearch_full_infrastructure.py \
  --policy_server http://127.0.0.1:8000 \
  --verifier_1 http://127.0.0.1:8002/predict \
  --verifier_2 http://127.0.0.1:8003/predict \
  --data_path go24_benchmark.csv \
  --beam 5 \
  --budget 5 \
  --limit 50 \
  --num_problems 100 \
  --workers 4
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--beam` | Beam width (top-K states per depth) | 5 |
| `--budget` | Policy calls per beam candidate | 5 |
| `--limit` | Maximum search depth | 50 |
| `--num_problems` | Limit to N problems (0=all 100) | 0 |
| `--workers` | Parallel processes | 4 |
| `--temperature` | Policy generation temperature | 0.8 |

### Analyzing Game of 24 Results

```python
import pickle

# Load tree structures
trees = pickle.load(open("logs_<JOBID>/beamsearch_results_<JOBID>.pkl", "rb"))

# Analyze each problem
for tree in trees:
    print(f"Problem: {tree.question}")
    print(f"Nodes explored: {len(tree.all_nodes)}")
    print(f"Max depth: {tree.return_timestep()}")
```

---

## Part 2: GPQA - FETCH Framework Evaluation

GPQA is a dataset of 448 graduate-level multiple-choice questions testing scalable oversight: whether non-experts can verify expert reasoning through cheaper verification methods.

### Dataset Overview

**GPQA Dataset:**
- 448 graduate-level multiple-choice questions (A, B, C, D)
- Topics: Biology, Physics, Chemistry
- Expert baseline: 65% accuracy
- Non-expert baseline (unrestricted web access): 34% accuracy
- GPT-4 with internet: 39.4% accuracy
- Task: Generate reasoning steps terminating with "The answer is <LETTER>"

### FETCH Framework - 2×2 Ablation Study

This implementation evaluates four FETCH configurations:

| Configuration | Semantic Merging | Variance Reduction | Description |
|---|---|---|---|
| **Baseline** | ✗ | ✗ | No optimization (baseline) |
| **FETCH-Merge** | ✓ | ✗ | Semantic state merging only |
| **FETCH-VR** | ✗ | ✓ | Dual-verifier ensemble only |
| **FETCH-Full** | ✓ | ✓ | Both optimizations (recommended) |

### Key FETCH Techniques

**1. Semantic Merging (ESM - Reduces Over-Exploration)**
- Clusters functionally equivalent reasoning steps using embedding similarity
- Distance threshold: 0.15 (cosine distance)
- Reduces search branching factor without losing solution paths
- Uses sentence-transformers (SimCSE-Large) for embeddings

**2. Variance Reduction (Dual Verifier Ensemble - Reduces Under-Exploration)**
- Two complementary verifiers: Llama-3-8B (Verifier A) + Mistral-8B (Verifier B)
- Scores averaged before beam selection: `score_ensemble = (scoreA + scoreB) / 2`
- Model diversity provides regularization
- Especially valuable early in search when individual scores are unreliable

### GPQA Files

| File | Purpose |
|------|---------|
| `beamsearch_nomerge.py` | Baseline: no merging, single verifier |
| `beamsearch_nomerge_varreduce.py` | FETCH-VR: dual verifiers, no merging |
| `beamsearch_merge_varreduce.py` | FETCH-Full: dual verifiers + semantic merging |
| `verifier_ensemble_config.py` | Master config file |
| `server_verifier_a.py` | Verifier A (Llama-3-8B) |
| `server_verifier_b.py` | Verifier B (Mistral-8B) |
| `server_cluster.py` | ESM clustering server |
| `full_*.sh` | Batch scripts for each configuration |
| `eval_ranking.py` | Results analysis and comparison |

### Hyperparameter Ablations

Based on 25 validation problems:

| Config | Depth | Budget | Beam | Top-1 Acc | Coverage | Time (s) | Speedup |
|---|---|---|---|---|---|---|---|
| 8-3-4 | 8 | 3 | 4 | 36.0% | 92.0% | 2,313 | 3.5× |
| 10-5-6 | 10 | 5 | 6 | 32.0% | 84.0% | 4,370 | 1.9× |
| **12-3-4** | 12 | 3 | 4 | **60.0%** | 100.0% | 3,300 | 2.5× |
| 16-2-3 | 16 | 2 | 3 | 32.0% | 68.0% | 2,399 | 3.4× |
| **20-5-5** | 20 | 5 | 5 | **64.0%** | 100.0% | 8,100 | 1.0× |
| **25-5-5** | 25 | 5 | 5 | **64.0%** | 100.0% | 8,091 | 1.0× |

**Selected Configuration**: 25-5-5 (depth=25, budget=5, beam=5) for optimal accuracy.

### Evaluation Batches

Three computational budgets tested:

1. **Batch 1 (Validation)**: 20 problems, shallow params (d=12, b=3, beam=4)
2. **Batch 2 (Main Ablation)**: 100 problems, medium params (d=15, b=3, beam=4)
3. **Batch 3 (Extended Analysis)**: 25 problems, high params (d=25, b=5, beam=5)

**Results Location**: `paper_experiments/` folder

**Note**: Constrained sample sizes (20-100 problems). Statistical significance requires 500+ problems with cross-validation.

### Quick Start - GPQA

#### Step 1: Infrastructure Setup

Each configuration requires 4 GPUs on Snellius:
- GPU 0: Policy server (vLLM, Llama-3.1-8B)
- GPU 1: Verifier A (Llama-3-8B)
- GPU 2: Verifier B (Mistral-8B)
- GPU 3: ESM clustering (if using semantic merging)

#### Step 2: Running Configurations

**Option A: Batch Submission (Recommended)**

```bash
# Baseline (no optimization)
sbatch full_nomerge_baseline.sh

# FETCH-VR (dual verifiers only)
sbatch full_nomerge_varreduce.sh

# FETCH-Merge (semantic merging only)
sbatch full_merge_gpqa.sh

# FETCH-Full (both optimizations)
sbatch full_merge_varreduce.sh
```

**Option B: tmux Helper Script (Testing Only)**

```bash
# Full FETCH-Full setup
bash start_gpqa_tmux.sh merge_varreduce

# Or specific configurations:
bash start_gpqa_tmux.sh nomerge              # Baseline
bash start_gpqa_tmux.sh nomerge_varreduce    # FETCH-VR
bash start_gpqa_tmux.sh merge                # FETCH-Merge
```

#### Step 3: Results Analysis

```python
import pickle
import pandas as pd

# Load results from specific configuration
trees = pickle.load(open("logs_<JOBID>/results.pkl", "rb"))

# Calculate accuracy
correct = sum(1 for tree in trees if tree.answer == tree.final_choice)
accuracy = correct / len(trees)

print(f"Accuracy: {accuracy:.1%}")
print(f"Coverage: {sum(1 for t in trees if t.final_choice)}/{len(trees)}")
```

Compare configurations:
```bash
python eval_ranking.py \
  --results results_nomerge.pkl results_varreduce.pkl results_merge.pkl results_full.pkl \
  --output comparison.csv
```

### Expected Performance - GPQA

| Configuration | Expected Accuracy | Speedup | Notes |
|---|---|---|---|
| Baseline | ~32-36% | 1.0× | Base policy only |
| FETCH-VR | ~40-45% | 1.2× | Variance reduction helps |
| FETCH-Merge | ~50-55% | 1.8× | Merging reduces branching |
| FETCH-Full | ~55-60% | 2.0× | Combined benefits |

### GPQA Policy Model

**Vanilla Llama-3.1-8B-Instruct** (no fine-tuning)

**Input Format:**
```
Question: <question_text>
A. <choice_1>
B. <choice_2>
C. <choice_3>
D. <choice_4>
Answer: <existing_reasoning>
```

**Output Format:**
- Non-terminal step: "A reasoning fragment"
- Terminal step: "The answer is <LETTER> <eos>"

### GPQA Verifier Design

**Verifier A: Llama-3-8B**
- Scores reasoning quality based on question, options, and accumulated path
- Result-focused: evaluates progression toward correct answer
- Hidden state extraction: mean of last token's embeddings

**Verifier B: Mistral-8B**
- Same scoring objective, different architecture
- Model diversity provides orthogonal quality signals
- Reduces variance through architectural complementarity

**Verifier Ensemble**
```
score_ensemble = (scoreA + scoreB) / 2
```

Averaging reduces variance, especially early in search when individual scores are unreliable.

---

## Setup (Both Projects)

### Dependencies

```bash
pip install -r requirements_comprehensive.txt
```

### Environment Variables

```bash
# Game of 24 policy model
export POLICY_MODEL_PATH="DanielRisss/Llama3-8b-go24"

# Optional: override server endpoints
export POLICY_SERVER="http://127.0.0.1:8000"
export VERIFIER_1="http://127.0.0.1:8002/predict"
export VERIFIER_2="http://127.0.0.1:8003/predict"
```

---

## Code Organization

**Game of 24 Pipeline**:
```
train_policy.py → fine-tuned model → policy_inference.py 
  ↓
vllm_policy_server.py (:8000)
  ↓
beamsearch_full_infrastructure.py ← server.py (:8002, :8003)
```

**GPQA Pipeline**:
```
Vanilla Llama-3.1-8B → vLLM policy server (:8000)
  ↓
beamsearch_[config].py ← Verifier A (:8002) + Verifier B (:8003) + ESM (:8004)
```

---

## Key Implementation Notes

1. **Scalable Oversight (GPQA)**: Tests whether cheaper multi-model verification can validate expert reasoning without direct expert review.

2. **Dual Verifiers**: Different models (Llama + Mistral) capture complementary reasoning signals, reducing early-search unreliability.

3. **Semantic Merging**: Consolidates redundant reasoning steps (e.g., different ways to express the same idea) without losing solutions.

4. **Capacity-Bound Tasks**: Game of 24 results suggest some reasoning tasks are fundamentally limited by model size rather than search strategy.

---

## Citation

```bibtex
@article{ris2025fetch,
  title={Cross-Domain Evaluation of the FETCH Framework for Tree-of-Thought LLM Reasoning},
  author={Ris, Daniël and Ilaş, Armand and Zeinstra, Tim},
  year={2025},
  institution={Eindhoven University of Technology}
}

@article{yao2023tot,
  title={Tree of Thought: Deliberate Problem Solving with Large Language Models},
  author={Yao, Shunyu and Yu, Dian and Zhao, Jeffrey and Shafran, Izhak and Griffiths, Thomas L and Cao, Yuan and Parisi, Gregorio},
  journal={arXiv preprint arXiv:2305.10601},
  year={2023}
}

@article{wang2025fetch,
  title={Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls},
  author={Wang, Ante and Song, Linfeng and Tian, Ye and Yu, Dian and Mi, Haitao and Duan, Xiangyu and Tu, Zhaopeng and Su, Jinsong and Yu, Dong},
  journal={arXiv preprint arXiv:2502.05095},
  year={2025}
}
```

---

## Acknowledgments

This research was conducted at Eindhoven University of Technology using the Snellius HPC cluster. We thank the TU/e AI Research Group for computational resources and support.

# Game of 24 - Policy Fine-Tuning and Beam Search

This repository contains code for running Game of 24 beam search inference using our fine-tuned Llama 3.1-8B policy model. 

**Important**: The policy model achieved 0% accuracy on the benchmark (0/100 problems solved). It fails at the final 2→1 consolidation step, indicating Game of 24 is capacity-bound and requires larger models.

## What is Game of 24?

Given four numbers, combine them using +, −, ×, ÷ to reach exactly 24. Each number must be used exactly once.

**Example**: Input `1 1 4 6` → Solution `(6 - 1) × (4 + 1) = 24`

## Files

| File | Purpose |
|------|---------|
| `policy_inference.py` | Core inference engine (loads model, generates moves, caches results) |
| `vllm_policy_server.py` | HTTP server wrapping policy inference (vLLM-compatible API) |
| `server.py` | Simple heuristic verifier (scores states based on proximity to 24) |
| `beamsearch_full_infrastructure.py` | Main beam search orchestrator |
| `run.sh` | Batch script for Snellius (SLURM) |

## Quick Start

**Information**: The dataset that we used to fine-tune the model can also be found on Huggingface, and was generated using 1262 games, scraped from 4nums.com (excluding the 100 hardest ones, which we use for inference testing). The 1262 games were disected into all possible moves from the solution, which is the dataset that we present: https://huggingface.co/datasets/DanielRisss/policytrainer_go24

### 1. Download Pre-trained Model

The fine-tuned policy model is available on HuggingFace:
https://huggingface.co/DanielRisss/Llama3-8b-go24

Make sure the model is put in the correct go24 folder, and the data paths in the batch and python files point to the correct folder.

### 2. Running on Snellius (HPC Cluster)

**Option A: Using Batch File (Recommended) on Snellius**

Edit `beamsearch_full_infrastructure.sbatch` with your parameters and submit:

```bash
sbatch beamsearch_full_infrastructure.sbatch
```

Make sure you are connected on Snellius's SSH.

This will run on GPU nodes with proper resource allocation. Check `logs_<JOBID>/` for results.

**Option B: Interactive tmux Session (Small Runs Only)**

For testing with a small number of problems:

```bash
# Create tmux session
tmux new-session -d -s game24

# Terminal 1: Policy server
tmux send-keys -t game24 "export POLICY_MODEL_PATH='<your-username>/<model-name>'" Enter
tmux send-keys -t game24 "python vllm_policy_server.py" Enter

# Terminal 2: Verifier server (first instance)
tmux new-window -t game24
tmux send-keys -t game24 "PORT=8002 python server.py" Enter

# Terminal 3: Verifier server (second instance)
tmux new-window -t game24
tmux send-keys -t game24 "PORT=8003 python server.py" Enter

# Terminal 4: Beam search (run on subset of problems)
tmux new-window -t game24
tmux send-keys -t game24 "python beamsearch_full_infrastructure.py --num_problems 5" Enter

# Monitor
tmux attach -t game24
```

**Warning**: tmux mode is slow and not suitable for full benchmark runs. Use Snellius batch submission for replicating our study.

## Setup

### Dependencies

Run the command below in your local python install, or on a venv:

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
# Model path (HuggingFace model ID)
export POLICY_MODEL_PATH="<your-username>/<model-name>"

# Optional: override server endpoints
export POLICY_SERVER="http://127.0.0.1:8000"
export VERIFIER_1="http://127.0.0.1:8002/predict"
export VERIFIER_2="http://127.0.0.1:8003/predict"
```

## Beam Search Parameters

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


4. **Analyze tree structures** (pickled Python objects):
  First, you should load the classes used in the main beamsearch_full_infrastructure.py, then:
   ```python
   import pickle
   trees = pickle.load(open("logs_<JOBID>/beamsearch_results_<JOBID>.pkl", "rb"))
   for tree in trees:
       print(f"Problem: {tree.question}")
       print(f"Nodes explored: {len(tree.all_nodes)}")
       print(f"Max depth: {tree.return_timestep()}")
   ```

# GPQA - FETCH Framework Evaluation

Code for evaluating the FETCH framework on GPQA (Graduate-level Python Question Answering), a dataset of 448 graduate-level multiple-choice questions in biology, physics, and chemistry.

## Dataset Overview

**GPQA Dataset:**
- 448 graduate-level multiple-choice questions (A, B, C, D)
- 2 PhD-level experts per question, non-experts scored 34%, GPT-4 achieved 39.4%
- Expert human subjects: 65% accuracy
- Task: Select correct option based on reasoning steps

**Scalable Oversight:** GPQA tests whether non-experts can verify expert reasoning through cheaper verification methods.

## FETCH Framework

This implementation evaluates four FETCH configurations (2×2 ablation):

| Configuration | Semantic Merging | Variance Reduction | Description |
|---|---|---|---|
| **Baseline** | ✗ | ✗ | No optimization |
| **FETCH-Merge** | ✓ | ✗ | Semantic state merging only |
| **FETCH-VR** | ✗ | ✓ | Dual-verifier ensemble only |
| **FETCH-Full** | ✓ | ✓ | Both optimizations |

### Key FETCH Techniques

**1. Semantic Merging (ESM)**
- Clusters functionally equivalent reasoning steps using embedding similarity
- Distance threshold: 0.15 (cosine distance)
- Reduces search branching factor without losing solution paths

**2. Variance Reduction (Dual Verifier Ensemble)**
- Two different verifiers: Llama-3-8B (Verifier A) + Mistral-8B (Verifier B)
- Scores averaged before beam selection: `score_ensemble = (scoreA + scoreB) / 2`
- Model diversity provides regularization, reduces unreliable early scores

## Files

| File | Purpose |
|------|---------|
| `beamsearch_nomerge.py` | Baseline: no semantic merging, single verifier |
| `beamsearch_nomerge_varreduce.py` | FETCH-VR: dual verifiers, no merging |
| `beamsearch_merge_varreduce.py` | FETCH-Full: dual verifiers + semantic merging |
| `verifier_ensemble_config.py` | Master config: supports both single + dual verifiers |
| `server_verifier_a.py` | Verifier A (Llama-3-8B) - result-focused |
| `server_verifier_b.py` | Verifier B (Mistral-8B) - diverse architecture |
| `server_cluster.py` | ESM clustering server (semantic merging) |
| `full_*.sh` | Snellius batch scripts for each configuration |
| `eval_ranking.py` | Results analysis and ranking |

## Batch Sizes

Three computational budgets were tested to assess scalability:

1. **Batch 1 (Validation):** 20 problems, shallow params (d=12, b=3, beam=4)
2. **Batch 2 (Main Ablation):** 100 problems, medium params (d=15, b=3, beam=4)
3. **Batch 3 (Extended Analysis):** 25 problems, high params (d=25, b=5, beam=5)

All of these experiments' outputs can be found in the folder **paper_experiments/**

## Policy Model

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

## Verifier Design

### Verifier A: Llama-3-8B
- Scores reasoning quality based on question, options, and accumulated path
- Result-focused: evaluates how well reasoning progresses toward correct answer
- Hidden state extraction: mean of last token's embeddings

### Verifier B: Mistral-8B
- Scores with same objective but different architecture
- Model diversity introduces regularization
- Diverse training structure captures orthogonal quality signals

### Verifier Ensemble
Scores combined via arithmetic mean:
```
score_ensemble = (scoreA + scoreB) / 2
```

This reduces variance in value estimates, especially early in search when individual scores are unreliable.

## Running on Snellius

### Setup

Each configuration requires 4 GPUs on Snellius:
- GPU 0: Policy server (vLLM)
- GPU 1: Verifier A server
- GPU 2: Verifier B server  
- GPU 3: ESM clustering server (semantic merging only)

### Quick Run

Submit a batch job:

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

### Manual Run

```bash
# Terminal 1: Policy server (GPU 0)
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000

# Terminal 2: Verifier A (GPU 1)
export CUDA_VISIBLE_DEVICES=1
uvicorn server_verifier_a:app --host 0.0.0.0 --port 8002

# Terminal 3: Verifier B (GPU 2)
export CUDA_VISIBLE_DEVICES=2
uvicorn server_verifier_b:app --host 0.0.0.0 --port 8003

# Terminal 4: ESM server (GPU 3, if using semantic merging)
export CUDA_VISIBLE_DEVICES=3
uvicorn server_cluster:app --host 0.0.0.0 --port 8004

# Terminal 5: Run beam search
python beamsearch_merge_varreduce.py
```

## Results Analysis

After runs complete, analyze results, compare configurations with eval_ranking.py:
```bash
python eval_ranking.py \
  --results results_nomerge.pkl results_varreduce.pkl results_merge.pkl results_full.pkl \
  --output comparison.csv
```

## Expected Performance

Based on hyperparameter ablations:

| Configuration | Expected Accuracy | Speedup | Notes |
|---|---|---|---|
| Baseline (nomerge) | ~32-36% | 1.0× | Base policy only |
| FETCH-VR | ~40-45% | 1.2× | Variance reduction helps |
| FETCH-Merge | ~50-55% | 1.8× | Merging reduces branch. |
| FETCH-Full | ~55-60% | 2.0× | Combined benefits |

*Note: These are exploratory estimates; actual results depend on dataset batch and hyperparameters.*

## Key Configuration Parameters

- `LIMIT`: Maximum search depth (25 for full evaluation)
- `BUDGET`: Policy calls per frontier state (5 for full eval)
- `BEAM`: Beam width (5 for full eval)
- `TEMPERATURE`: Policy sampling temp (0.8 standard)
- `DISTANCE`: ESM clustering threshold (0.15 cosine distance)

## Important Notes

1. **Scalable Oversight:** GPQA tests whether cheaper verification (Mistral + Llama) can validate expert reasoning without direct expert review.

2. **Dual Verifiers:** Two models capture complementary reasoning quality signals:
   - Llama: Often more capable at technical reasoning
   - Mistral: Different training, orthogonal insights

3. **Semantic Merging:** Consolidates redundant reasoning steps (e.g., different ways of saying "analyze the data") without losing solutions.

4. **Constrained Sample Sizes:** 20-100 problems per batch. Statistical significance would require 500+ problems and cross-validation.


## Citation

```bibtex
@article{game24_fetch,
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


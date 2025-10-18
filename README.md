# Fetch — Efficient Tree Search for LLM Reasoning

Code for the paper [Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls](https://arxiv.org/abs/2502.11183)

---

## 🚀 Setup

Follow the steps below to run our scripts:

### 📌 Step 1. Setup service of policy, verifier, and embedding model

#### 📚 Policy

We employ [vllm](https://docs.vllm.ai/en/latest/) for the policy. To start the policy service, run the following command:
```
python3 -m vllm.entrypoints.openai.api_server --model /path/to/policy/model --port 8000 --dtype float16 --tensor-parallel-size 2 --swap-space 8 --max-model-len 4096
```

#### 🔍 Verifier

1. Update your model path in `verifier/server.py`.
2. Run the script: `bash run.sh ./ 0` inside the `verifier` directory.

#### 📦 Embedding Model

If you're using state merging, follow these steps:
1. Update the path in `cluster/server_cluster.py`.
2. Run the script: `bash run_app.sh ./ 0` inside the `cluster` directory.

---

### 📌 Step 2. Run tree search algorithms

We provide three tree search algorithms: **BFS (Best-First Search)**, **Beam Search**, and **MCTS (Monte Carlo Tree Search)**.

1. Specify the input, output file paths, and other parameters in scripts such as `beamsearch.py`.

2. Simply execute the corresponding Python script. For instance, to run Beam Search: `python3 beamsearch.py`

---

## 🎯 Tips

- **Using Your Own Models**: If you prefer to use your own policy or verifier, please pay attention to the prompts fed to the models and the separator (e.g., "\n") to split steps.
- **Embedding Model Preparation**: Preparing an embedding model can improve performance. We provide scripts to train these models. You can also use pretrained SimCSE or other embedding models available on [Hugging Face Hub](https://huggingface.co/models).
- **Pre-trained Models**: If you don't want to train models, our trained models are available on HuggingFace Hub:
  - Policy: [`xmu-nlp/Llama-3-8b-gsm8k`](https://huggingface.co/xmu-nlp/Llama-3-8b-gsm8k)
  - Verifier: [`xmu-nlp/Llama-3-8b-gsm8k-value-A`](https://huggingface.co/xmu-nlp/Llama-3-8b-gsm8k-value-A) and [`xmu-nlp/Llama-3-8b-gsm8k-value-B`](https://huggingface.co/xmu-nlp/Llama-3-8b-gsm8k-value-B)
  - Emb: [`xmu-nlp/simcse-large-gsm8k`](https://huggingface.co/xmu-nlp/simcse-large-gsm8k)

---

## 📝 Citation

If you find our work useful, please cite our paper:

```
@misc{wang2025dontlosttreesstreamlining,
      title={Don't GetLost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls}, 
      author={Ante Wang and Linfeng Song and Ye Tian and Dian Yu and Haitao Mi and Xiangyu Duan and Zhaopeng Tu and Jinsong Su and Dong Yu},
      year={2025},
      eprint={2502.11183},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11183}, 
}
```

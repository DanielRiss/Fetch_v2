#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --job-name=Train_GPQA_verifier
#SBATCH --time=08:00:00
# Redirect SLURM’s own logs into the job-specific folder
#SBATCH --output=~/projects/Fetch/search/beamsearch/slurm_output_%j.txt
#SBATCH --error=~/projects/Fetch/search/beamsearch/slurm_error_%j.txt

MODEL=meta-llama/Llama-3.1-8B-Instruct
TRAIN_DATA="~/projects/Fetch/gpqa/dataset/gpqa_training_split.csv"
VALID_DATA="~/projects/Fetch/gpqa/dataset/gpqa_eval_split.csv"
OUTPUT="~/projects/Fetch/gpqa/models/verifier"

accelerate launch --main_process_port 29500 --config_file=accelerate_config.yaml train_gpqa.py $MODEL $TRAIN_DATA $VALID_DATA $OUTPUT

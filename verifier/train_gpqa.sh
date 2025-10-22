#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --job-name=Train_GPQA_verifier
#SBATCH --time=03:00:00
# Redirect SLURM’s own logs into the job-specific folder
#SBATCH --output=~/projects/Fetch/search/beamsearch/slurm_output_%j.txt
#SBATCH --error=~/projects/Fetch/search/beamsearch/slurm_error_%j.txt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=meta-llama/Llama-3.1-8B-Instruct
TRAIN_DATA="$SCRIPT_DIR/../gpqa/dataset/gpqa_training_split.csv"
VALID_DATA="$SCRIPT_DIR/../gpqa/dataset/gpqa_eval_split.csv"
OUTPUT="$SCRIPT_DIR/../gpqa/models/verifier"

accelerate launch --main_process_port 29500 --config_file=accelerate_config.yaml train_gpqa.py $MODEL $TRAIN_DATA $VALID_DATA $OUTPUT

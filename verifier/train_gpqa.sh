# Todo: set paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=meta-llama/Llama-3.1-8B-Instruct
TRAIN_DATA="$SCRIPT_DIR/../gpqa/dataset/gpqa_training_split.csv"
VALID_DATA="$SCRIPT_DIR/../gpqa/dataset/gpqa_eval_split.csv"
OUTPUT="$SCRIPT_DIR/../gpqa/models/verifier

accelerate launch --main_process_port 29500 --config_file=accelerate_config.yaml train_gpqa.py $MODEL $TRAIN_DATA $VALID_DATA $OUTPUT

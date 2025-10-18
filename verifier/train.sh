MODEL=/path/to/base/model
TRAIN_DATA=/path/to/train/data
VALID_DATA=/path/to/eval/data
OUTPUT=/path/to/output/model

accelerate launch --main_process_port 29500 --config_file=accelerate_config.yaml train.py $MODEL $TRAIN_DATA $VALID_DATA $OUTPUT

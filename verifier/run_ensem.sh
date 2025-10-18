#!/bin/bash
SERVER_DIR=$1 # ./
GPU_ID=$2 # 0,1

echo $SERVER_DIR
echo $GPU_ID

CUDA_VISIBLE_DEVICES=$GPU_ID uvicorn --app-dir "${SERVER_DIR}" server_ensem:app --host 0.0.0.0 --port 8002 >& value_ensem.log &
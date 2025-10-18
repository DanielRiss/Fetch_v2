#!/bin/bash
SERVER_DIR=$1
GPU_ID=$2

echo $SERVER_DIR
echo $GPU_ID

CUDA_VISIBLE_DEVICES=$GPU_ID nohup uvicorn --app-dir "${SERVER_DIR}" server:app --host 0.0.0.0 --port 8002 > value.log 2>&1 &


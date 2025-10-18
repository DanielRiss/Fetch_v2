#!/bin/bash
SERVER_DIR=$1
GPU_ID=$2

echo $SERVER_DIR
echo $GPU_ID

CUDA_VISIBLE_DEVICES=$GPU_ID uvicorn --app-dir "${SERVER_DIR}" server_cluster:app --host 0.0.0.0 --port 8003 >& merge.log &
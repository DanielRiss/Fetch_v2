#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Beamsearch_FETCH
#SBATCH --time=03:00:00
#SBATCH --output=~/projects/Fetch/search/beamsearch/logs_%j/slurm_output_%j.txt
#SBATCH --error=~/projects/Fetch/search/beamsearch/logs_%j/slurm_error_%j.txt

set -e
echo "[$(date +'%T')] Loading modules and activating virtualenv..."
module load 2025
module load CUDA/12.8.0
source ~/projects/Fetch/.venv/bin/activate
echo "[$(date +'%T')] Environment ready."

# Capture job ID and create log directory
JOBID=${SLURM_JOB_ID}
LOGDIR=~/projects/Fetch/search/beamsearch/logs_${JOBID}
mkdir -p ${LOGDIR}
echo "Logging into ${LOGDIR}"

# 1) Policy server on GPU 0
echo "[$(date +'%T')] Starting policy server on GPU 0 (port 8000)..."
CUDA_VISIBLE_DEVICES=0 srun --ntasks=1 --gpus-per-task=1 --cpus-per-task=9 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model xmu-nlp/Llama-3-8b-gsm8k \
    --port 8000 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --swap-space 8 \
    --max-model-len 4096 \
    --disable-custom-all-reduce \
  > ${LOGDIR}/policy_${JOBID}.log 2>&1 &
POLICY_PID=$!
echo "[$(date +'%T')] Policy server PID: $POLICY_PID"

echo "[$(date +'%T')] Waiting for policy server to become ready..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:8000/v1/models > /dev/null; then
    echo "[$(date +'%T')] Policy server is ready."
    break
  fi
  echo "[$(date +'%T')] Retry $i/30..."
  sleep 5
done

# 2) Verifier A on GPU 1, port 8002
echo "[$(date +'%T')] Starting verifier A on GPU 1 (port 8002)..."
export MODEL_PATH="xmu-nlp/Llama-3-8b-gsm8k-value-A"
CUDA_VISIBLE_DEVICES=1 nohup uvicorn --app-dir ~/projects/Fetch/verifier server_varreduce:app \
    --host 0.0.0.0 --port 8002 --workers 1 \
    > ${LOGDIR}/verifier_A_${JOBID}.log 2>&1 &
PID_A=$!
echo "[$(date +'%T')] Verifier A PID: $PID_A"

# 3) Verifier B on GPU 2, port 8003
echo "[$(date +'%T')] Starting verifier B on GPU 2 (port 8003)..."
export MODEL_PATH="xmu-nlp/Llama-3-8b-gsm8k-value-B"
CUDA_VISIBLE_DEVICES=2 nohup uvicorn --app-dir ~/projects/Fetch/verifier server_varreduce:app \
    --host 0.0.0.0 --port 8003 --workers 1 \
    > ${LOGDIR}/verifier_B_${JOBID}.log 2>&1 &
PID_B=$!
echo "[$(date +'%T')] Verifier B PID: $PID_B"

echo "[$(date +'%T')] Waiting for both verifiers to become ready..."
for PORT in 8002 8003; do
  for i in {1..30}; do
    if curl -s http://127.0.0.1:${PORT}/predict > /dev/null; then
      echo "[$(date +'%T')] Verifier on port $PORT is ready."
      break
    fi
    echo "[$(date +'%T')] Retry $i/30 for verifier $PORT..."
    sleep 5
  done
done

# 4) ESM server on GPU 3
echo "[$(date +'%T')] Starting ESM server on GPU 3 (port 8004)..."
CUDA_VISIBLE_DEVICES=3 nohup uvicorn --app-dir ~/projects/Fetch/cluster server_cluster:app \
    --host 0.0.0.0 --port 8004 --workers 2 \
    > ${LOGDIR}/esm_${JOBID}.out 2> ${LOGDIR}/esm_${JOBID}.err &
ESM_PID=$!
echo "[$(date +'%T')] ESM server PID: $ESM_PID"

echo "[$(date +'%T')] Pausing 5s for ESM warm-up..."
sleep 5

# 5) Beamsearch (client-side ensembling)
echo "[$(date +'%T')] Running beamsearch with client-side ensembling..."
nvidia-smi > ${LOGDIR}/nvidia_smi_${JOBID}.log
python3 -c "import torch;print(torch.cuda.device_count())" > ${LOGDIR}/num_gpus_${JOBID}.log
export USE_ENSEMBLE_HOST=False
python3 ~/projects/Fetch/search/beamsearch/beamsearch_merge.py \
  > ${LOGDIR}/beamsearch_FETCH_${JOBID}.log 2>&1
echo "[$(date +'%T')] Beamsearch complete."

# Cleanup
echo "[$(date +'%T')] Cleaning up processes..."
kill $POLICY_PID $PID_A $PID_B $ESM_PID || true
echo "[$(date +'%T')] All processes terminated. Job finished."

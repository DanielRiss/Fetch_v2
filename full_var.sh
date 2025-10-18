#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Beamsearch_varreduce
#SBATCH --time=03:00:00
# Redirect SLURM’s own logs into the job-specific folder
#SBATCH --output=~/projects/Fetch/search/beamsearch/slurm_output_%j.txt
#SBATCH --error=~/projects/Fetch/search/beamsearch/slurm_error_%j.txt

set -e

module purge
module load 2023
source ~/projects/Fetch/.venv/bin/activate

# Capture job ID and create log directory
JOBID=${SLURM_JOB_ID}
LOGDIR=~/projects/Fetch/search/beamsearch/logs_${JOBID}
mkdir -p ${LOGDIR}
echo "Logging into ${LOGDIR}"

# 1) Policy server on GPUs 0 and 1 with tensor parallel size 2
echo "Starting policy server on GPUs 0 and 1..."
CUDA_VISIBLE_DEVICES=0,1 srun --ntasks=1 --gpus=2 --cpus-per-task=16 \
    python3 -m vllm.entrypoints.openai.api_server \
    --model xmu-nlp/Llama-3-8b-gsm8k \
    --port 8000 \
    --dtype float16 \
    --tensor-parallel-size 2 \
    --swap-space 8 \
    --max-model-len 4096 \
    --disable-custom-all-reduce \
    > ${LOGDIR}/policy_${JOBID}.log 2>&1 &
POLICY_PID=$!

# Wait for policy
for i in {1..30}; do
  curl -s http://127.0.0.1:8000/v1/models && break
  echo "Waiting policy ($i/30)..."; sleep 10
  [[ $i -eq 30 ]] && { echo "Policy failed"; kill $POLICY_PID; exit 1; }
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

# 3) Run beamsearch
echo "All servers started: running beamsearch with var reduce"
cd ~/projects/Fetch/search/beamsearch
nvidia-smi > ${LOGDIR}/nvidia_smi_${JOBID}.log
python3 -c "import torch;print(torch.cuda.device_count())" > ${LOGDIR}/num_gpus_${JOBID}.log
python3 beamsearch.py > ${LOGDIR}/beamsearch_varreduce_${JOBID}.log 2>&1

echo "Beamsearch complete, cleaning up..."

# Cleanup
kill $POLICY_PID || true
kill $VERIFIER_PID || true

#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=Beamsearch_nomerge_baseline
#SBATCH --time=08:00:00
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

set -e

module purge
module load 2025
module load CUDA/12.8.0
source ~/projects/Fetch/.venv/bin/activate

cd /gpqa

# Capture job ID and create log directory
JOBID=${SLURM_JOB_ID}
LOGDIR=gpqa/jobs_${JOBID}
mkdir -p ${LOGDIR}

echo "Logging into ${LOGDIR}"

# 1) Policy server on GPU 0
echo "Starting policy server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 srun --ntasks=1 --cpus-per-task=9 --gpus-per-task=1 \
python3 -m vllm.entrypoints.openai.api_server \
--model meta-llama/Llama-3.1-8B-Instruct \
--port 8000 \
--dtype float16 \
--tensor-parallel-size 1 \
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

# 2) Start TWO verifier servers on GPUs 1 and 2, ports 8002 and 8003
# Both use the SAME server.py (round-robin baseline, NO merging)
VERIFIER_GPUS=(1 2)
VERIFIER_PORTS=(8002 8003)

for i in ${!VERIFIER_GPUS[@]}; do
  GPU=${VERIFIER_GPUS[$i]}
  PORT=${VERIFIER_PORTS[$i]}
  
  echo "Starting verifier $i on GPU $GPU port $PORT..."
  export CUDA_VISIBLE_DEVICES=$GPU
  
  nohup uvicorn server:app \
  --host 0.0.0.0 --port $PORT --workers 1 \
  > "${LOGDIR}/verifier_${PORT}_${JOBID}.log" 2>&1 &
  
  VERIFIER_PIDS[$i]=$!
done

# Wait for verifiers to start
for PORT in "${VERIFIER_PORTS[@]}"; do
  for i in {1..30}; do
    if curl -s http://127.0.0.1:${PORT}/predict > /dev/null 2>&1; then
      echo "Verifier on port $PORT is ready!"
      break
    fi
    echo "Waiting for verifier on port $PORT, attempt $i/30..."
    sleep 10
    if [ $i -eq 30 ]; then
      echo "ERROR: Verifier server on port $PORT failed to start"
      kill $POLICY_PID || true
      for pid in "${VERIFIER_PIDS[@]}"; do kill $pid || true; done
      exit 1
    fi
  done
done

# 3) NO ESM server needed - baseline has no merging

echo "All servers started: running beamsearch WITHOUT merging (baseline)"

python3 beamsearch_nomerge.py > ${LOGDIR}/beamsearch_nomerge_${JOBID}.log 2>&1

echo "Beamsearch complete, cleaning up..."

# Cleanup
kill $POLICY_PID || true
kill "${VERIFIER_PIDS[@]}" || true

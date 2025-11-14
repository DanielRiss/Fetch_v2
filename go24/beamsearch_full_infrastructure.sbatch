#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --job-name=GO24_Beamsearch_Full
#SBATCH --time=03:00:00
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt

set -e

# Setup
module purge
module load 2025
module load CUDA/12.8.0

#change the source of the venv to your venv location.
source /.venv/bin/activate

export FLASHINFER_USE_TVM_CPP=0

# Logging
JOBID=${SLURM_JOB_ID}
LOGDIR=/logs_${JOBID}
mkdir -p ${LOGDIR}
export LOGDIR

echo "Logging to: ${LOGDIR}"
echo "Job ID: ${JOBID}"

# ============================================================================
# 1) POLICY SERVER using PolicyInference
# ============================================================================
echo "Starting policy server on GPUs 0-1..."

CUDA_VISIBLE_DEVICES=0,1 srun --ntasks=1 --gpus=2 --cpus-per-task=16 \
    python3 vllm_policy_server.py \
    --port 8000 \
    --host 0.0.0.0 \
    > ${LOGDIR}/policy_${JOBID}.log 2>&1 &
POLICY_PID=$!

# Wait for policy server
echo "Waiting for policy server..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "✓ Policy server ready!"
        break
    fi
    echo "  Attempt $i/30..."
    sleep 5
    if [ $i -eq 30 ]; then
        echo "ERROR: Policy server failed to start"
        kill $POLICY_PID || true
        exit 1
    fi
done

# ============================================================================
# 2) VERIFIER SERVERS
# ============================================================================
VERIFIER_GPUS=(2 3)
VERIFIER_PORTS=(8002 8003)

for i in ${!VERIFIER_GPUS[@]}; do
  GPU=${VERIFIER_GPUS[$i]}
  PORT=${VERIFIER_PORTS[$i]}
  echo "Starting verifier on GPU $GPU port $PORT..."

  export CUDA_VISIBLE_DEVICES=$GPU
  nohup uvicorn server:app \
    --host 0.0.0.0 --port $PORT --workers 1 \
    > "${LOGDIR}/verifier_${PORT}_${JOBID}.log" 2>&1 &
  VERIFIER_PIDS[$i]=$!
done

# Wait for verifiers
for PORT in "${VERIFIER_PORTS[@]}"; do
    for i in {1..30}; do
        if curl -s http://127.0.0.1:${PORT}/predict > /dev/null 2>&1; then
            echo "✓ Verifier on port $PORT ready!"
            break
        fi
        echo "  Verifier $PORT attempt $i/30..."
        sleep 5
        if [ $i -eq 30 ]; then
            echo "ERROR: Verifier on port $PORT failed to start"
            kill $POLICY_PID || true
            for pid in "${VERIFIER_PIDS[@]}"; do kill $pid || true; done
            exit 1
        fi
    done
done

# ============================================================================
# 3) RUN BEAM SEARCH
# ============================================================================
echo "All servers started, running beam search..."

export POLICY_SERVER="http://127.0.0.1:8000"
export VERIFIER_1="http://127.0.0.1:8002/predict"
export VERIFIER_2="http://127.0.0.1:8003/predict"
export DATA_PATH="go24-benchmark.csv"

python beamsearch_full_infrastructure.py \
    --num_problems 1 \
    --limit 25 \
    --beam 5 \
    --budget 5 \
    --workers 16 \
    --policy_server "http://127.0.0.1:8000" \
    --verifier_1 "http://127.0.0.1:8002/predict" \
    --verifier_2 "http://127.0.0.1:8003/predict" \
    > ${LOGDIR}/beamsearch_${JOBID}.log 2>&1

# ============================================================================
# 4) CLEANUP
# ============================================================================
echo "Beam search complete, cleaning up..."
kill $POLICY_PID || true
for pid in "${VERIFIER_PIDS[@]}"; do kill $pid || true; done

echo "Done! Results in: ${LOGDIR}"

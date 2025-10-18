#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --job-name=Beamsearch_classic
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

# Wait for policy server
echo "Waiting for policy server to start..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
        echo "Policy server is ready!"
        break
    fi
    echo "Attempt $i/30: Policy server not ready, waiting 10s..."
    sleep 10
    if [ $i -eq 30 ]; then
        echo "ERROR: Policy server failed to start"
        kill $POLICY_PID || true
        exit 1
    fi
done

# 2) Start two verifier server instances on GPUs 2 and 3, ports 8002 and 8003
VERIFIER_GPUS=(2 3)
VERIFIER_PORTS=(8002 8003)

for i in ${!VERIFIER_GPUS[@]}; do
  GPU=${VERIFIER_GPUS[$i]}
  PORT=${VERIFIER_PORTS[$i]}
  echo "Starting verifier on GPU $GPU port $PORT…"
  export CUDA_VISIBLE_DEVICES=$GPU
  nohup uvicorn --app-dir ~/projects/Fetch/verifier server:app \
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

# 3) Run beamsearch
echo "All servers started: running beamsearch"
cd ~/projects/Fetch/search/beamsearch
nvidia-smi > ${LOGDIR}/nvidia_smi_${JOBID}.log
python3 -c "import torch;print(torch.cuda.device_count())" > ${LOGDIR}/num_gpus_${JOBID}.log

python3 beamsearch.py > ${LOGDIR}/beamsearch_${JOBID}.log 2>&1

echo "Beamsearch complete, cleaning up..."

# Cleanup
kill $POLICY_PID || true
for pid in "${VERIFIER_PIDS[@]}"; do kill $pid || true; done

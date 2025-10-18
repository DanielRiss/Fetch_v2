#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=Beamsearch_merge
#SBATCH --time=03:00:00
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

# 1) Policy server on GPU 0
echo "Starting policy server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 srun --ntasks=1 --cpus-per-task=9 --gpus-per-task=1 \
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

# Wait for policy
for i in {1..30}; do
  curl -s http://127.0.0.1:8000/v1/models && break
  echo "Waiting policy ($i/30)..."; sleep 10
  [[ $i -eq 30 ]] && { echo "Policy failed"; kill $POLICY_PID; exit 1; }
done

# 2) Start two verifier server instances on GPUs 1 and 2, ports 8002 and 8003
VERIFIER_GPUS=(1 2)
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

# 3) ESM server on GPU 3
ESM_PORT=8004
echo "Starting ESM server on GPU 3 port $ESM_PORT with TERM trap..."
cd ~/projects/Fetch/cluster

trap '' TERM
CUDA_VISIBLE_DEVICES="3" nohup uvicorn server_cluster:app \
    --host 0.0.0.0 --port ${ESM_PORT} --workers 2 --log-level debug \
    > ${LOGDIR}/esm_${JOBID}.out 2> ${LOGDIR}/esm_${JOBID}.err < /dev/null &
ESM_PID=$!
trap - TERM

echo "ESM PID: $ESM_PID"
sleep 5

# 4) Beamsearch merge
echo "All servers started: running beamsearch with merge"
cd ~/projects/Fetch/search/beamsearch
nvidia-smi > ${LOGDIR}/nvidia_smi_${JOBID}.log
python3 -c "import torch;print(torch.cuda.device_count())" > ${LOGDIR}/num_gpus_${JOBID}.log

python3 beamsearch_merge.py > ${LOGDIR}/beamsearch_merge_${JOBID}.log 2>&1

echo "Beamsearch complete, cleaning up..."

# Cleanup
kill $POLICY_PID || true
kill "${VERIFIER_PIDS[@]}" || true
kill $ESM_PID || true

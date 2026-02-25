#!/bin/bash
#SBATCH --job-name=metamind_socialiqa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/laredo.ei/logs/metamind_%j.out
#SBATCH --error=/scratch/laredo.ei/logs/metamind_%j.err

# ============================================================
# MetaMind + vLLM on Explorer Cluster
# Serves gpt-oss-120b locally, runs Social IQA through MetaMind
# ============================================================

SCRATCH=/scratch/laredo.ei
MODEL_PATH=$SCRATCH/models/gpt-oss-120b
METAMIND_PATH=$SCRATCH/MetaMind
RESULTS_DIR=$SCRATCH/results

# Create directories
mkdir -p $SCRATCH/logs
mkdir -p $RESULTS_DIR

# ============================================================
# CRITICAL: Unset proxy so localhost requests aren't intercepted
# ============================================================
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset no_proxy
unset NO_PROXY
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

# Load modules
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

# Activate conda env
source activate $SCRATCH/envs/inference

# Install vLLM and dependencies (first run only — cached after)
pip install vllm openai transformers accelerate --quiet

# ============================================================
# Step 1: Start vLLM server in the background
# ============================================================
echo "Starting vLLM server..."
echo "Model: $MODEL_PATH"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name openai/gpt-oss-120b \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    > $SCRATCH/logs/vllm_server_$SLURM_JOB_ID.log 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready (model loading takes several minutes)
echo "Waiting for vLLM server to load model (this may take 5-10 minutes)..."
for i in $(seq 1 600); do
    if curl -s --noproxy localhost http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready! (took ${i}s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check log:"
        tail -50 $SCRATCH/logs/vllm_server_$SLURM_JOB_ID.log
        exit 1
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Still waiting... (${i}s elapsed)"
    fi
    sleep 1
done

# Final verification
if ! curl -s --noproxy localhost http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within 600 seconds"
    echo "vLLM log tail:"
    tail -50 $SCRATCH/logs/vllm_server_$SLURM_JOB_ID.log
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Test with a simple request
echo "Testing vLLM with a simple request..."
TEST_RESPONSE=$(curl -s --noproxy localhost http://127.0.0.1:8000/v1/models)
echo "Available models: $TEST_RESPONSE"

# ============================================================
# Step 2: Run MetaMind
# ============================================================
echo "Starting MetaMind Social IQA run..."
cd $METAMIND_PATH

python run_socialiqa_cluster.py \
    --dev_path $SCRATCH/socialiqa-train-dev/dev.jsonl \
    --labels_path $SCRATCH/socialiqa-train-dev/dev-labels.lst \
    --output_dir $RESULTS_DIR \
    --max_samples 1000

# ============================================================
# Cleanup
# ============================================================
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Done! Results in $RESULTS_DIR"
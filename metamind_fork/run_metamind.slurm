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
MODEL_PATH=$SCRATCH/models/gpt-oss-120b-hf
METAMIND_PATH=$SCRATCH/MetaMind
RESULTS_DIR=$SCRATCH/results

# Create directories
mkdir -p $SCRATCH/logs
mkdir -p $RESULTS_DIR

# ============================================================
# Keep proxy active for internet access, bypass localhost only
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export HF_HOME=$SCRATCH/.cache/huggingface

# Load modules
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

# Activate conda env
source activate $SCRATCH/envs/inference

# Keep proxy active for internet access, bypass localhost only
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export HF_HOME=$SCRATCH/.cache/huggingface

# Install vLLM and dependencies (first run only — cached after)
pip install vllm openai transformers accelerate --quiet

# ============================================================
# Step 1: Start vLLM server in the background
# ============================================================
echo "Starting vLLM server..."
echo "Model: $MODEL_PATH"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

# Keep GPU busy during model loading to prevent idle cancellation
python3 -c "
import torch, time
print('GPU warmup started')
x = torch.randn(1000, 1000, device='cuda')
while True:
    x = torch.mm(x, x)
    time.sleep(0.1)
" &
WARMUP_PID=$!
echo "GPU warmup PID: $WARMUP_PID"

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
for i in $(seq 1 1200); do
    if curl -s --noproxy localhost http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready! (took ${i}s)"
        kill $WARMUP_PID 2>/dev/null
        echo "GPU warmup stopped"
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
    echo "ERROR: vLLM server failed to start within 1200 seconds"
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
sleep 30
cd $METAMIND_PATH

python run_socialiqa_cluster.py \
    --dev_path $SCRATCH/socialiqa-train-dev/train.jsonl \
    --labels_path $SCRATCH/socialiqa-train-dev/train-labels.lst \
    --output_dir $RESULTS_DIR \
    --start_sample ${START_SAMPLE:-1} \
    --max_samples ${MAX_SAMPLES:-500}

# ============================================================
# Cleanup
# ============================================================
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Done! Results in $RESULTS_DIR"
#!/bin/bash
# ─── One-time environment setup for DPO training ────────────────────────────
# Run this on a compute node (not login node):
#   srun --partition=short --nodes=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty /bin/bash
#   bash /scratch/laredo.ei/dpo_training/setup_env.sh

set -e

SCRATCH="/scratch/laredo.ei"
ENV_PATH="${SCRATCH}/envs/dpo"
HF_CACHE="${SCRATCH}/.cache/huggingface"
DPO_DIR="${SCRATCH}/dpo_training"

export HF_HOME="${HF_CACHE}"

# ── Load modules ─────────────────────────────────────────────────────────────
module load anaconda3/2024.06
module load cuda/12.3.0

# ── Create conda env ────────────────────────────────────────────────────────
if [ -d "${ENV_PATH}" ]; then
    echo "Conda env already exists at ${ENV_PATH}, skipping creation."
else
    echo "Creating conda env at ${ENV_PATH}..."
    conda create -c conda-forge python=3.10 -y --prefix "${ENV_PATH}"
fi

source activate "${ENV_PATH}"

# ── Install PyTorch (CUDA 12.1 compatible) ───────────────────────────────────
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Install training dependencies ────────────────────────────────────────────
echo "Installing training dependencies..."
pip install \
    transformers \
    accelerate \
    trl \
    peft \
    datasets \
    bitsandbytes \
    wandb \
    huggingface_hub \
    flash-attn --no-build-isolation

# ── Download Qwen 2.5 7B Instruct ───────────────────────────────────────────
echo "Downloading Qwen 2.5 7B Instruct..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir "${SCRATCH}/models/Qwen2.5-7B-Instruct"

# ── Create directory structure ───────────────────────────────────────────────
echo "Creating directory structure..."
mkdir -p "${DPO_DIR}/dpo_data"
mkdir -p "${DPO_DIR}/dpo_output"
mkdir -p "${SCRATCH}/logs"

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Setup complete. Verify:"
echo "============================================================"
echo ""
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"

echo ""
echo "Model downloaded to: ${SCRATCH}/models/Qwen2.5-7B-Instruct/"
ls "${SCRATCH}/models/Qwen2.5-7B-Instruct/" | head -5
echo ""
echo "Next steps:"
echo "  1. scp best_checkpoint to ${DPO_DIR}/best_checkpoint/"
echo "  2. scp dpo_dataset.jsonl to ${DPO_DIR}/dpo_data/"
echo "  3. Run: sbatch ${DPO_DIR}/run_dpo.slurm"
echo "============================================================"
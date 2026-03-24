#!/bin/bash
#SBATCH --job-name=dpo-qwen7b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:50:00
#SBATCH --output=/scratch/laredo.ei/logs/dpo_%j.out
#SBATCH --error=/scratch/laredo.ei/logs/dpo_%j.err
#SBATCH --signal=B:SIGTERM@120
#
# ─── DPO Training Slurm Job ─────────────────────────────────────────────────
# 
# The --signal flag sends SIGTERM 120 seconds before the time limit,
# giving the training script time to save a checkpoint before kill.
#
# Usage:
#   First run:   sbatch run_dpo.slurm
#   Resume run:  sbatch --export=ALL,RESUME=1 run_dpo.slurm
# ─────────────────────────────────────────────────────────────────────────────

SCRATCH="/scratch/laredo.ei"
DPO_DIR="${SCRATCH}/dpo_training"

# ── Load modules ─────────────────────────────────────────────────────────────
module load anaconda3/2024.06
module load cuda/12.3.0

# ── Activate env ─────────────────────────────────────────────────────────────
source activate "${SCRATCH}/envs/dpo"

# ── Environment variables ────────────────────────────────────────────────────
export HF_HOME="${SCRATCH}/.cache/huggingface"
export WANDB_PROJECT="capstone-dpo"
export TOKENIZERS_PARALLELISM=false

# ── Print job info ───────────────────────────────────────────────────────────
echo "============================================================"
echo "DPO Training Job"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPUs:         ${SLURM_GPUS_ON_NODE:-1}"
echo "Start time:   $(date)"
echo "Time limit:   07:50:00 (SIGTERM at 07:48:00)"
echo "Resume:       ${RESUME:-0}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

# ── Build launch command ─────────────────────────────────────────────────────
CMD="python ${DPO_DIR}/train_dpo.py"

if [ "${RESUME}" = "1" ]; then
    CMD="${CMD} --resume_from_checkpoint"
    echo "Resuming from latest checkpoint..."
fi

# ── Run ──────────────────────────────────────────────────────────────────────
echo "Running: ${CMD}"
echo ""

${CMD}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"
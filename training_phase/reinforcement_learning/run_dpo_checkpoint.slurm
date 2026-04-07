#!/bin/bash
#SBATCH --job-name=dpo-ckpt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:50:00
#SBATCH --output=/scratch/laredo.ei/logs/dpo_ckpt_%j.out
#SBATCH --error=/scratch/laredo.ei/logs/dpo_ckpt_%j.err
#SBATCH --signal=B:SIGTERM@120
#
# ─── DPO Checkpoint Training ────────────────────────────────────────────────
#
# Runs train_dpo_checkpoint.py with configurable beta/lr.
# Saves checkpoints every 50 steps for early stopping analysis.
#
# Usage:
#   sbatch --export=ALL,BETA=1.0,LR=5e-6 run_dpo_checkpoint.slurm
#   sbatch --export=ALL,BETA=2.0,LR=5e-6 run_dpo_checkpoint.slurm
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
echo "DPO Checkpoint Training"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Beta:         ${BETA:-1.0}"
echo "LR:           ${LR:-5e-6}"
echo "Start time:   $(date)"
echo "Time limit:   07:50:00 (SIGTERM at 07:48:00)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

# ── Build launch command ─────────────────────────────────────────────────────
CMD="python ${DPO_DIR}/train_dpo_checkpoint.py --beta ${BETA:-1.0} --lr ${LR:-5e-6}"

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
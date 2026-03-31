#!/bin/bash
#SBATCH --job-name=dpo-grid
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:50:00
#SBATCH --output=/scratch/laredo.ei/logs/dpo_grid_%j.out
#SBATCH --error=/scratch/laredo.ei/logs/dpo_grid_%j.err
#SBATCH --array=0-3
#
# ─── DPO Grid Search — Array Job ────────────────────────────────────────────
#
# Submits 4 jobs at once, each running 3 configs:
#   batch 0: beta=0.1  x all LRs
#   batch 1: beta=0.2  x all LRs
#   batch 2: beta=0.3  x all LRs
#   batch 3: beta=0.5  x all LRs
#
# Usage:
#   sbatch run_grid_search.slurm        # launches all 4
# ─────────────────────────────────────────────────────────────────────────────

SCRATCH="/scratch/laredo.ei"
DPO_DIR="${SCRATCH}/dpo_training"

module load anaconda3/2024.06
module load cuda/12.3.0
source activate "${SCRATCH}/envs/dpo"

export HF_HOME="${SCRATCH}/.cache/huggingface"
export WANDB_PROJECT="capstone-dpo-grid"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "DPO Grid Search — Batch ${SLURM_ARRAY_TASK_ID}"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

python ${DPO_DIR}/grid_search_dpo.py --batch ${SLURM_ARRAY_TASK_ID}

echo ""
echo "Batch ${SLURM_ARRAY_TASK_ID} finished at: $(date)"
echo "Exit code: $?"
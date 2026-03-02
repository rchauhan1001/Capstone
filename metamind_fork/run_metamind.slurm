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

SCRATCH=/scratch/laredo.ei
METAMIND_PATH=$SCRATCH/MetaMind
RESULTS_DIR=$SCRATCH/results

mkdir -p $SCRATCH/logs
mkdir -p $RESULTS_DIR

export HF_HOME=$SCRATCH/.cache/huggingface

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

source activate $SCRATCH/envs/inference

cd $METAMIND_PATH

echo "Starting MetaMind Social IQA run..."
python run_socialiqa_cluster.py \
    --dev_path $SCRATCH/socialiqa-train-dev/train.jsonl \
    --labels_path $SCRATCH/socialiqa-train-dev/train-labels.lst \
    --output_dir $RESULTS_DIR \
    --start_sample ${START_SAMPLE:-1} \
    --max_samples ${MAX_SAMPLES:-500}

echo "Done! Results in $RESULTS_DIR"
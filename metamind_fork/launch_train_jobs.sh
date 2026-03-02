#!/bin/bash
sbatch --export=ALL,START_SAMPLE=5172,MAX_SAMPLES=329 /scratch/laredo.ei/MetaMind/run_metamind.slurm
echo "Submitted job: samples 5172 to 5500"

sbatch --export=ALL,START_SAMPLE=5501,MAX_SAMPLES=500 /scratch/laredo.ei/MetaMind/run_metamind.slurm
echo "Submitted job: samples 5501 to 6000"

sbatch --export=ALL,START_SAMPLE=6001,MAX_SAMPLES=500 /scratch/laredo.ei/MetaMind/run_metamind.slurm
echo "Submitted job: samples 6001 to 6500"

sbatch --export=ALL,START_SAMPLE=6501,MAX_SAMPLES=500 /scratch/laredo.ei/MetaMind/run_metamind.slurm
echo "Submitted job: samples 6501 to 7000"

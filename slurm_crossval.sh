#!/bin/bash
#SBATCH --job-name=train_crossval
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --exclusive

OMP_NUM_THREADS=1 time srun --unbuffered python train_crossval.py

LATEST_RESULTS_DIR=$(ls -dt ./results/*/ | head -1)

if [ -z "$LATEST_RESULTS_DIR" ]; then
    echo "Error: No results directory found in ./results/. Exiting."
    exit 1
fi

echo "Latest training results directory: $LATEST_RESULTS_DIR"

OMP_NUM_THREADS=1 time srun --unbuffered python test_crossval.py "$LATEST_RESULTS_DIR"
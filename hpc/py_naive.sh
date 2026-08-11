#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# Override with: PYTHON=/path/to/python sbatch py_naive.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Naive (pooled, no covariate adjustment) ROC baseline -- pure empirical-CDF pooling, no
# neural network, no GPU needed.

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/simulation/naive_roc_baseline.py -i $1 -o $2"
echo $command
$command

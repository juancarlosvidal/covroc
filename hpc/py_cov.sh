#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch py_cov.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Measured cost at -b 150 -k 5 -e 800: ~2.45h/replicate at n=5000, ~9.1h/replicate at
# n=20000 -- and this script processes every file in -i sequentially in ONE GPU
# allocation, so pointing -i at a full 100-replicate scenario folder means one GPU
# grinding through it for weeks. For a real scenario sweep, use py_cov_array.sh/
# run_cov_array.sh instead (one replicate per SLURM array task, run concurrently). Only
# use this script directly against a small, already-curated -i folder (e.g. a handful of
# files you've deliberately picked or symlinked together).

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/simulation/coverage_bootstrap_crossfit.py -i $1 -o $2 -b 150 -k 5 -e 800"
echo $command
$command

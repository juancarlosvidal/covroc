#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Override with: PYTHON=/path/to/python sbatch py_naive_real_data.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Naive (pooled, no covariate adjustment) ROC/AUC baseline for the NHANES case study
# (Reviewer 1, Major Concern 3), per sex and mortality horizon. CPU-only, pandas/scipy
# only -- no GPU, instantaneous -- but still submitted via sbatch rather than run on the
# login node, same reasoning as py_post.sh.

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/real_data/naive_roc_baseline.py -i ./data -o ./output/real_data/naive_roc_summary.csv"
echo $command
$command

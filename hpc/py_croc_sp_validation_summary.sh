#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Override with: PYTHON=/path/to/python sbatch py_croc_sp_validation_summary.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Combines R/croc_sp_validation.R's and src/baselines/croc_sp_validation.py's per-scenario
# output CSVs (written by hpc/croc_sp_validation_array.sh) into one R-vs-Python diff
# summary table. Pandas over a handful of tiny CSVs -- no GPU, seconds at most -- but
# still submitted via sbatch rather than run on the login node, same reasoning as
# py_post.sh/py_bnp_summary.sh.

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/postprocessing/croc_sp_validation_summary.py --input-dir output/croc_sp_validation --output-csv output/croc_sp_validation/croc_sp_validation_summary.csv"
echo $command
$command

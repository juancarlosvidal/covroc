#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Override with: PYTHON=/path/to/python sbatch py_bnp_summary.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Combines R/aroc_bnp_timing.R's output/aroc_bnp_timing/*_timing.csv files (one per
# scenario/sample-size, written by hpc/r_bnp_timing_array.sh) into one summary table.
# Pandas over 18 tiny CSVs -- no GPU, seconds at most -- but still submitted via sbatch
# rather than run on the login node, same reasoning as py_post.sh.

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/postprocessing/aroc_bnp_timing_summary.py --input-dir output/aroc_bnp_timing --output-csv output/aroc_bnp_timing/aroc_bnp_timing_summary.csv"
echo $command
$command

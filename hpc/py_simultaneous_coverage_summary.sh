#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Override with: PYTHON=/path/to/python sbatch py_simultaneous_coverage_summary.sh [root_dir]
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Bonferroni-corrected simultaneous coverage from the already-written coverage.csv files
# (Reviewer 2, Major Comment 4). Pure pandas over existing per-replicate CSVs -- no GPU,
# no recomputation of the bootstrap-crossfit-OOB engine -- but still submitted via sbatch
# rather than run on the login node, same reasoning as py_post.sh.
# $1: combined root dir (default output/combined -- one subfolder per replicate, each
#     with a coverage.csv, same layout statistics_summary.py's --root-dir expects)
ROOT=${1:-output/combined}

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/postprocessing/simultaneous_coverage_summary.py --root-dir $ROOT --output-csv $ROOT/simultaneous_coverage_summary.csv"
echo $command
$command

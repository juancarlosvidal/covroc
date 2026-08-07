#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# Override with: PYTHON=/path/to/python sbatch py_post.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Post-processing (statistics_summary.py, mean_std_mse_boxplots.py, write_latex_table.py)
# is pandas/matplotlib aggregation over CSVs already produced by py_mlp.sh/py_rfo.sh/
# py_cov.sh -- no GPU, no heavy compute. This is a small CPU-only job, deliberately
# without --gres=gpu, so it doesn't sit in a GPU queue for no reason; still submitted via
# sbatch (rather than run on the login node) since login nodes are usually not meant for
# even light processing.
# $1: combined root dir (one method-prefixed subfolder per replicate -- see the README's
#     "combine each method into one root with a prefix" step)
ROOT=$1

# Run from the repository root (SLURM's working directory is the submission directory)
commands=(
  "$PYTHON src/postprocessing/statistics_summary.py --root-dir $ROOT --output-csv $ROOT/statistics_summary.csv --timing-csv $ROOT/timing_summary.csv --mean-std-mse-csv $ROOT/mean_std_mse_summary.csv --coverage-csv $ROOT/coverage_summary.csv"
  "$PYTHON src/postprocessing/mean_std_mse_boxplots.py --root-dir $ROOT --output-dir $ROOT/mean_std_mse_boxplots"
  "$PYTHON src/postprocessing/write_latex_table.py --stats-csv $ROOT/statistics_summary.csv --tex-output $ROOT/summary_table_mse.tex --dat-output $ROOT/summary_table_mse.dat"
)
for command in "${commands[@]}"; do
  echo $command
  $command
done

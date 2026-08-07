#!/bin/bash
# Combines the per-method replicate folders written by run_mlp.sh/run_rfo.sh/run_cov.sh/etc.
# (<output_root>/<method>/scenario_N/scenario_N_n_rep/) into a single root with
# method-prefixed names (<combined_dir>/<method>_scenario_N_n_rep/) -- the layout
# statistics_summary.py / mean_std_mse_boxplots.py / write_latex_table.py expect for
# cross-method comparison (see the README's Post-processing step). Uses symlinks rather
# than copies, so it's instant and doesn't duplicate any data, and is light enough to run
# directly (no sbatch needed) even on a login node.
#
# Usage: hpc/combine_outputs.sh <output_root> <combined_dir> <method1> [method2 ...]
# Example: ./hpc/combine_outputs.sh output output/combined fnn rf coverage
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <output_root> <combined_dir> <method1> [method2 ...]" >&2
  exit 1
fi

# Run from the repository root regardless of where this script is invoked from
cd "$(dirname "$0")/.."

OUTPUT_ROOT=$1
COMBINED_DIR=$2
shift 2
METHODS=("$@")

mkdir -p "$COMBINED_DIR"
shopt -s nullglob
for m in "${METHODS[@]}"; do
  count=0
  for d in "$OUTPUT_ROOT/$m"/*/*/; do
    name=$(basename "$d")
    ln -sfn "$(cd "$d" && pwd)" "$COMBINED_DIR/${m}_${name}"
    count=$((count + 1))
  done
  if [ "$count" -eq 0 ]; then
    echo "Warning: no replicate folders found under $OUTPUT_ROOT/$m/*/*/ -- typo, or that method hasn't finished yet?" >&2
  else
    echo "$m: linked $count replicate folders into $COMBINED_DIR"
  fi
done

#!/bin/bash
# Combines the per-method replicate folders written by run_mlp.sh/run_rfo.sh/run_cov.sh/etc.
# into a single root with method-prefixed names (<combined_dir>/<method>_scenario_N_n_rep/)
# -- the layout statistics_summary.py / mean_std_mse_boxplots.py / write_latex_table.py
# expect for cross-method comparison (see the README's Post-processing step). Uses
# symlinks rather than copies, so it's instant and doesn't duplicate any data, and is
# light enough to run directly (no sbatch needed) even on a login node.
#
# Finds replicate folders by content (wherever a timing.csv sits), not by a fixed path
# depth: mlp_reg_data_simulation_multi.py/coverage_bootstrap_crossfit.py/naive_roc_baseline.py/
# linear_reg_data_simulation.py write directly to <output_root>/<method>/scenario_N/scenario_N_n_rep/
# (one level under scenario_N), but rf_reg_data_simulation_multi_2.py adds an extra
# "<var_to_group>_rf" level (<output_root>/<method>/scenario_N/mortstat_rf/scenario_N_n_rep/)
# -- a fixed-depth glob silently missed RF's replicate folders entirely.
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
for m in "${METHODS[@]}"; do
  count=0
  while IFS= read -r -d '' timing_file; do
    d=$(dirname "$timing_file")
    name=$(basename "$d")
    ln -sfn "$(cd "$d" && pwd)" "$COMBINED_DIR/${m}_${name}"
    count=$((count + 1))
  done < <(find "$OUTPUT_ROOT/$m" -name timing.csv -print0 2>/dev/null)
  if [ "$count" -eq 0 ]; then
    echo "Warning: no timing.csv found anywhere under $OUTPUT_ROOT/$m -- typo, or that method hasn't finished yet?" >&2
  else
    echo "$m: linked $count replicate folders into $COMBINED_DIR"
  fi
done

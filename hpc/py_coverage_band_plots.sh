#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Override with: PYTHON=/path/to/python sbatch py_coverage_band_plots.sh [combined_root]
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Plots one representative replicate's bootstrap-OOB confidence intervals per scenario
# (Reviewer 2, Major Comment 4), picking the first coverage_scenario_<N>_* folder found
# under the combined output root (see hpc/combine_outputs.sh). CPU-only, matplotlib
# only -- no GPU, seconds per scenario.
# $1: combined root dir (default output/combined)
ROOT=${1:-output/combined}
OUT_DIR="$ROOT/coverage_band_plots"
mkdir -p "$OUT_DIR"

for SCENARIO in 1 2 3 4 5 6 7 8 9; do
  REPLICATE_DIR=$(ls -d "$ROOT"/coverage_scenario_${SCENARIO}_*/ 2>/dev/null | sort | head -n 1)
  if [ -z "$REPLICATE_DIR" ]; then
    echo "=== scenario_$SCENARIO: no coverage replicate found under $ROOT, skipping ==="
    continue
  fi
  COVERAGE_CSV="${REPLICATE_DIR}coverage.csv"
  echo "=== scenario_$SCENARIO: $COVERAGE_CSV ==="
  command="$PYTHON src/postprocessing/coverage_band_plot.py $COVERAGE_CSV --output-png $OUT_DIR/scenario_${SCENARIO}_coverage_band.png"
  echo $command
  $command
done

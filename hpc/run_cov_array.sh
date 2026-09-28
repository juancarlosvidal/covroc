#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_cov_array.sh <input_scenario_dir> <output_dir> [n_replicates] [max_concurrent] [method] [ci_levels] [early_stop_patience]
# Example: ./hpc/run_cov_array.sh input_real_2/scenario_1 output/coverage/scenario_1
#          ./hpc/run_cov_array.sh input_real_2/scenario_1 output/coverage/scenario_1 15 5
#          (cap at 5 tasks running at once, e.g. if the queue/GPU quota needs it)
#          ./hpc/run_cov_array.sh input_real_2/scenario_4 output/validation/scenario_4_gaussian 3 "" gaussian "0.95,0.8,0.5"
#          (small validation run, 3 replicates, gaussian method, 3 CI levels from the
#          same bootstrap replicates -- leave [max_concurrent] empty to use it)
#          ./hpc/run_cov_array.sh input_real_2/scenario_4 output/validation3/scenario_4_gaussian 3 "" gaussian "0.95,0.8,0.5" 50
#          (same, but early_stop_patience=50 -- see py_cov_array.sh's $5 comment for why)
# method: oob (default) | parametric | gaussian -- see coverage_bootstrap_crossfit.py -m.
# ci_levels: comma-separated, default "0.95" -- oob only accepts a single level.
# early_stop_patience: default 10 (coverage_bootstrap_crossfit.py's own default).
cd "$(dirname "$0")/.."

N=${3:-15}
SPEC="0-$((N - 1))"
if [ -n "$4" ]; then
  SPEC="${SPEC}%${4}"
fi
METHOD=${5:-oob}
CI_LEVELS=${6:-0.95}
EARLY_STOP=${7:-10}

mkdir -p "$2"
sbatch --array="$SPEC" hpc/py_cov_array.sh "$1" "$2" "$METHOD" "$CI_LEVELS" "$EARLY_STOP"

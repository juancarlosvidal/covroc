#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_cov_array.sh <input_scenario_dir> <output_dir> [n_replicates] [max_concurrent]
# Example: ./hpc/run_cov_array.sh input_real_2/scenario_1 output/coverage/scenario_1
#          ./hpc/run_cov_array.sh input_real_2/scenario_1 output/coverage/scenario_1 15 5
#          (cap at 5 tasks running at once, e.g. if the queue/GPU quota needs it)
cd "$(dirname "$0")/.."

N=${3:-15}
SPEC="0-$((N - 1))"
if [ -n "$4" ]; then
  SPEC="${SPEC}%${4}"
fi

mkdir -p "$2"
sbatch --array="$SPEC" hpc/py_cov_array.sh "$1" "$2"

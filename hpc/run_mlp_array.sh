#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_mlp_array.sh <input_scenario_dir> <output_dir> [n_replicates] [max_concurrent]
# Example: ./hpc/run_mlp_array.sh input_real_2/scenario_7 output/fnn/scenario_7
#          ./hpc/run_mlp_array.sh input_real_2/scenario_7 output/fnn/scenario_7 200 10
#          (cap at 10 tasks running at once, e.g. if the GPU quota needs it)
cd "$(dirname "$0")/.."

N=${3:-200}
SPEC="0-$((N - 1))"
if [ -n "$4" ]; then
  SPEC="${SPEC}%${4}"
fi

mkdir -p "$2"
sbatch --array="$SPEC" hpc/py_mlp_array.sh "$1" "$2"

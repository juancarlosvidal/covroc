#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_baselines_array.sh [max_concurrent]
# Example: ./hpc/run_baselines_array.sh        # all 9 scenarios at once
#          ./hpc/run_baselines_array.sh 3       # cap at 3 tasks running at once
cd "$(dirname "$0")/.."

SPEC="1-9"
if [ -n "${1:-}" ]; then
  SPEC="${SPEC}%${1}"
fi

mkdir -p output/naive output/linear output/spline
sbatch --array="$SPEC" hpc/py_baselines_array.sh

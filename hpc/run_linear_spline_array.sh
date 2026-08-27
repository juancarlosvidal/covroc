#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_linear_spline_array.sh [scenario_spec]
# Example: ./hpc/run_linear_spline_array.sh        # all 9 scenarios at once
#          ./hpc/run_linear_spline_array.sh 4       # just scenario 4
cd "$(dirname "$0")/.."

SPEC="${1:-1-9}"

mkdir -p output/linear output/spline
sbatch --array="$SPEC" hpc/py_linear_spline_array.sh

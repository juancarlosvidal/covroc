#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_bnp_timing_array.sh [scenario_spec]
# Example: ./hpc/run_bnp_timing_array.sh                 # scenarios 2,3,4,5,6,7,9
#                                                          (skips 1 and 8, already timed)
#          ./hpc/run_bnp_timing_array.sh 1-9               # all 9, re-running 1 and 8 too
cd "$(dirname "$0")/.."

SPEC="${1:-2,3,4,5,6,7,9}"

mkdir -p output/aroc_bnp_timing
sbatch --array="$SPEC" hpc/r_bnp_timing_array.sh

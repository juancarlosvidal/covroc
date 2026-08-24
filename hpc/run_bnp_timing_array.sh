#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_bnp_timing_array.sh [scenario_spec]
# Example: ./hpc/run_bnp_timing_array.sh          # all 9 scenarios; r_bnp_timing_array.sh
#                                                   skips any (scenario, sample size) that
#                                                   already has a *_timing.csv, so this is
#                                                   safe to re-run against a
#                                                   partially-complete output/aroc_bnp_timing/
#                                                   to fill in whatever's missing
#          ./hpc/run_bnp_timing_array.sh 8          # just re-check/fill scenario 8
cd "$(dirname "$0")/.."

SPEC="${1:-1-9}"

mkdir -p output/aroc_bnp_timing
sbatch --array="$SPEC" hpc/r_bnp_timing_array.sh

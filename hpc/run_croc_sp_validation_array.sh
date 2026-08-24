#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_croc_sp_validation_array.sh [scenario_spec]
# Example: ./hpc/run_croc_sp_validation_array.sh          # all 9 scenarios; skips any
#                                                            (scenario, N, language) whose
#                                                            output CSV already exists
#          ./hpc/run_croc_sp_validation_array.sh 1          # just scenario 1
cd "$(dirname "$0")/.."

SPEC="${1:-1-9}"

mkdir -p output/croc_sp_validation
sbatch --array="$SPEC" hpc/croc_sp_validation_array.sh

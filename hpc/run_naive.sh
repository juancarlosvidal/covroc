#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_naive.sh <input_scenario_dir> <output_dir>
cd "$(dirname "$0")/.." && sbatch hpc/py_naive.sh "$1" "$2"

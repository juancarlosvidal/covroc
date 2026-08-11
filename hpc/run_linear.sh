#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_linear.sh <input_scenario_dir> <output_dir> [formula-type]
# Example: ./hpc/run_linear.sh input_real_2/scenario_1 output/linear/scenario_1 linear
#          ./hpc/run_linear.sh input_real_2/scenario_1 output/spline/scenario_1 spline
cd "$(dirname "$0")/.." && sbatch hpc/py_linear.sh "$1" "$2" "$3"

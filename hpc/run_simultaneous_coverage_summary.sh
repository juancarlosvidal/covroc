#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_simultaneous_coverage_summary.sh [root_dir]
cd "$(dirname "$0")/.." && sbatch hpc/py_simultaneous_coverage_summary.sh "$1"

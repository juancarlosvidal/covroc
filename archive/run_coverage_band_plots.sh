#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_coverage_band_plots.sh [combined_root]
cd "$(dirname "$0")/.." && sbatch hpc/py_coverage_band_plots.sh "$1"

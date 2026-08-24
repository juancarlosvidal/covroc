#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
cd "$(dirname "$0")/.." && sbatch hpc/py_bnp_summary.sh

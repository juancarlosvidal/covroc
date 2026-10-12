#!/bin/bash
# Submits the learning-rate sweep from the repository root, wherever it is invoked from.
#
# Usage: hpc/run_lr_sweep_array.sh [max_concurrent]
#        SEEDS="42 43 44 45 46" hpc/run_lr_sweep_array.sh 8
#
# 24 tasks = 4 learning rates x 6 horizon/sex combinations, seeds in sequence, CPU only.
# Afterwards:
#   python src/real_data/summarize_bias_experiment.py -i output/lr_sweep --reference lr1e-2
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"42 43 44"}
RATES=(${RATES:-1e-2 1e-3 5e-4 1e-4})
SPEC="1-$(( ${#RATES[@]} * 6 ))"
if [ -n "${1:-}" ]; then
  SPEC="${SPEC}%${1}"
fi

# Pre-create every directory so the array never races over them.
for r in "${RATES[@]}"; do
  for s in $SEEDS; do mkdir -p "output/lr_sweep/lr$r/seed$s"; done
done

echo "submitting $SPEC (${#RATES[@]} rates x 6 combinations, seeds: $SEEDS)"
sbatch --array="$SPEC" --export=ALL,SEEDS="$SEEDS",RATES="${RATES[*]}" \
  hpc/py_lr_sweep_array.sh

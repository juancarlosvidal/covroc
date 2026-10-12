#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# Override with: PYTHON=/path/to/python sbatch --array=1-24 py_lr_sweep_array.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Does a smaller learning rate make the fit less variable?
#
# WHY. The seed experiment measured the spread of a single fit at lr=1e-2: sigma = 0.032
# pooled, reaching 0.056 in the panels with fewest cases. That is a third of the excess
# the plug-in shows over the band, and it is what has made every configuration comparison
# so far uninterpretable -- architecture, weight decay, bias exemption all moved the
# answer by about as much as the seed does. 1e-2 is high for Adam on a network this size,
# and a learning rate too large for the problem shows up exactly as run-to-run spread:
# the optimiser bounces rather than settles.
#
# THE CRITERION IS SIGMA, NOT AGREEMENT. The learning rate has no principled a-priori
# value, so it has to be chosen by something, and the something must not be the AUC. The
# spread of the fit across seeds is an optimisation diagnostic: it is measured on the
# fits, says nothing about whether the plug-in agrees with the band, and cannot be gamed
# into a better-looking result. Picking the rate that minimises it is defensible; picking
# the rate that brings the plug-in onto the band is not.
#
# LAYOUT. 24 tasks = 4 learning rates x 6 horizon/sex combinations, each running the
# seeds in sequence, writing to the same <rate>/seed<N>/ layout the bias experiment used
# so summarize_bias_experiment.py reads it unchanged.
#
# Biases stay inside the L2 penalty here: the seed experiment settled that exempting them
# does nothing (pooled difference -0.004 over six combinations), so this sweep varies one
# thing only.

set -euo pipefail
# No cd: SLURM runs this from a spool directory, and the job already starts where sbatch
# was invoked. Same convention as the other py_*.sh scripts.

TASK=${SLURM_ARRAY_TASK_ID:?set --array, or SLURM_ARRAY_TASK_ID for a local test}
SEEDS=${SEEDS:-"42 43 44"}
RATES=(${RATES:-1e-2 1e-3 5e-4 1e-4})
OUT_ROOT=${OUT_ROOT:-output/lr_sweep}

N_COMBOS=6
LR_IDX=$(( (TASK - 1) / N_COMBOS ))
COMBO_IDX=$(( (TASK - 1) % N_COMBOS + 1 ))
LR=${RATES[$LR_IDX]}

ARGS=$($PYTHON src/real_data/select_k_per_group.py --sh | sed -n "${COMBO_IDX}p")
if [ -z "$ARGS" ]; then
  echo "no combination at index $COMBO_IDX -- the selector returned fewer lines" >&2
  exit 1
fi

echo "=== task $TASK | lr $LR | $ARGS | seeds: $SEEDS ==="
for SEED in $SEEDS; do
  DEST="$OUT_ROOT/lr$LR/seed$SEED"
  # Tolerate the race: several tasks create the first levels of this path at once on a
  # shared filesystem, where mkdir -p is not atomic.
  mkdir -p "$DEST" 2>/dev/null || true
  [ -d "$DEST" ] || { echo "could not create $DEST" >&2; exit 1; }
  command="$PYTHON src/real_data/local_knn_bands_real.py $ARGS \
    --seed $SEED -lr $LR --transform log --orientation auto -ns 1 \
    --device cpu --threads ${SLURM_CPUS_PER_TASK:-8} -o $DEST"
  echo "$command"
  $command
done

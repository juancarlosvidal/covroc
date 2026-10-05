#!/bin/bash
# Lanza coverage_local_knn.py sobre todos los escenarios.
#   ./hpc/run_localknn_all.sh                      (envia 9 trabajos a SLURM)
#   MODE=local nohup ./hpc/run_localknn_all.sh &   (secuencial en este proceso)
# Reanudable: salta las replicas cuyo coverage.csv ya existe.
# Variables: SCENARIOS, INPUT_ROOT, OUT_ROOT, K_LIST, N_POINTS, TRAIN_FRAC, EARLY_STOP,
#            PATTERN, CI_LEVELS, GRID_POINTS, LOCALIZE, ESTIMATOR, N_BOOT,
#            OVERWRITE, MODE, PYTHON
set -u
cd "$(dirname "$0")/.."

SCENARIOS=${SCENARIOS:-"1 2 3 4 5 6 7 8 9"}
INPUT_ROOT=${INPUT_ROOT:-input_real_2}
OUT_ROOT=${OUT_ROOT:-output/knn100}
K_LIST=${K_LIST:-5,10,15,20,50,100,300,1000}
N_POINTS=${N_POINTS:-100}
TRAIN_FRAC=${TRAIN_FRAC:-0.5}
EARLY_STOP=${EARLY_STOP:-40}
PATTERN=${PATTERN:-_5000_.*_data\.csv$}
CI_LEVELS=${CI_LEVELS:-0.95,0.8,0.5}
GRID_POINTS=${GRID_POINTS:-11}
LOCALIZE=${LOCALIZE:-model}
ESTIMATOR=${ESTIMATOR:-plain}
N_BOOT=${N_BOOT:-1000}
OVERWRITE=${OVERWRITE:-}
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

if [ -z "${MODE:-}" ]; then
  if command -v sbatch >/dev/null 2>&1; then MODE=sbatch; else MODE=local; fi
fi
OW_FLAG=""; [ -n "$OVERWRITE" ] && OW_FLAG="-ow"

echo "=== run_localknn_all.sh | modo=$MODE | k_list=$K_LIST | ci=$CI_LEVELS | gp=$GRID_POINTS | loc=$LOCALIZE | est=$ESTIMATOR | $(date) ==="
n_done=0
for S in $SCENARIOS; do
  IN="$INPUT_ROOT/scenario_$S"
  if [ ! -d "$IN" ]; then echo "!! $IN no existe -- saltando" >&2; continue; fi
  OUT="$OUT_ROOT/s${S}"
  mkdir -p "$OUT"
  if [ "$MODE" = "sbatch" ]; then
    echo "[$(date +%H:%M:%S)] sbatch  escenario $S -> $OUT"
    sbatch hpc/py_localknn.sh "$IN" "$OUT" "$K_LIST" "$N_POINTS" "$TRAIN_FRAC" "$EARLY_STOP" "$PATTERN" "$CI_LEVELS" "$GRID_POINTS" "$LOCALIZE" "$ESTIMATOR" "$N_BOOT"
  else
    echo "[$(date +%H:%M:%S)] INICIO  escenario $S -> $OUT"
    t0=$(date +%s)
    $PYTHON src/simulation/coverage_local_knn.py -i "$IN" -o "$OUT" -e 800 -es "$EARLY_STOP" \
      -kl "$K_LIST" -np "$N_POINTS" -tf "$TRAIN_FRAC" -ci "$CI_LEVELS" -gp "$GRID_POINTS" -loc "$LOCALIZE" -est "$ESTIMATOR" -nb "$N_BOOT" -pt "$PATTERN" $OW_FLAG
    echo "[$(date +%H:%M:%S)] FIN     escenario $S en $(( ($(date +%s)-t0)/60 )) min"
  fi
  n_done=$((n_done+1))
done
echo "=== $n_done escenarios | $(date) ==="

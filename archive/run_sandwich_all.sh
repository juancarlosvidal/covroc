#!/bin/bash
# Runs coverage_sandwich.py over every scenario and both residual variants.
#
#   Modo SLURM (por defecto): envia un trabajo por combinacion y termina enseguida.
#       ./hpc/run_sandwich_all.sh
#
#   Modo local (para nohup): ejecuta todo secuencialmente en ESTE proceso.
#       nohup ./hpc/run_sandwich_all.sh > sandwich_all.log 2>&1 &
#       tail -f sandwich_all.log
#     Poner MODE=local explicitamente tambien vale: MODE=local nohup ./hpc/run_sandwich_all.sh ...
#     Se auto-detecta: si no hay 'sbatch' en el PATH, usa modo local.
#
# Es REANUDABLE: coverage_sandwich.py salta cualquier replica cuyo coverage.csv ya exista,
# asi que si el proceso muere puedes relanzar el mismo comando y continua donde iba.
# Para recalcular desde cero, borra el directorio de salida (o pasa OVERWRITE=1).
#
# Coste medido (n=5000, 100 replicas/escenario): insample ~3.3h, oof ~5.5h por combinacion,
# ~79 GPU-horas en total para las 18. En modo local y secuencial eso son ~3.3 dias; en modo
# SLURM con varias GPUs a la vez, bastante menos (18 trabajos de 1 tarea, no roza el QOS).
#
# Variables de entorno:
#   SCENARIOS  lista de escenarios          (default "1 2 3 4 5 6 7 8 9")
#   VARIANTS   variantes de residuos        (default "insample oof")
#   INPUT_ROOT raiz de los datos            (default input_real_2)
#   OUT_ROOT   raiz de salida               (default output/sw100)
#   N_POINTS   sujetos evaluados/replica    (default 100)
#   CI_LEVELS  niveles de confianza         (default 0.95,0.8,0.5)
#   EARLY_STOP paciencia de early stopping  (default 40)
#   PATTERN    regex de fichero             (default solo n=5000)
#   TARGET     pointwise | avg                (default pointwise)
#   OVERWRITE  1 para recalcular            (default vacio)
#   MODE       sbatch | local               (default: sbatch si existe, si no local)
#   PYTHON     interprete en modo local
set -u

cd "$(dirname "$0")/.."

SCENARIOS=${SCENARIOS:-"1 2 3 4 5 6 7 8 9"}
VARIANTS=${VARIANTS:-"insample oof"}
INPUT_ROOT=${INPUT_ROOT:-input_real_2}
OUT_ROOT=${OUT_ROOT:-output/sw100}
N_POINTS=${N_POINTS:-100}
CI_LEVELS=${CI_LEVELS:-0.95,0.8,0.5}
EARLY_STOP=${EARLY_STOP:-40}
PATTERN=${PATTERN:-_5000_.*_data\.csv$}
OVERWRITE=${OVERWRITE:-}
TARGET=${TARGET:-pointwise}
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

if [ -z "${MODE:-}" ]; then
  if command -v sbatch >/dev/null 2>&1; then MODE=sbatch; else MODE=local; fi
fi

OW_FLAG=""
[ -n "$OVERWRITE" ] && OW_FLAG="-ow"

echo "=== run_sandwich_all.sh | modo=$MODE | $(date) ==="
echo "    escenarios: $SCENARIOS"
echo "    variantes : $VARIANTS"
echo "    salida    : $OUT_ROOT"
echo "    target    : $TARGET"
echo

n_done=0
for S in $SCENARIOS; do
  IN="$INPUT_ROOT/scenario_$S"
  if [ ! -d "$IN" ]; then
    echo "!! $IN no existe -- saltando escenario $S" >&2
    continue
  fi
  for V in $VARIANTS; do
    OUT="$OUT_ROOT/s${S}_${V}"
    mkdir -p "$OUT"
    if [ "$MODE" = "sbatch" ]; then
      echo "[$(date +%H:%M:%S)] sbatch  escenario $S / $V -> $OUT"
      sbatch hpc/py_sandwich.sh "$IN" "$OUT" "$N_POINTS" "$CI_LEVELS" "$EARLY_STOP" "$V" "$PATTERN" "$TARGET"
    else
      echo "[$(date +%H:%M:%S)] INICIO  escenario $S / $V -> $OUT"
      t0=$(date +%s)
      $PYTHON src/simulation/coverage_sandwich.py \
        -i "$IN" -o "$OUT" -e 800 -es "$EARLY_STOP" \
        -np "$N_POINTS" -ci "$CI_LEVELS" -r "$V" -pt "$PATTERN" -tg "$TARGET" $OW_FLAG
      rc=$?
      t1=$(date +%s)
      if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] !! FALLO (rc=$rc) escenario $S / $V tras $(( (t1-t0)/60 )) min -- continuo con el resto" >&2
      else
        echo "[$(date +%H:%M:%S)] FIN     escenario $S / $V en $(( (t1-t0)/60 )) min"
      fi
    fi
    n_done=$((n_done + 1))
  done
done

echo
echo "=== $n_done combinaciones $( [ "$MODE" = sbatch ] && echo enviadas || echo procesadas ) | $(date) ==="

#!/bin/bash
# Re-puntua el escenario VII contra la verdad reorientada, para la FNN y el Random Forest.
#
#   ./hpc/run_rescore_s7.sh            prueba de humo: 3 replicas por metodo (~1 min)
#   ./hpc/run_rescore_s7.sh full       la pasada completa: 100 replicas (~37 min cada una)
#
# Haz SIEMPRE la prueba de humo primero. Lo que decide todo es la linea de auto-chequeo:
#
#   OK: la reconstruccion reproduce la cifra publicada  -> la MSE orientada es fiable y
#                                                          sustituye a la de la tabla
#   !! la reconstruccion NO reproduce la cifra publicada -> no la uses; hay que relanzar
#                                                          el escenario entero
#
# Si la prueba de humo falla, no gastes los 37 minutos.
#
# Variables de entorno:
#   METHODS    cuales lanzar: "fnn rf", "fnn", "rf"  (default "fnn rf")
#              p.ej. METHODS=rf ./hpc/run_rescore_s7.sh  para no repetir la FNN
#   FNN_DIR    carpeta del escenario 7 de la FNN   (default output/fnn/scenario_7)
#   RF_DIR     idem del Random Forest             (default output/rf/scenario_7/mortstat_rf)
#   OUT_DIR    donde dejar los CSV y los logs     (default output/rescore_s7)
#   PATTERN    tamano muestral                    (default _5000_)
#   MODE       sbatch | local                     (default: sbatch si existe)
#   PYTHON     interprete en modo local
set -u
cd "$(dirname "$0")/.."

MODE_ARG=${1:-smoke}
case "$MODE_ARG" in
  smoke) LIMIT=3  ;;
  full)  LIMIT=0  ;;
  *) echo "uso: $0 [smoke|full]" >&2; exit 2 ;;
esac

FNN_DIR=${FNN_DIR:-output/fnn/scenario_7}
RF_DIR=${RF_DIR:-output/rf/scenario_7/mortstat_rf}
METHODS=${METHODS:-"fnn rf"}
OUT_DIR=${OUT_DIR:-output/rescore_s7}
PATTERN=${PATTERN:-_5000_}
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

if [ -z "${MODE:-}" ]; then
  if command -v sbatch >/dev/null 2>&1; then MODE=sbatch; else MODE=local; fi
fi

SUF=$( [ "$LIMIT" -eq 0 ] && echo full || echo smoke )
# El tamano muestral entra en el nombre: sin esto, una pasada con PATTERN='_20000_'
# escribiria sobre el CSV de la de '_5000_', que cuesta 37 min y no se recupera.
SIZE_TAG=$(printf '%s' "$PATTERN" | tr -cd '0-9')
[ -n "$SIZE_TAG" ] && SUF="${SUF}_n${SIZE_TAG}"
mkdir -p "$OUT_DIR"

echo "=== run_rescore_s7.sh | $MODE_ARG | modo=$MODE | $(date) ==="
echo "    patron : $PATTERN"
echo "    salida : $OUT_DIR"
echo

n_sent=0
n_fail=0
for TAG in $METHODS; do
  case "$TAG" in
    fnn) DIR=$FNN_DIR ;;
    rf)  DIR=$RF_DIR  ;;
    *) echo "!! metodo desconocido '$TAG' (usa fnn, rf, o ambos)" >&2; continue ;;
  esac
  if [ ! -d "$DIR" ]; then
    echo "!! $DIR no existe -- saltando $TAG" >&2
    echo "   (si la ruta del RF es otra, relanza con RF_DIR=... $0 $MODE_ARG)" >&2
    continue
  fi
  CSV="$OUT_DIR/rescore_${TAG}_s7_${SUF}.csv"
  if [ "$MODE" = "sbatch" ]; then
    echo "[$(date +%H:%M:%S)] sbatch  $TAG  $DIR -> $CSV"
    sbatch --job-name="rescore_${TAG}" \
           --output="$OUT_DIR/rescore_${TAG}_${SUF}_%j.log" \
           hpc/py_rescore_s7.sh "$DIR" "$CSV" "$PATTERN" "$LIMIT"
  else
    echo "[$(date +%H:%M:%S)] INICIO  $TAG  $DIR -> $CSV"
    t0=$(date +%s)
    $PYTHON src/simulation/rescore_scenario7_orientation.py \
      -i "$DIR" -o "$CSV" -pt "$PATTERN" -l "$LIMIT" -s 7 -nmc 20000
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] !! FALLO (rc=$rc) en $TAG -- continuo con el resto" >&2
      n_fail=$((n_fail+1))
    else
      echo "[$(date +%H:%M:%S)] FIN     $TAG en $(( ($(date +%s)-t0)/60 )) min"
    fi
  fi
  n_sent=$((n_sent+1))
done

echo
echo "=== $n_sent trabajo(s) $( [ "$MODE" = sbatch ] && echo enviados || echo procesados ) | $(date) ==="
if [ "$n_fail" -gt 0 ]; then echo "!! $n_fail fallaron -- mira los mensajes de arriba" >&2; fi
if [ "$MODE" = "sbatch" ] && [ "$n_sent" -gt 0 ]; then
  echo
  echo "Para ver el resultado cuando acaben:"
  echo "  grep -A2 -E 'OK:|!!' $OUT_DIR/rescore_*_${SUF}_*.log"
fi

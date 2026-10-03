#!/bin/bash
# Re-corre los tres baselines CPU (naive, linear, spline) sobre el escenario VII contra la
# verdad reorientada (commit 95fc290), como tres trabajos en paralelo.
#
#   ./hpc/run_baselines_s7.sh           solo n=5000  (~37 min cada uno)
#   ./hpc/run_baselines_s7.sh both      ambos tamanos (~3.1 h cada uno)
#
# A diferencia de la FNN y el Random Forest, estos tres no guardan nada por sujeto, asi
# que no se pueden re-puntuar con rescore_scenario7_orientation.py -- hay que reajustarlos.
# El ajuste es gratis (sub-segundo por replica); el coste es recalcular la verdad Monte
# Carlo, que es lo mismo que pagaria la re-puntuacion.
#
# Esto es lo que decide la fila VII de tab:roc_mse_ground_truth: ahi el semiparametrico
# (spline) figura como ganador con 0.0021, y ese numero esta contra el blanco viejo.
#
# La salida va a carpetas NUEVAS (sufijo _s7fix) para no pisar la corrida publicada --
# asi puedes comparar las dos cifras en vez de perder la antigua.
#
# Variables de entorno:
#   BASELINES  cuales lanzar          (default "naive linear spline")
#   INPUT_DIR  datos del escenario    (default input_real_2/scenario_7)
#   OUT_ROOT   raiz de salida         (default output)
#   SUFFIX     sufijo de carpeta      (default _s7fix)
#   MODE       sbatch | local         (default: sbatch si existe)
#   PYTHON     interprete en modo local
set -u
cd "$(dirname "$0")/.."

SIZE_ARG=${1:-n5000}
case "$SIZE_ARG" in
  n5000)  PATTERN='_5000_'  ;;
  n20000) PATTERN='_20000_' ;;
  both)   PATTERN=''        ;;
  *) echo "uso: $0 [n5000|n20000|both]" >&2; exit 2 ;;
esac

BASELINES=${BASELINES:-"naive linear spline"}
INPUT_DIR=${INPUT_DIR:-input_real_2/scenario_7}
OUT_ROOT=${OUT_ROOT:-output}
SUFFIX=${SUFFIX:-_s7fix}
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

if [ -z "${MODE:-}" ]; then
  if command -v sbatch >/dev/null 2>&1; then MODE=sbatch; else MODE=local; fi
fi

if [ ! -d "$INPUT_DIR" ]; then
  echo "!! $INPUT_DIR no existe" >&2
  exit 1
fi

echo "=== run_baselines_s7.sh | $SIZE_ARG | modo=$MODE | $(date) ==="
echo "    entrada  : $INPUT_DIR"
echo "    patron   : ${PATTERN:-(todas las replicas)}"
echo "    baselines: $BASELINES"
echo

mkdir -p "$OUT_ROOT/logs"
n_sent=0
n_fail=0
for B in $BASELINES; do
  case "$B" in
    naive|linear|spline) ;;
    *) echo "!! baseline desconocido '$B' -- usa naive, linear o spline" >&2; continue ;;
  esac
  OUT="$OUT_ROOT/${B}${SUFFIX}/scenario_7"
  mkdir -p "$OUT"
  if [ "$MODE" = "sbatch" ]; then
    echo "[$(date +%H:%M:%S)] sbatch  $B -> $OUT"
    sbatch --job-name="s7fix_${B}" \
           --output="$OUT_ROOT/logs/s7fix_${B}_%j.log" \
           hpc/py_baseline_one.sh "$B" "$INPUT_DIR" "$OUT" "$PATTERN"
  else
    echo "[$(date +%H:%M:%S)] INICIO  $B -> $OUT"
    t0=$(date +%s)
    PYTHON="$PYTHON" bash hpc/py_baseline_one.sh "$B" "$INPUT_DIR" "$OUT" "$PATTERN"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] !! FALLO (rc=$rc) en $B -- continuo con el resto" >&2
      n_fail=$((n_fail+1))
    else
      echo "[$(date +%H:%M:%S)] FIN     $B en $(( ($(date +%s)-t0)/60 )) min"
    fi
  fi
  n_sent=$((n_sent+1))
done

echo
echo "=== $n_sent trabajo(s) $( [ "$MODE" = sbatch ] && echo enviados || echo procesados ) | $(date) ==="
[ "$n_fail" -gt 0 ] && echo "!! $n_fail fallaron -- mira los mensajes de arriba" >&2
if [ "$MODE" = "sbatch" ] && [ "$n_sent" -gt 0 ]; then
  echo
  echo "Cuando acaben, la MSE media de cada uno:"
  echo "  python3 src/postprocessing/statistics_summary.py -i $OUT_ROOT/spline$SUFFIX  # etc."
  echo "o directamente:"
  echo "  for d in $OUT_ROOT/{naive,linear,spline}$SUFFIX/scenario_7; do"
  echo "    echo -n \"\$d: \"; find \$d -name roc_mse_values.csv -exec cat {} + \\"
  echo "      | awk -F, '\$2+0==\$2 {s+=\$2; n++} END {printf \"%.5f  (%d filas)\\n\", s/n, n}'"
  echo "  done"
fi

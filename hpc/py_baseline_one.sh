#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
# Override with: PYTHON=/path/to/python sbatch py_baseline_one.sh <naive|linear|spline> <input_dir> <output_dir> [pattern]
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# UN baseline sobre UN escenario, para poder lanzar los tres en paralelo en lugar de
# encadenados como hace py_baselines_array.sh. Existe para el re-calculo del escenario VII
# tras reorientar su verdad (commit 95fc290): ahi el ajuste no cuesta nada (naive ~1 ms,
# linear/spline ~0.1 s por replica) y todo el tiempo se va en el Monte Carlo de la verdad,
# ~22 s por replica a n=5000 y ~88 s a n=20000. Encadenar los tres multiplica por tres un
# tiempo que no tiene por que ser secuencial.
#
# Con -pt '_5000_' son ~37 min; sin filtro, las 200 replicas de ambos tamanos, ~3.1 h.
# El limite de 8 h cubre el caso sin filtro con margen.
#
# $1: baseline -- naive | linear | spline
# $2: carpeta del escenario (e.g. input_real_2/scenario_7)
# $3: carpeta de salida
# $4: pattern de replica (default vacio = todas; usa '_5000_' para un solo tamano)
BASELINE=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
PATTERN=${4:-}

mkdir -p "$OUTPUT_DIR"

case "$BASELINE" in
  naive)
    SCRIPT="src/simulation/naive_roc_baseline.py"
    EXTRA=""
    ;;
  linear)
    SCRIPT="src/simulation/linear_reg_data_simulation.py"
    EXTRA="--formula-type linear"
    ;;
  spline)
    SCRIPT="src/simulation/linear_reg_data_simulation.py"
    EXTRA="--formula-type spline"
    ;;
  *)
    echo "!! baseline desconocido '$BASELINE' -- usa naive, linear o spline" >&2
    exit 2
    ;;
esac

PT_FLAG=""
[ -n "$PATTERN" ] && PT_FLAG="-pt $PATTERN"

command="$PYTHON $SCRIPT -i $INPUT_DIR -o $OUTPUT_DIR $EXTRA $PT_FLAG"
echo $command
$command

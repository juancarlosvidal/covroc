#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
# Override with: PYTHON=/path/to/python sbatch py_rescore_s7.sh <input_dir> <output_csv> [pattern] [limit] [scenario] [n_mc]
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Re-puntua una corrida ya hecha del escenario VII contra la verdad reorientada
# (commit 95fc290), SIN reentrenar: reconstruye a y b desde las columnas Predicted Mean /
# Predicted Std Dev que ya estan en full_results_{healthy,diseased}_*.csv.
#
# Sin GPU: todo el coste es el Monte Carlo de la verdad (ordenar 20.000 sorteos por
# sujeto). Medido ~22 s por replica a n=5000, o sea ~37 min para 100. El limite de 4 h
# deja margen de sobra incluso a n=20000 (~4x mas caro por replica).
#
# La memoria la fija el troceado de true_roc_curve: 2.000 sujetos x 20.000 sorteos
# = 320 MB por array, con varias copias ordenadas vivas a la vez. 24G es holgado.
#
# $1: carpeta del escenario con las subcarpetas por replica
#     (e.g. output/fnn/scenario_7, output/fnn_rf/scenario_7)
# $2: CSV de salida con la tabla por replica
# $3: pattern de replica (default solo n=5000)
# $4: limit, 0 = todas (default 0). Poner 3 para la prueba de humo.
# $5: scenario (default 7)
# $6: n_mc, sorteos Monte Carlo por sujeto (default 20000 -- debe coincidir con la
#     corrida original, o la verdad no sera la misma)
INPUT_DIR=$1
OUTPUT_CSV=$2
PATTERN=${3:-_5000_}
LIMIT=${4:-0}
SCENARIO=${5:-7}
N_MC=${6:-20000}

mkdir -p "$(dirname "$OUTPUT_CSV")"
command="$PYTHON src/simulation/rescore_scenario7_orientation.py -i $INPUT_DIR -o $OUTPUT_CSV -pt $PATTERN -l $LIMIT -s $SCENARIO -nmc $N_MC"
echo $command
$command

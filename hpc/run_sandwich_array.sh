#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_sandwich_array.sh <input_scenario_dir> <output_dir> [n_replicates] [max_concurrent] [n_points] [ci_levels] [early_stop_patience] [residuals] [start_index]
# Example: ./hpc/run_sandwich_array.sh input_real_2/scenario_4 output/sandwich/scenario_4 3
#          ./hpc/run_sandwich_array.sh input_real_2/scenario_4 output/sandwich/scenario_4 15 "" 100 "0.95,0.8,0.5" 40
# start_index: primer indice del array (default 0). Para pasar del limite de QOS, envia
#   en tandas: ... 25 "" ... 0, esperar, luego ... 25 "" ... 25, etc. py_*_array.sh indexa
#   directamente sobre la lista ordenada de ficheros, asi que el offset selecciona replicas
#   distintas sin repetir.
# residuals: insample (default) | oof -- see py_sandwich.sh. Use a different output dir per value.
# Keep n_replicates x scenarios under ~30 submitted tasks at a time (SLURM QOS submit
# limit -- the %max_concurrent throttle does NOT help, it caps running not submitted).
cd "$(dirname "$0")/.."

N=${3:-15}
START=${9:-0}
SPEC="${START}-$((START + N - 1))"
if [ -n "$4" ]; then
  SPEC="${SPEC}%${4}"
fi
N_POINTS=${5:-100}
CI_LEVELS=${6:-0.95,0.8,0.5}
EARLY_STOP=${7:-40}
RESIDUALS=${8:-insample}

mkdir -p "$2"
sbatch --array="$SPEC" hpc/py_sandwich_array.sh "$1" "$2" "$N_POINTS" "$CI_LEVELS" "$EARLY_STOP" "$RESIDUALS"

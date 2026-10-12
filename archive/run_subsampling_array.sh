#!/bin/bash
# Submits from the repository root regardless of where this script is invoked from
# Usage: hpc/run_subsampling_array.sh <input_scenario_dir> <output_dir> [n_replicates] [max_concurrent] [m_fracs] [B] [n_points] [early_stop_patience] [start_index]
# Example: ./hpc/run_subsampling_array.sh input_real_2/scenario_1 output/subsampling/scenario_1 3
# start_index (9o): primer indice del array (default 0), para trocear por el limite de QOS.
# Expect ~1-1.5h per replicate at n=5000 with the defaults -- see py_subsampling_array.sh.
cd "$(dirname "$0")/.."

N=${3:-3}
START=${9:-0}
SPEC="${START}-$((START + N - 1))"
if [ -n "$4" ]; then
  SPEC="${SPEC}%${4}"
fi
M_FRACS=${5:-0.1,0.2,0.4}
B_REPS=${6:-100}
N_POINTS=${7:-100}
EARLY_STOP=${8:-40}

mkdir -p "$2"
sbatch --array="$SPEC" hpc/py_subsampling_array.sh "$1" "$2" "$M_FRACS" "$B_REPS" "$N_POINTS" "$EARLY_STOP"

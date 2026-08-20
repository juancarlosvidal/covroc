#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# Override with: PYTHON=/path/to/python sbatch --array=0-N py_mlp_array.sh <input_dir> <output_dir>
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# mlp_reg_data_simulation_multi.py processes every file it finds in -i sequentially, in one
# GPU allocation. For scenario_7 (whose Monte Carlo ground-truth evaluation adds cost on
# top of the usual FNN fit), job 5520906 -- run with --time=10:00:00 -- confirmed via
# `sacct` (TimelimitRaw=600, State=TIMEOUT) that it hit that limit after only 31/200
# replicates: ~19.4 min/replicate average, i.e. ~64.5h for all 200 run sequentially. This
# script instead makes each SLURM array task responsible for exactly one replicate CSV, so
# replicates run concurrently across as many GPUs as the cluster/queue gives this array job
# at once -- same pattern as py_cov_array.sh. --time=03:00:00 is a generous per-task cap
# (>9x the observed average) since one task now does one replicate, not 200.
#
# $1: scenario input dir (e.g. input_real_2/scenario_7)
# $2: output dir (e.g. output/fnn/scenario_7)
# SLURM_ARRAY_TASK_ID selects the (0-indexed) file from the sorted file list -- both
# n=5000 and n=20000 files included (unlike py_cov_array.sh, which splits by size).
INPUT_DIR=$1
OUTPUT_DIR=$2

FILES=($(ls "$INPUT_DIR" | grep '_data\.csv$' | sort -V))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
if [ -z "$FILE" ]; then
  echo "No file at array index $SLURM_ARRAY_TASK_ID in $INPUT_DIR (found ${#FILES[@]} total)" >&2
  exit 1
fi

# Skip replicates already computed (e.g. by an earlier partial sequential/array run) --
# mlp_reg_data_simulation_multi.py's output subfolder name is the input filename with
# "_data.csv" stripped.
SCENARIO=${FILE%_data.csv}
if [ -f "$OUTPUT_DIR/$SCENARIO/timing.csv" ]; then
  echo "$OUTPUT_DIR/$SCENARIO/timing.csv already exists -- skipping $FILE"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
# mlp_reg_data_simulation_multi.py takes a *directory* and processes everything in it, so
# stage a scratch directory containing only this task's one file (symlinked, no copy) to
# isolate it from the other replicates sitting in $INPUT_DIR.
SCRATCH=$(mktemp -d "$OUTPUT_DIR/.task_XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
ln -s "$(realpath "$INPUT_DIR/$FILE")" "$SCRATCH/$FILE"

command="$PYTHON src/simulation/mlp_reg_data_simulation_multi.py -i $SCRATCH -o $OUTPUT_DIR -c 6 -f 5 -e 800 -b 64"
echo $command
$command

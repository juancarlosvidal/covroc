#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch --array=0-N py_sandwich_array.sh <input_dir> <output_dir> ...
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# One SLURM array task per replicate CSV, same structure as py_cov_array.sh. For the
# sandwich method this is only worth it for a multi-scenario sweep -- a single replicate is
# cheap enough (minutes, see py_sandwich.sh's comment) that a sequential run over one
# scenario folder is usually fine.
#
# $1: scenario input dir (e.g. input_real_2/scenario_4) -- only its *_5000_*_data.csv files
# $2: output dir (e.g. output/sandwich/scenario_4)
# $3: n_points per replicate (default 100)
# $4: ci_levels, comma-separated (default "0.95,0.8,0.5")
# $5: early_stop_patience (default 40 -- see py_sandwich.sh's comment on why not 10)
# SLURM_ARRAY_TASK_ID selects the (0-indexed) file from the sorted n=5000 file list.
# $6: residuals -- insample (default) | oof. "oof" trains the variance net on genuinely
#     out-of-fold residuals (K extra mean-net fits per group, ~+100-300s/replicate),
#     removing the downward sigma bias in-sample residuals carry. Different values MUST
#     use different output dirs (the Method column differs but coverage.csv's path does not).
INPUT_DIR=$1
OUTPUT_DIR=$2
N_POINTS=${3:-100}
CI_LEVELS=${4:-0.95,0.8,0.5}
EARLY_STOP=${5:-40}
RESIDUALS=${6:-insample}

FILES=($(ls "$INPUT_DIR" | grep '_5000_.*_data\.csv$' | sort -V))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
if [ -z "$FILE" ]; then
  echo "No n=5000 file at array index $SLURM_ARRAY_TASK_ID in $INPUT_DIR (found ${#FILES[@]} total)" >&2
  exit 1
fi

SCENARIO=${FILE%_data.csv}
if [ -f "$OUTPUT_DIR/$SCENARIO/coverage.csv" ]; then
  echo "$OUTPUT_DIR/$SCENARIO/coverage.csv already exists -- skipping $FILE"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
# coverage_sandwich.py takes a *directory*, so stage a scratch dir holding only this task's
# one file (symlinked, no copy) -- same trick as py_cov_array.sh.
SCRATCH=$(mktemp -d "$OUTPUT_DIR/.task_XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
ln -s "$(realpath "$INPUT_DIR/$FILE")" "$SCRATCH/$FILE"

command="$PYTHON src/simulation/coverage_sandwich.py -i $SCRATCH -o $OUTPUT_DIR -e 800 -es $EARLY_STOP -np $N_POINTS -ci $CI_LEVELS -r $RESIDUALS"
echo $command
$command

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch --array=0-N py_cov_array.sh <input_dir> <output_dir>
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# coverage_bootstrap_crossfit.py processes every file it finds in -i sequentially, in one
# GPU allocation (for f in onlyfiles: ...). Measured cost: ~2.45h/replicate at n=5000,
# ~9.1h/replicate at n=20000 (-b 150 -k 5 -e 800) -- pointing -i at a whole 100-replicate
# scenario folder means one GPU grinding through it for weeks. This script instead makes
# each SLURM array task responsible for exactly one replicate CSV, so replicates run
# concurrently across as many GPUs as the cluster/queue gives this array job at once.
#
# $1: scenario input dir (e.g. input_real_2/scenario_1) -- only its *_5000_*_data.csv
#     files are considered; see run_cov_array.sh to also cover n=20000.
# $2: output dir (e.g. output/coverage/scenario_1)
# $3: confidence-band method -- oob (default) | parametric | gaussian, forwarded as
#     coverage_bootstrap_crossfit.py's -m (see that script's docstring). Different methods
#     must use different output dirs: coverage.csv's path doesn't encode the method, so a
#     second method run into the same $OUTPUT_DIR/$SCENARIO would overwrite the first's.
# $4: confidence level(s), comma-separated (e.g. "0.95,0.8,0.5"), forwarded as
#     coverage_bootstrap_crossfit.py's -ci. Multiple levels for parametric/gaussian cost
#     nothing extra (same bootstrap replicates, just re-percentiled); oob only accepts one.
# $5: early_stop_patience (default 10). variance_decomposition_diagnostic.py showed early
#     stopping firing at epoch ~14-19 out of an 800 budget with the default 10, and that
#     raising this to 50 cut the training-noise share of bootstrap variance substantially
#     (~40-60% down to ~12-35%) -- but it also shrinks total bootstrap variance, which may
#     help an over-covering scenario and hurt an already-under-covering one (bias-driven)
#     scenario. This run is what actually checks real coverage/PIT, not just variance.
# SLURM_ARRAY_TASK_ID selects the (0-indexed) file from the sorted n=5000 file list.
INPUT_DIR=$1
OUTPUT_DIR=$2
METHOD=${3:-oob}
CI_LEVELS=${4:-0.95}
EARLY_STOP=${5:-10}

FILES=($(ls "$INPUT_DIR" | grep '_5000_.*_data\.csv$' | sort -V))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
if [ -z "$FILE" ]; then
  echo "No n=5000 file at array index $SLURM_ARRAY_TASK_ID in $INPUT_DIR (found ${#FILES[@]} total)" >&2
  exit 1
fi

# Skip replicates already computed (e.g. by an earlier sequential py_cov.sh run, or a
# previous array submission) -- coverage_bootstrap_crossfit.py's output subfolder name is
# the input filename with "_data.csv" stripped.
SCENARIO=${FILE%_data.csv}
if [ -f "$OUTPUT_DIR/$SCENARIO/coverage.csv" ]; then
  echo "$OUTPUT_DIR/$SCENARIO/coverage.csv already exists -- skipping $FILE"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
# coverage_bootstrap_crossfit.py takes a *directory* and processes everything in it, so
# stage a scratch directory containing only this task's one file (symlinked, no copy) to
# isolate it from the other 99 replicates sitting in $INPUT_DIR.
SCRATCH=$(mktemp -d "$OUTPUT_DIR/.task_XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
ln -s "$(realpath "$INPUT_DIR/$FILE")" "$SCRATCH/$FILE"

command="$PYTHON src/simulation/coverage_bootstrap_crossfit.py -i $SCRATCH -o $OUTPUT_DIR -b 150 -k 5 -e 800 -es $EARLY_STOP -m $METHOD -ci $CI_LEVELS"
echo $command
$command

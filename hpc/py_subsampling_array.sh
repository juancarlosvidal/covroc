#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch --array=0-N py_subsampling_array.sh <input_dir> <output_dir> ...
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Simultaneous band by subsampling (coverage_subsampling.py). Cost is dominated by the
# B subsample refits, so it sits BETWEEN the sandwich and the bootstrap: at n=5000/group
# with the defaults below (m/n = 0.1/0.2/0.4, B = 40/40/100) expect roughly 1-1.5h per
# replicate, vs ~2 min for coverage_sandwich.py and 2.5-11h for the bootstrap methods.
# Use this for the SIMULTANEOUS band (which the sandwich does not give); keep the sandwich
# for pointwise intervals, where it is far cheaper and carries no training noise.
#
# $1: scenario input dir (e.g. input_real_2/scenario_1) -- only its *_5000_*_data.csv files
# $2: output dir (e.g. output/subsampling/scenario_1)
# $3: m_fracs, comma-separated subsample fractions m/n (default "0.1,0.2,0.4"). MORE THAN
#     ONE is what makes the convergence rate beta estimable instead of assumed -- see
#     coverage_subsampling.py's docstring on why assuming sqrt-n (beta=0.5) is risky here.
#     Classical theory wants m/n -> 0, so keep the largest fraction well below 1.
# $4: B at the largest m, the one that builds the band (default 100)
# $5: n_points, grid size the sup is taken over (default 100)
# $6: early_stop_patience (default 40, as for the sandwich)
# SLURM_ARRAY_TASK_ID selects the (0-indexed) file from the sorted n=5000 file list.
INPUT_DIR=$1
OUTPUT_DIR=$2
M_FRACS=${3:-0.1,0.2,0.4}
B_REPS=${4:-100}
N_POINTS=${5:-100}
EARLY_STOP=${6:-40}

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
SCRATCH=$(mktemp -d "$OUTPUT_DIR/.task_XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
ln -s "$(realpath "$INPUT_DIR/$FILE")" "$SCRATCH/$FILE"

command="$PYTHON src/simulation/coverage_subsampling.py -i $SCRATCH -o $OUTPUT_DIR -e 800 -es $EARLY_STOP -mf $M_FRACS -b $B_REPS -np $N_POINTS"
echo $command
$command

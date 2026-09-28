#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch py_variance_diag.sh <input_file> <output_dir> [n_repeat] [early_stop_patience]
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# variance_decomposition_diagnostic.py's two arms both cost about the same as one
# coverage_bootstrap_crossfit.py bootstrap replicate each (K folds x 2 networks x 2
# groups), so total cost is roughly (2*n_repeat + 2) / 150 of a full -b 150 run, i.e. at
# n_repeat=30 (default) and the ~2.45h/replicate-run cost measured at n=5000/-b 150 in
# py_cov_array.sh's comment, expect on the order of ~1h here (much cheaper than a full
# validation run -- this is meant to run on ONE replicate file at a time, not an array).
#
# $1: a single scenario replicate CSV (e.g. input_real_2/scenario_1/scenario_1_5000_1_data.csv)
# $2: output dir (e.g. output/variance_decomposition/scenario_1)
# $3: n_repeat per arm (default 30)
# $4: early_stop_patience (default 10, coverage_bootstrap_crossfit.py's own default) --
#     the first run showed early stopping firing at epoch ~14-19 out of an 800 budget, so
#     the fitted model may depend heavily on _fit_group_bootstrap's unseeded inner
#     train/val split at that early a stopping point. Raising this lets training run
#     longer before the val-loss plateau check stops it, testing whether that instability
#     is what's driving the training-noise share of bootstrap variance. Cost scales with
#     however many extra epochs actually get used before re-plateauing, not with the cap
#     itself -- expect a few times the baseline run's cost, not 5x flat.
INPUT_FILE=$1
OUTPUT_DIR=$2
N_REPEAT=${3:-30}
EARLY_STOP=${4:-10}

command="$PYTHON src/simulation/variance_decomposition_diagnostic.py -i $INPUT_FILE -o $OUTPUT_DIR -n $N_REPEAT -k 5 -e 800 -es $EARLY_STOP"
echo $command
$command

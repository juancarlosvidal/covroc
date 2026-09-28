#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch py_sandwich.sh <input_dir> <output_dir> [n_points] [ci_levels] [early_stop_patience] [residuals] [pattern]
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# coverage_sandwich.py is MUCH cheaper per replicate than coverage_bootstrap_crossfit.py:
# it fits 4 networks once (no bootstrap replicates, no cross-fitting) and then does a
# conjugate-gradient solve per evaluated subject. Benchmarked on CPU at n=500/group:
# ~1.3s to fit + ~0.25s per evaluated subject, so with the default --n_points 100 expect
# on the order of minutes per replicate rather than the 2.5-11h/replicate the bootstrap
# methods cost. That means, unlike py_cov.sh, pointing -i at a whole scenario folder is
# actually viable here -- though for a full 9-scenario sweep prefer run_sandwich_array.sh.
#
# $1: input dir (a scenario folder, or a small curated folder of replicate CSVs)
# $2: output dir (e.g. output/sandwich/scenario_4)
# $3: n_points -- subjects evaluated per replicate (default 100). NOT all N: the sandwich
#     variance needs a CG solve per query point, so evaluating all ~5000 would cost about
#     what the bootstrap does. See coverage_sandwich.py's module docstring.
# $4: ci_levels, comma-separated (default "0.95,0.8,0.5") -- extra levels are free, they
#     reuse the same standard error.
# $5: early_stop_patience (default 40, NOT the pipeline's usual 10). The sandwich/delta
#     approximation linearizes around a STATIONARY point of the training loss; with
#     patience 10 the networks stop at epoch ~15-20 nowhere near one, and the resulting
#     near-singular/indefinite Hessian produced degenerate standard errors (exactly 0 for
#     some subjects, ~1e6 for others) in local testing. Raising patience fixed that. The
#     run prints a "mean training-score norm" per group -- it should be small; if it is
#     large, the CIs from that replicate should not be trusted (no calibrated threshold
#     for this has been established yet).
# $8: target -- pointwise (default) | avg. "avg" hace inferencia sobre el funcional
#     promediado mean_i AUC(x_i): UN numero por replica, una sola resolucion CG en vez de
#     n_points, y ~0.3s en vez de ~44s. Ver averaged_auc_ci en sandwich_delta_ci.py.
# $7: pattern -- regex sobre el nombre de fichero (default: solo n=5000). Sin esto, una
#     ejecucion secuencial sobre input_real_2/scenario_N procesaria TAMBIEN los ficheros
#     n=20000 que viven en la misma carpeta, a 4x el coste.
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
PATTERN=${7:-_5000_.*_data\.csv$}
TARGET=${8:-pointwise}

mkdir -p "$OUTPUT_DIR"

command="$PYTHON src/simulation/coverage_sandwich.py -i $INPUT_DIR -o $OUTPUT_DIR -e 800 -es $EARLY_STOP -np $N_POINTS -ci $CI_LEVELS -r $RESIDUALS -pt "$PATTERN" -tg $TARGET"
echo $command
$command

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
# Override with: RSCRIPT=/path/to/Rscript sbatch --array=1-9 r_bnp_timing_array.sh
RSCRIPT=${RSCRIPT:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/r-covroc/bin/Rscript}

# One array task = one scenario, running R/aroc_bnp_timing.R's single-replicate timing
# of ROCnReg::AROC.bnp at both sample sizes (Reviewer 1, Major Concern 2 -- the same
# review response commitment py_baselines_array.sh covers for naive/linear/spline).
# CPU-only, no GPU needed. Covariate lists mirror true_dgp._COVARIATE_COLUMNS in
# src/simulation/true_dgp.py -- keep in sync if that ever changes. Skips any (scenario,
# sample size) that already has a *_timing.csv under output/aroc_bnp_timing/ -- safe to
# re-run/extend against a partially-complete output dir instead of needing a hardcoded
# list of what's already done (which is exactly what silently left scenario 8's n=20000
# untimed the first time this ran).
#
# Measured on scenario_1 (1 covariate): ~382s at n=5000, ~1569s at n=20000 (roughly
# linear in N). Measured on scenario_8 (4 covariates) at n=5000: ~434s, only ~14% over
# scenario_1's 1-covariate cost -- N dominates over covariate dimension. Worst case
# across all 9 scenarios at n=20000 is expected to stay well under this job's 2h budget.
SCENARIO=$SLURM_ARRAY_TASK_ID

case $SCENARIO in
  1) COVARIATES="x_D_1" ;;
  2) COVARIATES="x_D_1" ;;
  3) COVARIATES="x_D_1" ;;
  4) COVARIATES="x_D_1" ;;
  5) COVARIATES="x_D_1 x_D_2" ;;
  6) COVARIATES="x_D_1 x_D_3" ;;
  7) COVARIATES="x_D_1" ;;
  8) COVARIATES="x_D_1 x_D_6 x_D_7 x_D_8" ;;
  9) COVARIATES="x_D_1 x_D_6 x_D_7 x_D_8" ;;
esac

for N in 5000 20000; do
  FILE="input_real_2/scenario_${SCENARIO}/scenario_${SCENARIO}_${N}_1_data.csv"
  OUT_FILE="output/aroc_bnp_timing/scenario_${SCENARIO}_${N}_1_data_timing.csv"
  if [ -f "$OUT_FILE" ]; then
    echo "=== scenario_$SCENARIO, n=$N: $OUT_FILE already exists, skipping ==="
    continue
  fi
  echo "=== scenario_$SCENARIO, n=$N ==="
  command="$RSCRIPT R/aroc_bnp_timing.R $FILE $COVARIATES"
  echo $command
  $command
done

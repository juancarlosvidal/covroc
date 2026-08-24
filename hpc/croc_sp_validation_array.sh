#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
# Override with: RSCRIPT=/path/to/Rscript PYTHON=/path/to/python sbatch --array=1-9 croc_sp_validation_array.sh
RSCRIPT=${RSCRIPT:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/r-covroc/bin/Rscript}
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# One array task = one scenario, running both R/croc_sp_validation.R (real
# ROCnReg::cROC.sp) and src/baselines/croc_sp_validation.py (the Python port) at both
# sample sizes, so their outputs can be diffed to confirm the port is faithful (Reviewer
# 1, Major Concern 2 -- linear_reg_data_simulation.py's already-run 9-scenario results
# rely on this port, not the real R package). CPU-only, no GPU, closed-form OLS so this
# is fast (seconds per call) unlike hpc/r_bnp_timing_array.sh's MCMC-based AROC.bnp.
# Covariate lists mirror true_dgp._COVARIATE_COLUMNS in src/simulation/true_dgp.py --
# keep in sync if that ever changes. Skips either half independently if its output CSV
# already exists, same reproducibility idiom as hpc/r_bnp_timing_array.sh.
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
  NAME="scenario_${SCENARIO}_${N}_1_data"

  R_OUT="output/croc_sp_validation/${NAME}_auc_R.csv"
  if [ -f "$R_OUT" ]; then
    echo "=== scenario_$SCENARIO, n=$N, R: $R_OUT already exists, skipping ==="
  else
    echo "=== scenario_$SCENARIO, n=$N: R ==="
    command="$RSCRIPT R/croc_sp_validation.R $FILE $COVARIATES"
    echo $command
    $command
  fi

  PY_OUT="output/croc_sp_validation/${NAME}_auc_python.csv"
  if [ -f "$PY_OUT" ]; then
    echo "=== scenario_$SCENARIO, n=$N, Python: $PY_OUT already exists, skipping ==="
  else
    echo "=== scenario_$SCENARIO, n=$N: Python ==="
    command="$PYTHON src/baselines/croc_sp_validation.py $FILE $COVARIATES"
    echo $command
    $command
  fi
done

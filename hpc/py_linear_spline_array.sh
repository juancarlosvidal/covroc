#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Override with: PYTHON=/path/to/python sbatch --array=1-9 py_linear_spline_array.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# One array task = one scenario, re-running the linear and semiparametric-additive
# (spline) baselines with croc_linear_baseline.py's adaptive_direction=True fix
# (Reviewer 2, Major Comment 5 investigation -- ground truth comparison now uses the
# same adaptive a(x)/b(x) direction convention as the FNN/RF/ground_truth_auc.py, not
# the fixed direction ROCnReg::cROC.sp itself uses -- see
# src/simulation/linear_reg_data_simulation.py). CPU-only, no GPU. Cheap (well under 1s
# per replicate per timing_summary.csv), so a full 9-scenario x 2-sample-size
# x 100-replicate rerun takes minutes, not hours -- no array-index throttling needed.
SCENARIO=$SLURM_ARRAY_TASK_ID
INPUT_DIR=input_real_2/scenario_$SCENARIO

echo "=== scenario_$SCENARIO: linear ==="
command="$PYTHON src/simulation/linear_reg_data_simulation.py -i $INPUT_DIR -o output/linear/scenario_$SCENARIO --formula-type linear"
echo $command
$command

echo "=== scenario_$SCENARIO: spline ==="
command="$PYTHON src/simulation/linear_reg_data_simulation.py -i $INPUT_DIR -o output/spline/scenario_$SCENARIO --formula-type spline"
echo $command
$command

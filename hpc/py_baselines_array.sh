#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# Override with: PYTHON=/path/to/python sbatch --array=1-9 py_baselines_array.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# One array task = one scenario, running all three CPU-only baselines (Reviewer 1, Major
# Concern 2) one after another -- naive/linear/spline are each fast (sub-second per
# replicate in local testing), so splitting them into separate array tasks isn't worth
# the extra submitted-job count (the cluster enforces a QOSMaxSubmitJobPerUserLimit --
# see py_cov_array.sh/run_cov_array.sh for where that was hit). No --gres=gpu: none of
# these three use a neural network.
SCENARIO=$SLURM_ARRAY_TASK_ID
INPUT_DIR=input_real_2/scenario_$SCENARIO

echo "=== scenario_$SCENARIO: naive ==="
command="$PYTHON src/simulation/naive_roc_baseline.py -i $INPUT_DIR -o output/naive/scenario_$SCENARIO"
echo $command
$command

echo "=== scenario_$SCENARIO: linear ==="
command="$PYTHON src/simulation/linear_reg_data_simulation.py -i $INPUT_DIR -o output/linear/scenario_$SCENARIO --formula-type linear"
echo $command
$command

echo "=== scenario_$SCENARIO: spline ==="
command="$PYTHON src/simulation/linear_reg_data_simulation.py -i $INPUT_DIR -o output/spline/scenario_$SCENARIO --formula-type spline"
echo $command
$command

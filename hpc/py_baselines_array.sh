#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
# Override with: PYTHON=/path/to/python sbatch --array=1-9 py_baselines_array.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# One array task = one scenario, running all three CPU-only baselines (Reviewer 1, Major
# Concern 2) one after another -- naive/linear/spline are each fast (sub-second per
# replicate) for 8 of the 9 scenarios, which have a closed-form ground-truth ROC curve.
# Scenario 7 has no closed form (skew-normal/Student-t mixture healthy arm), so its
# ground truth needs a 20000-sample Monte Carlo estimate per subject
# (ground_truth_auc.true_roc_curve) -- ~100s for a single n=20000 replicate even after
# vectorizing that estimate (src/simulation/ground_truth_auc.py's
# _roc_from_samples_batch), so scenario 7's task alone is estimated at ~10-12h for all
# three baselines combined (100 replicates x 2 sample sizes x 3 scripts). --time=16:00:00
# gives that a safety margin; without an explicit --time, the partition default silently
# killed scenario 7's task before it produced almost any output while the other 8
# scenarios (seconds each) finished fine. If your partition caps --time below 16h,
# either request a QOS override or ask for this to be split into three separate
# per-baseline array jobs so each only needs ~1/3 of this budget.
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

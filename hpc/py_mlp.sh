#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=10:00:00
# Override with: PYTHON=/path/to/python sbatch py_mlp.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Scenario 7 has no closed-form ground-truth ROC curve, so evaluating against it costs a
# 20000-sample Monte Carlo estimate per subject on top of the usual FNN fit -- observed to
# leave ~18/200 of scenario 7's replicates unprocessed (timing_summary.csv Method=FNN,
# N=182 instead of 200) with no explicit --time, presumably hitting the partition default.
# The other 8 scenarios (closed-form ground truth) are unaffected either way.

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/simulation/mlp_reg_data_simulation_multi.py -i $1 -o $2 -c 6 -f 5 -e 800 -b 64"
echo $command
$command


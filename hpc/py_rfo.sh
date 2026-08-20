#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch py_rfo.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# No explicit --time: the cluster's partition has no default time limit, and RF already
# completed all 200/200 replicates for every scenario (including Scenario 7's Monte-Carlo-
# based ground truth) without one. An earlier --time=10:00:00 added here was pure
# precaution with no observed failure to justify it -- reverted after confirming (via
# py_mlp.sh's job 5520906 and `sacct`) that adding a --time cap here would only risk
# introducing a cutoff that didn't otherwise exist, for no benefit.

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/simulation/rf_reg_data_simulation_multi_2.py -i $1 -o $2 -c 6 -f 2 -e 800 -b 64"
echo $command
$command


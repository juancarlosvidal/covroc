#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=10:00:00
# Override with: PYTHON=/path/to/python sbatch py_rfo.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# RF already completed all 200/200 replicates for every scenario (including Scenario 7's
# Monte-Carlo-based ground truth, see py_mlp.sh) without an explicit --time -- this is
# just a safety margin, not a fix for an observed failure here.

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/simulation/rf_reg_data_simulation_multi_2.py -i $1 -o $2 -c 6 -f 2 -e 800 -b 64"
echo $command
$command


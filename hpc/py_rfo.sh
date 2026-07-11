#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch py_rfo.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniconda3/envs/diabetes/bin/python3.10}

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/simulation/rf_reg_data_simulation_multi_2.py -i $1 -o $2 -c 6 -f 2 -e 800 -b 64"
echo $command
$command


#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch py_gen.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/simulation/data_generation.py"
echo $command
$command


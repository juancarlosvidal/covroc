#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# Override with: PYTHON=/path/to/python sbatch py_linear.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Linear / semiparametric-additive aROC baseline -- statsmodels regression (cROC_sp), no
# neural network, no GPU needed.
# $3: --formula-type, "linear" or "spline" (default "linear")
FORMULA_TYPE=${3:-linear}

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/simulation/linear_reg_data_simulation.py -i $1 -o $2 --formula-type $FORMULA_TYPE"
echo $command
$command

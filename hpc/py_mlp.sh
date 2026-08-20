#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch py_mlp.sh
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# No explicit --time here: the cluster's partition has no default time limit, so this runs
# until all replicates in -i are done, however long that takes. (An earlier --time=10:00:00
# added here was based on an incorrect guess that a missing --time was why scenario_7 once
# stopped at 182/200 replicates -- confirmed wrong via `sacct` on a later job that DID carry
# --time=10:00:00: it hit TIMEOUT after only 31/200 replicates, i.e. this script would need
# ~65h for scenario_7's 200 replicates sequentially (~19.4 min/replicate observed). For
# scenario_7 specifically, use hpc/run_mlp_array.sh instead -- one SLURM array task per
# replicate, running concurrently across GPUs instead of one GPU grinding through all 200.
# The other 8 scenarios (closed-form ground truth, presumably much cheaper per replicate)
# have run fine sequentially through this script.

# Run from the repository root (SLURM's working directory is the submission directory)
command="$PYTHON src/simulation/mlp_reg_data_simulation_multi.py -i $1 -o $2 -c 6 -f 5 -e 800 -b 64"
echo $command
$command


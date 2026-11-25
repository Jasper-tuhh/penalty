#!/bin/bash -l
#SBATCH --job-name=hertz
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=12:00:00
#SBATCH --output=penalty_%x_%j.out
#SBATCH --error=penalty_%x_%j.err

module purge
module load anaconda/2023.07-1
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/fenicsx

mpirun -n "$SLURM_NTASKS" python hertz_penalty.py
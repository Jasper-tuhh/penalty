#!/bin/bash -l
#SBATCH --job-name=cbox_contact
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=12:00:00
#SBATCH --output=penalty_%x_%j.out
#SBATCH --error=penalty_%x_%j.err

cd /fibus/fs0/1f/cvu7589/penalty

module purge
module load anaconda/2023.07-1
eval "$(conda shell.bash hook)"
conda activate "$WORK/envs/fenicsx-contact"

mpirun -n "$SLURM_NTASKS" python cbox_penalty.py
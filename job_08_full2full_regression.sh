#!/bin/bash -l
#SBATCH --job-name=full2full_reg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Step 8: Full-corpus-to-full-corpus cross-corpus PHQ-8 regression.
# E1: All Proposed (all participants) → All E-DAIC (all 275)
# E2: All E-DAIC (all 275)            → All Proposed (all participants)
# Runs all 7 source configs.

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"

srun python "${SCRIPT_DIR}/08_full_to_full_regression.py"

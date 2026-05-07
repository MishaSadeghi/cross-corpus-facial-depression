#!/bin/bash -l
#SBATCH --job-name=full2full_cls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Step 9: Full-corpus-to-full-corpus cross-corpus classification.
# E1: All Proposed (all participants) -> All E-DAIC (all 275)
# E2: All E-DAIC (all 275)            -> All Proposed (all participants)
# Both label settings: SCID (Setting 1) and PHQ-8>=10 (Setting 2).

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"

srun python "${SCRIPT_DIR}/09_full_to_full_classification.py"

#!/bin/bash -l
#SBATCH --job-name=crosscorpus_video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Step 3: Cross-corpus PHQ-8 regression (EmpkinS ↔ E-DAIC).
# Source: emotion_induction_1 / ADK — best classification F1 (0.83) in
# a passive-observation phase that maps onto E-DAIC interview context.

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"

srun python "${SCRIPT_DIR}/03_cross_corpus_video_regression.py" \
    --phase emotion_induction_1 \
    --condition ADK \
    --experiments A B C

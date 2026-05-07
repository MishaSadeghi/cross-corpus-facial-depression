#!/bin/bash -l
#SBATCH --job-name=reg_phq8_%a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=%x.o%j

# Step 2: PHQ-8 regression holdout — all 4 phases (SLURM array job).
# Array index → phase mapping:
#   0 = latency
#   1 = emotion_induction_1
#   2 = negative_training
#   3 = positive_training

PHASES=("latency" "emotion_induction_1" "negative_training" "positive_training")
PHASE="${PHASES[$SLURM_ARRAY_TASK_ID]}"

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"

echo "Running PHQ-8 regression for phase: ${PHASE}"
srun python "${SCRIPT_DIR}/02_holdout_regression_phq8.py" --phase "${PHASE}"

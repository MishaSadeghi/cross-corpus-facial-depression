#!/bin/bash -l
#SBATCH --job-name=crosscorpus_reg_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Step 6: Cross-corpus PHQ-8 REGRESSION v2 (EmpkinS ↔ E-DAIC).
# Primary metric: CCC (Lin's Concordance Correlation Coefficient).
#
# Source configs — selected by within-corpus PHQ-8 CCC (882 restricted features):
#   LAT_ADK  — latency / ADK   (within-corpus CCC=0.66; passive observation, best PHQ-8)
#   LAT_SHAM — latency / SHAM  (within-corpus CCC=0.55; control condition, no intervention)
#   EI1_ADK  — emotion_induction_1 / ADK  (CCC=0.46; for comparison with classification)
#   ALL      — all 4 phases × all conditions pooled (largest training set baseline)

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"

srun python "${SCRIPT_DIR}/06_cross_corpus_regression_v2.py" \
    --configs LAT_ADK LAT_SHAM EI1_ADK ALL \
    --experiments A B C

#!/bin/bash -l
#SBATCH --job-name=crosscorpus_cls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Step 5: Cross-corpus binary depression CLASSIFICATION (EmpkinS ↔ E-DAIC).
#
# Source configs:
#   EI1_ADK   — emotion_induction_1 / ADK  (best within-corpus F1=0.83)
#   POS_ADK   — positive_training / ADK
#   POS_CRADK — positive_training / CRADK
#   ALL       — all 4 phases × all conditions pooled
#
# Experiments A / B / C (both cross-corpus directions + split-aware):
#   A : EmpkinS (ALL) → E-DAIC test
#   B : E-DAIC (train+dev) → EmpkinS test
#   C : C1 (EmpkinS-train → E-DAIC-test) + C2 (E-DAIC-train → EmpkinS-test)

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"

srun python "${SCRIPT_DIR}/05_cross_corpus_classification.py" \
    --configs EI1_ADK POS_ADK POS_CRADK ALL \
    --experiments A B C

#!/bin/bash -l
#SBATCH --job-name=cc_reg_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Cross-corpus PHQ-8 regression — 6 configs × experiments A B C D.
# Configs: LAT_ADK, LAT_SHAM (existing A/B/C results), POS_ADK (existing older run),
#          NEG_CRADK, NEG_CR, NEG_SHAM (all new — no prior results).
# Runs ALL experiments for all configs to get consistent timestamps.

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"

echo "====== CROSS-CORPUS PHQ-8 REGRESSION — 6 CONFIGS (LAT + BEST-TRAINING-PHASE) ======"
srun python "${SCRIPT_DIR}/06_cross_corpus_regression_v2.py" \
    --configs LAT_ADK LAT_SHAM LAT_ALL POS_ADK NEG_CRADK NEG_CR NEG_SHAM \
    --experiments A B C D

#!/bin/bash -l
#SBATCH --job-name=cc_cls_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Cross-corpus SCID classification — 6 configs × all experiments A B C D.
# Configs: LAT_ADK, LAT_SHAM, POS_ADK (POS_ADK has partial prior results),
#          NEG_CRADK, NEG_CR, NEG_SHAM (all new).
# Runs all experiments for clean consistent output.

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"

echo "====== CROSS-CORPUS SCID CLASSIFICATION — 6 CONFIGS (LAT + BEST-TRAINING-PHASE) ======"
srun python "${SCRIPT_DIR}/05_cross_corpus_classification.py" \
    --configs LAT_ADK LAT_SHAM LAT_ALL POS_ADK NEG_CRADK NEG_CR NEG_SHAM \
    --experiments A B C D

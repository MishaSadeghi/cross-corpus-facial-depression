#!/bin/bash -l
#SBATCH --job-name=cc_cls_phq8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Cross-corpus PHQ-8>=10 classification — 7 configs × all experiments A B C D.
# Both EmpkinS (PHQ-8>=10 threshold) and E-DAIC (depressed column = PHQ-8>=10).
# All results are new — no prior results exist.

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper"

echo "====== CROSS-CORPUS PHQ-8>=10 CLASSIFICATION — 6 CONFIGS ======"
srun python "${SCRIPT_DIR}/07_cross_corpus_classification_phq8.py" \
    --configs LAT_ADK LAT_SHAM LAT_ALL POS_ADK NEG_CRADK NEG_CR NEG_SHAM \
    --experiments A B C D

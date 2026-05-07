#!/bin/bash -l
#SBATCH --job-name=ncv_allcond_strat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Within-corpus nested CV — ALL_CONDITIONS only, with condition+SCID stratified folds.
# Fixes the fold imbalance issue for the pooled condition analysis.

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper/within_corpus_regression"
ALL_TARGETS="phq_8 phq_9 ads.1 HRSD_6.1 HRSD_17.1 HRSD_21.1 HRSD_24.1"

echo "====== NESTED CV — ALL_CONDITIONS with condition+SCID stratified folds ======"
for PHASE in latency emotion_induction_1 negative_training positive_training; do
    echo "--- Phase: $PHASE ---"
    srun python "${SCRIPT_DIR}/nested_cv_phq8_regression.py" \
        --phase "$PHASE" \
        --conditions ALL_CONDITIONS \
        --targets $ALL_TARGETS \
        --restrict-to-cross-corpus
done

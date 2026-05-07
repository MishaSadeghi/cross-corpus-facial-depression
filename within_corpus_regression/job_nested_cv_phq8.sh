#!/bin/bash -l
#SBATCH --job-name=nestedcv_phq8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x.o%j

# Within-corpus multi-scale nested CV (5×3) regression on EmpkinS-EKSpression.
# Runs all 4 phases × 5 conditions × 7 targets with full features.
# Then re-runs with --restrict-to-cross-corpus (882-feature E-DAIC subset).
#
# Targets: phq_8 phq_9 ads.1 HRSD_6.1 HRSD_17.1 HRSD_21.1 HRSD_24.1
#
# Note: 48h wall time because 7 targets × 5 conditions × 4 phases × 5 outer folds
#       × 3 scalers × 12 models × GridSearch is substantial.

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper/within_corpus_regression"
ALL_TARGETS="phq_8 phq_9 ads.1 HRSD_6.1 HRSD_17.1 HRSD_21.1 HRSD_24.1"

echo "====== NESTED CV REGRESSION — FULL FEATURES ======"
for PHASE in latency emotion_induction_1 negative_training positive_training; do
    echo "--- Phase: $PHASE ---"
    srun python "${SCRIPT_DIR}/nested_cv_phq8_regression.py" \
        --phase "$PHASE" \
        --conditions ADK CR CRADK SHAM ALL_CONDITIONS \
        --targets $ALL_TARGETS
done

echo ""
echo "====== NESTED CV REGRESSION — CROSS-CORPUS FEATURES (882) ======"
for PHASE in latency emotion_induction_1 negative_training positive_training; do
    echo "--- Phase: $PHASE (restricted) ---"
    srun python "${SCRIPT_DIR}/nested_cv_phq8_regression.py" \
        --phase "$PHASE" \
        --conditions ADK CR CRADK SHAM ALL_CONDITIONS \
        --targets $ALL_TARGETS \
        --restrict-to-cross-corpus
done

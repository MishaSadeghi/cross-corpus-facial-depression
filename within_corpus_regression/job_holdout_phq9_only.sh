#!/bin/bash -l
#SBATCH --job-name=holdout_phq9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=%x.o%j

# PHQ-9 holdout regression only — re-run after pd.to_numeric fix in shared pipeline.
# All other 6 scales already completed in holdout_phq8.o11487046.

source /home/woody/empk/empk004h/software/private/myenv/bin/activate
SCRIPT_DIR="/home/hpc/empk/empk004h/depression-detection/KDD_paper/within_corpus_regression"

echo "====== HOLDOUT — PHQ-9 — FULL FEATURES ======"
for PHASE in latency emotion_induction_1 negative_training positive_training; do
    echo "--- Phase: $PHASE ---"
    srun python "${SCRIPT_DIR}/holdout_phq8_regression.py" \
        --phase "$PHASE" \
        --conditions ADK CR CRADK SHAM ALL_CONDITIONS \
        --targets phq_9
done

echo ""
echo "====== HOLDOUT — PHQ-9 — CROSS-CORPUS FEATURES (882) ======"
for PHASE in latency emotion_induction_1 negative_training positive_training; do
    echo "--- Phase: $PHASE (restricted) ---"
    srun python "${SCRIPT_DIR}/holdout_phq8_regression.py" \
        --phase "$PHASE" \
        --conditions ADK CR CRADK SHAM ALL_CONDITIONS \
        --targets phq_9 \
        --restrict-to-cross-corpus
done

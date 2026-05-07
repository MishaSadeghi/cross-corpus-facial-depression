"""
config.py
=========
Set the paths below to match your local data layout before running any scripts.

ProposedCorpus refers to the structured RCT corpus used in this study.
InterviewCorpus refers to the E-DAIC interview corpus.
"""

# ── ProposedCorpus data paths ──────────────────────────────────────────────

# CSV with one row per participant (contains PHQ-8, condition, split columns)
MASTER_DATA_PATH = "/path/to/proposed_corpus/participant_master_data.csv"

# Directory containing shared_train_ids_*.csv / shared_test_ids_*.csv split files
PROPOSED_SPLITS_DIR = "/path/to/proposed_corpus/splits/"

# Pre-aggregated OpenFace feature CSVs per phase
PROPOSED_PHASE_FEATURES = {
    "latency": "/path/to/proposed_corpus/features/latency.csv",
    "emotion_induction_1": "/path/to/proposed_corpus/features/emotion_induction.csv",
    "negative_training": "/path/to/proposed_corpus/features/negative_training.csv",
    "positive_training": "/path/to/proposed_corpus/features/positive_training.csv",
}

# ── InterviewCorpus (E-DAIC) data paths ────────────────────────────────────

# CSV with columns: Participant_ID, PHQ_score, Split (train/dev/test)
EDAIC_LABELS_PATH = "/path/to/edaic/edaic_labels.csv"

# Directory containing per-participant OpenFace raw feature files
EDAIC_OPENFACE_ROOT = "/path/to/edaic/openface_extracted/"

# Pre-aggregated E-DAIC features (produced by 01_aggregate_edaic_video_features.py)
EDAIC_AGGREGATED_PATH = "/path/to/edaic/edaic_openface_aggregated.csv"

# ── Output directories ─────────────────────────────────────────────────────

# Root directory for all saved results and models
RESULTS_ROOT = "/path/to/output/results/"

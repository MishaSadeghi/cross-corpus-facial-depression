# Cross-Corpus Facial Biomarkers of Depression

Code for the paper:
> **Do Depressive Facial Patterns Transfer Across Cultures and Contexts? Evidence from a German RCT and E-DAIC**

> Accepted at Workshop on AI for Cognitive and Mental Health Support (AI4Mental) @ KDD 2026.

---

## Overview

This repository contains the analysis pipeline for a bidirectional cross-corpus study of
facial depression assessment. We pair a structured randomised controlled trial (RCT)
corpus with a semi-structured interview corpus (E-DAIC) and evaluate
continuous PHQ-8 severity regression and binary MDD classification in both transfer directions.

---

## Requirements

```bash
pip install scikit-learn xgboost lightgbm catboost scipy numpy pandas joblib shap matplotlib seaborn
```

Python 3.8+. LightGBM and CatBoost are optional (skipped if not installed).

---

## Setup

Edit `config.py` and set the paths to your local copies of the two corpora and the
desired output directory. All scripts import from this file — no other path changes needed.

```python
# config.py
MASTER_DATA_PATH    = "/path/to/proposed_corpus/participant_master_data.csv"
PROPOSED_SPLITS_DIR = "/path/to/proposed_corpus/splits/"
PROPOSED_PHASE_FEATURES = {
    "latency":            "/path/to/proposed_corpus/features/latency.csv",
    "emotion_induction_1":"/path/to/proposed_corpus/features/emotion_induction.csv",
    "negative_training":  "/path/to/proposed_corpus/features/negative_training.csv",
    "positive_training":  "/path/to/proposed_corpus/features/positive_training.csv",
}
EDAIC_LABELS_PATH    = "/path/to/edaic/edaic_labels.csv"
EDAIC_OPENFACE_ROOT  = "/path/to/edaic/openface_extracted/"
EDAIC_AGGREGATED_PATH = "/path/to/edaic/edaic_openface_aggregated.csv"
RESULTS_ROOT         = "/path/to/output/"
```

---

## Data

**ProposedCorpus** — structured RCT with N=256 participants (128 MDD, 128 HC),
SCID-5-CV diagnosis, 4 intervention conditions × 4 experimental phases.
Features are pre-aggregated per phase/condition. PHQ-8 was used as the main regression target.

**E-DAIC** — [Extended Distress Analysis Interview Corpus] (https://dcapswoz.ict.usc.edu/), N=275 participants,
semi-structured interview. PHQ-8 ≥ 10 as depression threshold.
Official train+dev (n=219) / test (n=56) split used throughout.

---

## Scripts

Run in order:

| Script | Description |
|--------|-------------|
| `aggregate_edaic_video_features.py` | Aggregate E-DAIC raw OpenFace frame-level CSVs → one row per participant |
| `within_corpus_regression/holdout_phq8_regression.py` | Within-corpus multi-scale regression on ProposedCorpus (fixed holdout, all regression targets) |
| `within_corpus_regression/nested_cv_phq8_regression.py` | Within-corpus multi-scale regression (5-fold nested CV, all regression targets) |
| `holdout_regression_phq8.py` | Within-corpus PHQ-8 regression — extended version with cross-corpus feature restriction |
| `cross_corpus_classification.py` | Cross-corpus binary classification (SCID labels, Setting 1) |
| `cross_corpus_regression.py` | Cross-corpus regression with within-corpus baselines |
| `cross_corpus_classification_phq8.py` | Cross-corpus classification (PHQ-8-based labels, Setting 2) |
| `full_to_full_regression.py` | Full-corpus regression without held-out split |
| `full_to_full_classification.py` | Full-corpus classification without held-out split |
| `shap_cross_corpus_cls.py` | SHAP beeswarm plots for the two best cross-corpus classifiers |
| `result_visualizations.py` | AUC heatmap, domain-gap slope chart, Setting 1 vs 2 scatter |

### Shared library

`shared_regression_pipeline.py` — imported by all scripts. Contains data loaders,
feature selection, the full regressor/classifier grid, and evaluation utilities.

---

## Regression Targets

The within-corpus regression scripts (`holdout_phq8_regression.py`,
`nested_cv_phq8_regression.py`) evaluate all depression severity scales available
in ProposedCorpus:

| Column | Scale | Administration |
|--------|-------|----------------|
| `phq_8` | PHQ-8 | Remote self-report (before lab visit) |
| `phq_9` | PHQ-9 | In-lab self-report |
| `ads.1` | CES-D / ADS | In-lab clinician-administered |
| `HRSD_17.1` | HRSD-17 | In-lab clinician-administered |

PHQ-8 is the only scale shared with E-DAIC and is therefore used as the cross-corpus
regression target. The paper reports within-corpus results for PHQ-8, PHQ-9, CES-D,
and HRSD-17. The scripts also support HRSD-6, HRSD-21, and HRSD-24 via `--targets`
but these are not reported in the paper. Targets can be selected via:

```bash
python within_corpus_regression/nested_cv_phq8_regression.py \
    --phase latency --targets phq_8 phq_9 ads.1 HRSD_17.1
```

---

## Experimental Design

**Seven source configurations** (phase × condition combinations from ProposedCorpus):
`Prep./AFE`, `Prep./SHAM`, `Prep./All`, `Pos./AFE`, `Neg./CR+AFE`, `Neg./CR`, `Neg./SHAM`

**Cross-corpus experiments:**
- **Exp A** — ProposedCorpus ALL → E-DAIC test (maximum training data)
- **Exp B** — E-DAIC train+dev → ProposedCorpus test
- **Exp C1** — ProposedCorpus train (80%) → E-DAIC test
- **Exp D1/D2** — Within-corpus baselines

**Two label settings** (classification only):
- Setting 1: SCID-5-CV labels for ProposedCorpus, PHQ-8 ≥ 10 for E-DAIC
- Setting 2: PHQ-8 ≥ 10 for both corpora

---

## Feature List

The 882 shared features used in all cross-corpus experiments are listed in
`cross_corpus_882_features.txt`.

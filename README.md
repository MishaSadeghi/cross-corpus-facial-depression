# KDD AI4Mental 2026 — Cross-Corpus Facial Biomarkers of Depression

**Workshop**: AI for Cognitive and Mental Health Support (AI4Mental) @ KDD 2026
**Date**: August 9, 2026, Jeju, South Korea
**Deadline**: April 30, 2026
**Page limit**: 4–6 pages
**Submission pillar**: "AI as Assessment"

---

## 1. Research Question

Can context-specific facial biomarkers of depression — elicited during structured
emotion regulation tasks in a clinical RCT — generalize to an independent
clinical interview dataset?

This paper extends the journal work (IEEE Transactions on Affective Computing,
in preparation) with a cross-corpus validation that has not been published before.

---

## 2. Datasets

### 2a. EmpkinS-EKSpression RCT (source / training corpus)

| Property | Value |
|---|---|
| N | 256 (128 MDD, 128 Healthy Controls) |
| Diagnosis | SCID-5-CV gold standard |
| Design | 4 intervention conditions × 4 experimental phases |
| Video | Sony camera, OpenFace 2.1.0 |
| Features | AUs (intensity + presence), gaze angles, head pose rotations |
| PHQ-8 | Available for 255/256 participants |

**Conditions**: ADK (AFE only), CR (Cognitive Restructuring only),
CRADK (CR + AFE), SHAM (control)

**Phases**:
1. Latency / Preparatory (passive baseline)
2. Mood Induction (emotion activation)
3. Negative Training (negative emotion regulation)
4. Positive Training (positive emotion regulation)

**Key classification result** (from journal paper):
- Passive phases: F1 ≈ 0.56–0.61 (near chance)
- CRADK + Positive Training: F1 = 0.77 (nested CV), 0.86 (holdout)
- Best features: AU14 erratic peaks + low variability speed (psychomotor
  dysregulation signature), gaze entropy restriction, head pitch variability

**Pre-aggregated feature CSVs** (used directly by scripts):
```
Latency:
  /home/woody/empk/empk004h/D02_dataset/depression_face_analysis/
  processed_data_stats/sony_videos_final/latency_sony_final.csv

Emotion Induction:
  .../emotion_induction_stats_final/emotion_induction_1_orig_without_norm_augmented.csv

Negative Training:
  .../training_original_stats/negative_training_orig_without_norm.csv

Positive Training:
  .../training_original_stats/positive_training_orig_without_norm.csv
```

**Master data** (PHQ-8, SCID labels, demographics):
```
/home/hpc/empk/empk004h/depression-detection/d02_data/data_info/participant_master_data.csv
  Columns: ID, gender, age, condition, label, phq_9, phq_8, ads.1, HRSD_17.1, ...
```

**Shared train/test splits** (pre-computed, 80/20 stratified, same across all phases):
```
/home/hpc/empk/empk004h/depression-detection/affective_paper_final_revisions/
  classical_ML_fixed/scripts/shared_train_ids_{ADK|CR|CRADK|SHAM|ALL}.csv
                              shared_test_ids_{ADK|CR|CRADK|SHAM|ALL}.csv
```

---

### 2b. E-DAIC (Extended Distress Analysis Interview Corpus)

| Property | Value |
|---|---|
| N | 275 (66 depressed PHQ-8≥10, 209 healthy) |
| Diagnosis | PHQ-8 ≥ 10 threshold |
| Context | Unstructured clinical interview (~20 min, Ellie virtual agent) |
| Video | OpenFace 2.1.0 (same version as EmpkinS) |
| Split | Official: 163 train / 56 dev / 56 test |

**Label file**:
```
/home/hpc/empk/empk004h/depression-detection/d02_data/Locs_project/
  Interspeech/cross_corpus_analysis/E-DAIC_data/edaic_labels.csv
  Columns: participant_id, depressed, PHQ_score, split, gender, age
```

**Raw OpenFace frame-level CSVs**:
```
/home/vault/empkins/tpD/D02/processed_data/DAIC_datasets/E-DAIC/
  extracted/{ID}_P/features/{ID}_OpenFace2.1.0_Pose_gaze_AUs.csv
  Columns: frame, timestamp, confidence, success,
           pose_Tx/Ty/Tz, pose_Rx/Ry/Rz,
           gaze_0_x/y/z, gaze_1_x/y/z, gaze_angle_x/y,
           AU01_r ... AU45_r, AU01_c ... AU45_c
```

**Aggregated features** (produced by script 01, stored on woody):
```
/home/woody/empk/empk004h/D02_dataset/depression_face_analysis/
  processed_data_stats/edaic_openface_aggregated.csv
```

---

## 3. Feature Alignment

Both datasets use **OpenFace 2.1.0** with identical raw output columns.
EmpkinS features are already aggregated using 18 statistical functionals per column.
Script `01` applies the same functionals to E-DAIC frame-level data.

**18 functionals per signal column**:
| Group | Functionals |
|---|---|
| Original signal | mean, std, min, max, skew, kurt, range, entropy |
| Frame differences (deriv) | mean, std, min, max, skew, kurt, range, entropy |
| Dynamic | rate_of_change, peaks_count |

**Column naming**: `{signal}__{functional}` — e.g., `AU14_r__std_deriv`, `gaze_angle_x__entropy_orig`

After aggregation, both datasets have the same column naming convention.
The cross-corpus scripts automatically compute the column intersection.

---

## 4. Scripts

### Execution order

```
Step 1 (run once):   01_aggregate_edaic_video_features.py
Step 2 (all phases): 02_holdout_regression_phq8.py
Step 3 (after step 2, pick best phase/condition): 03_cross_corpus_video_regression.py
```

---

### `shared_regression_pipeline.py`
Shared library imported by all other scripts. Contains:
- All path constants
- `load_empkins_data(phase, condition)` — loads pre-aggregated EmpkinS CSV + PHQ-8
- `load_edaic_data(split_filter)` — loads E-DAIC aggregated CSV
- `load_shared_split(tag)` — reads pre-computed EmpkinS train/test split files
- `aggregate_openface_frames(df)` — 18 functionals applied to frame-level OpenFace data
- `spearman_select(X, y, top_n=50)` — feature selection by |Spearman r| with target
- `pearson_redundancy_prune(X, threshold=0.90)` — remove highly correlated features
- `get_regressors()` — Ridge, Lasso, ElasticNet, SVR, KNN, RF, ET, GB, XGB, MLP, LGBM, CatBoost
- `get_param_grids()` — hyperparameter grids for GridSearchCV
- `evaluate_regressor(model, X_test, y_test)` — MAE, RMSE, R², Pearson r, Spearman ρ
- `align_features(X_source, X_target)` — column intersection for cross-corpus

---

### `01_aggregate_edaic_video_features.py`
**Purpose**: Convert E-DAIC raw OpenFace frame-level CSVs → one row per participant.

**What it does**:
1. Reads each participant's `{ID}_OpenFace2.1.0_Pose_gaze_AUs.csv`
2. Filters frames with `confidence < 0.5` (OpenFace quality filter)
3. Skips participants with fewer than 50 high-confidence frames
4. Applies 18 statistical functionals → one row per participant
5. Merges PHQ-8 labels and train/dev/test split
6. Saves to `edaic_openface_aggregated.csv` on woody

**Runtime**: ~20–30 min (275 participants × ~15,000 frames each)

**Usage**:
```bash
python 01_aggregate_edaic_video_features.py
python 01_aggregate_edaic_video_features.py --min_confidence 0.5 --min_frames 100
sbatch job_01_aggregate_edaic.sh
```

---

### `02_holdout_regression_phq8.py`
**Purpose**: Predict continuous PHQ-8 scores from EmpkinS video features, one phase at a time.
Identifies the best phase × condition combination for the cross-corpus experiment.

**Pipeline** (no data leakage):
1. Load pre-aggregated EmpkinS features for specified phase
2. Filter to shared train/test split participants
3. Imputer fit on train, transform test
4. Scaler fit on train, transform test (3 scalers × 10+ models)
5. Feature selection: Spearman |r| top 50 on train only
6. Redundancy pruning: Pearson r ≥ 0.90 on train only
7. GridSearchCV 5-fold on train (neg MAE)
8. Evaluate on test: MAE, RMSE, R², Pearson r, Spearman ρ
9. Save results CSV + best model .joblib per phase/condition

**Output location**:
```
/home/woody/empk/empk004h/D02_dataset/D02_final_results_and_models/
  KDD_paper/regression_phq8/{phase}/
    regression_{phase}_{condition}_{timestamp}.csv
    best_model_{phase}_{condition}_{timestamp}.joblib
```

**Usage**:
```bash
# Single phase
python 02_holdout_regression_phq8.py --phase positive_training
python 02_holdout_regression_phq8.py --phase latency --conditions ADK CRADK

# All 4 phases as SLURM array (recommended)
sbatch --array=0-3 job_02_regression_all_phases.sh
```

**Expected results** (hypothesis based on classification findings):
- Latency / Mood Induction: weak regression (R² ≈ 0, Pearson r ≈ 0.1–0.2)
- Negative Training: moderate (Pearson r ≈ 0.2–0.4)
- Positive Training CRADK: best (Pearson r ≈ 0.4–0.6)

---

### `03_cross_corpus_video_regression.py`
**Purpose**: Test whether EmpkinS-trained regression models generalize to E-DAIC and vice versa.

**Three experiments**:

| Exp | Train | Test | Scientific question |
|---|---|---|---|
| A | EmpkinS ALL | E-DAIC test | Do RCT task biomarkers transfer to clinical interview? |
| B | E-DAIC train+dev | EmpkinS test | Does interview-trained model work on RCT phases? |
| C1 | EmpkinS train (80%) | E-DAIC test | Controlled transfer (official splits both sides) |
| C2 | E-DAIC train+dev | EmpkinS test (20%) | Symmetric to C1 |

**Pipeline** (same no-leakage design):
1. Load both datasets, align to common feature columns
2. Imputer fit on source (train), transform target (test)
3. Scaler fit on source, transform target
4. Spearman + Pearson feature selection on source only
5. GridSearchCV on source
6. Evaluate on target

**Output location**:
```
/home/woody/empk/empk004h/D02_dataset/D02_final_results_and_models/
  KDD_paper/cross_corpus_regression/{phase}_{condition}/
    crosscorpus_A_emp2daic_{phase}_{condition}_{timestamp}.csv
    crosscorpus_B_daic2emp_{phase}_{condition}_{timestamp}.csv
    crosscorpus_C1_...csv  /  crosscorpus_C2_...csv
    SUMMARY_{phase}_{condition}_{timestamp}.csv   ← main results table
    best_model_{experiment}_{timestamp}.joblib
```

**Usage**:
```bash
# Default (positive_training / CRADK — change after reviewing step 2 results)
python 03_cross_corpus_video_regression.py

# Custom phase/condition
python 03_cross_corpus_video_regression.py --phase positive_training --condition CRADK

# Only specific experiments
python 03_cross_corpus_video_regression.py --experiments A C

sbatch job_03_cross_corpus.sh
```

---

## 5. Workshop Paper Structure (4–6 pages)

**Title idea**: *"Context-Dependent Facial Biomarkers of Depression: From Clinical RCT to Cross-Corpus Generalization"*

| Section | Content | Source |
|---|---|---|
| Introduction | Context-dependency hypothesis, AI4Mental relevance | New writing |
| Dataset & Methods | EmpkinS RCT (brief), E-DAIC, OpenFace, ML pipeline | Journal paper §II–III condensed |
| Within-corpus classification | Phase × condition F1/AUC heatmap (passive→active delta) | Journal paper Table IV |
| PHQ-8 regression (within-corpus) | Best phase/condition MAE + Pearson r | Script 02 results |
| Cross-corpus regression | Experiment A/B/C summary table | Script 03 results |
| Interpretability | AU14 paradox, gaze entropy (condensed) | Journal paper §V |
| Conclusion | Context matters; partial transfer shows promise | New writing |

**What to leave out** (keep for journal):
- Full regression tables (PHQ-9, CES-D, HRSD-17)
- Deep learning comparisons (SEG, SPG, AVEC2019, OpenTSLM)
- All DL architecture details and hyperparameter tables
- Full SHAP beeswarm plots per condition
- Clinical treatment implications

---

## 6. Narrative / Story for Paper

1. **Problem**: Passive observation (resting state, neutral interview) yields near-chance
   depression detection from facial cues. This explains mixed results in prior work.

2. **Finding**: Structured emotion regulation tasks in an RCT dramatically unlock
   discriminative facial biomarkers (F1 jumps from 0.56 to 0.86).

3. **Key biomarker**: AU14 (buccinator) shows erratic intensity peaks combined with
   rigid low-variability dynamics — a psychomotor dysregulation signature.
   Gaze entropy is restricted; head pitch variability is reduced.

4. **New contribution for KDD paper**: Cross-corpus validation — do these context-specific
   biomarkers transfer to clinical interview (E-DAIC)?
   Experiment A tests EmpkinS → E-DAIC; Experiment B tests the reverse direction.
   The degree of transfer quantifies how context-specific the features are.

5. **Takeaway for AI4Mental**: Depression detection systems should use structured
   affective tasks, not passive observation. Context is a design variable, not a nuisance.

---

## 7. Key Challenges for Paper Discussion

| Challenge | Detail |
|---|---|
| Label mismatch | SCID-5-CV (clinical gold standard) vs PHQ-8 ≥ 10 (self-report threshold) |
| Context mismatch | Structured task (4 defined phases) vs unstructured interview (20 min) |
| Class imbalance | E-DAIC: 66 dep / 209 HC (24% positive) vs EmpkinS: 50/50 |
| Language/culture | EmpkinS = German participants; E-DAIC = English-speaking US |
| Sample size | E-DAIC test = 56 participants; limits statistical power |

---

## 8. Metrics to Report

**Regression** (PHQ-8 prediction):
- MAE (primary) — directly interpretable in PHQ-8 units (0–24 scale)
- RMSE
- Pearson r (effect size, standard in clinical prediction literature)
- R²

**Classification** (from journal paper, within-corpus):
- F1-score (binary, macro)
- AUC-ROC
- Sensitivity / Specificity

---

## 9. Environment

```bash
# Activate environment
source /home/woody/empk/empk004h/software/private/myenv/bin/activate

# Check dependencies
python3 -c "import sklearn, xgboost, scipy, shap; print('OK')"
```

Required packages: `scikit-learn`, `xgboost`, `scipy`, `numpy`, `pandas`,
`joblib`, `lightgbm` (optional), `catboost` (optional)

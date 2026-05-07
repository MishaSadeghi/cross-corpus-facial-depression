# KDD AI4Mental 2026 — Paper Outline with Results

**Title**: *"Context-Dependent Facial Biomarkers of Depression: From Clinical RCT to Cross-Corpus Generalization"*
**Workshop**: AI for Cognitive and Mental Health Support (AI4Mental) @ KDD 2026
**Deadline**: April 30, 2026 | **Page limit**: 4–6 pages

---

## 1. Introduction

**Core argument**: Passive facial observation yields near-chance depression detection.
Structured emotion regulation tasks in a clinical RCT dramatically unlock discriminative
facial biomarkers. The key question for this KDD paper: do these context-specific
biomarkers *generalize* to an independent clinical interview dataset (E-DAIC)?

**Context-dependency hypothesis**: Depression-related psychomotor dysregulation is more
visible under affective challenge. This explains why prior passive-observation depression
detection studies report mixed results.

**Contribution for AI4Mental audience**: Context is a *design variable* for mental health
AI systems, not a nuisance to control. Systems should use structured affective tasks.

---

## 2. Datasets

### 2.1 EmpkinS-EKSpression RCT (source corpus)

| Property | Value |
|---|---|
| N | 256 (128 MDD, 128 Healthy Controls) |
| Diagnosis | SCID-5-CV (gold standard, trained psychologist) |
| Design | 4 conditions × 4 phases |
| Video | Sony camera, OpenFace 2.1.0 |
| PHQ-8 | Available for 255/256; collected remotely (at home, before lab visit) |
| Other scales | PHQ-9, CES-D, HRSD-6/17/21/24 (in-lab, clinically supervised) |

**Conditions**: ADK (AFE only), CR (Cognitive Restructuring), CRADK (CR+AFE), SHAM (control)

**Phases**: (1) Latency/baseline, (2) Mood Induction, (3) Negative Training, (4) Positive Training

### 2.2 E-DAIC (Extended Distress Analysis Interview Corpus)

| Property | Value |
|---|---|
| N | 275 (66 depressed [PHQ-8≥10], 209 healthy; 24% positive rate) |
| Diagnosis | PHQ-8 ≥ 10 (self-report threshold — note: weaker label than SCID) |
| Context | Unstructured clinical interview ~20 min (Ellie virtual agent) |
| Video | OpenFace 2.1.0 (same version as EmpkinS) |
| Split | Official: 163 train / 56 dev / 56 test |

### 2.3 Feature Alignment

Both datasets use OpenFace 2.1.0 with identical raw output.
**882 shared features**: AUs (intensity + presence), gaze angles, head pose — 18 statistical
functionals (mean, std, min, max, skew, kurt, range, entropy × original and first-difference
signals; plus rate_of_change and peaks_count).

**Key finding on features**: Restricting to these 882 features (vs. full EmpkinS feature set)
substantially improves within-corpus PHQ-8 prediction — likely because it removes noisy
landmark/composite features and implicitly acts as regularization (see §4 below).

---

## 3. Within-Corpus Classification (from journal paper — EmpkinS only)

Reported for context; key numbers from SCID-binary classification:

| Phase | Condition | F1 | AUC |
|---|---|---|---|
| Latency (baseline) | any | 0.56–0.61 | ~0.55 |
| Positive Training | CRADK | **0.77** (nested CV) / **0.86** (holdout) | — |

**Takeaway**: Passive baseline = near chance. Structured task (CRADK, Positive Training) = large jump.
This phase × condition interaction is the core within-corpus finding of the journal paper.

---

## 4. Within-Corpus Regression — PHQ-8 (EmpkinS, new for KDD)

### 4.1 Label reliability note (important for interpretation)

PHQ-8 was collected **remotely at home**, before the lab visit, without clinical supervision.
This makes it the *least reliable* label in the dataset. Ordering by reliability:
**SCID > HRSD/CES-D > PHQ-8**.
This explains why PHQ-8 regression is harder than HRSD regression and why cross-corpus
PHQ-8 failure is unsurprising.

### 4.2 Repeated holdout (full features, PHQ-8, 5 random seeds)

| Phase | Condition | Best model | MAE (mean±std) | Pearson r (mean±std) |
|---|---|---|---|---|
| Latency | ADK | LGBM/Robust | 4.65 ± 0.75 | 0.27 ± 0.10 |
| Latency | CR | CatBoost/Robust | 4.88 ± 0.68 | 0.03 ± 0.30 |
| Latency | CRADK | LGBM/Standard | 5.57 ± 0.39 | −0.14 ± 0.25 |
| Latency | SHAM | ElasticNet/Standard | 5.43 ± 0.94 | −0.07 ± 0.29 |

### 4.3 Single holdout — restricted 882 features (PHQ-8)

From `within_corpus_regression_findings.md` (top-3 configs per scale):

| Phase | Condition | CCC | MAE | Pearson r | n_test |
|---|---|---|---|---|---|
| Latency | ADK | **0.665** | 3.94 | 0.665 | 12 |
| Latency | CRADK | 0.661 | 4.47 | 0.674 | 11 |
| Positive Training | CR | 0.626 | 3.43 | 0.678 | 12 |

> **Caveat**: n_test = 11–13 for single-condition holdout; results can be inflated.
> Nested CV (restricted features, ALL_CONDITIONS, n=47) gives CCC = 0.065 — much more conservative.
> **Paper should report nested CV as primary, holdout as supplementary.**

### 4.4 Within-corpus results for other scales (best across phases, restricted features)

| Scale | Collection | Best phase/condition | CCC | Pearson r |
|---|---|---|---|---|
| HRSD-6 | In-lab, clinician | neg_training/CR | **0.736** | 0.792 |
| HRSD-17 | In-lab, clinician | latency/CR | 0.632 | 0.740 |
| HRSD-21 | In-lab, clinician | neg_training/CR | 0.572 | 0.706 |
| HRSD-24 | In-lab, clinician | latency/CR | 0.569 | 0.667 |
| CES-D | In-lab, supervised | pos_training/SHAM | 0.582 | 0.582 |
| PHQ-8 | Remote, unsupervised | latency/ADK | 0.665 | 0.665 |

**Pattern**: CR condition dominates for HRSD (clinician-rated), while ADK/CRADK dominate
for PHQ-8. Different phases capture *severity* (passive baseline) vs *diagnosis-relevant
dynamics* (active phases). This discrepancy between classification-optimal and
regression-optimal phases is itself a finding worth noting.

---

## 5. Cross-Corpus Regression (EmpkinS ↔ E-DAIC, new for KDD)

Four experiments (A/B/C1/C2) × four configs (LAT_ADK, LAT_SHAM, EI1_ADK, ALL).
All use the 882 restricted features.

### 5.1 Config selection rationale

| Tag | Phase | Condition | Within-corpus PHQ-8 CCC | Rationale |
|---|---|---|---|---|
| LAT_ADK | Latency | ADK | 0.665 | Best PHQ-8 single holdout; passive, AFE condition |
| LAT_SHAM | Latency | SHAM | 0.554 | Control condition; no intervention |
| EI1_ADK | Emotion Induction | ADK | 0.457 | Cross-method bridge to classification |
| ALL | All phases | All conditions | ~0.07 (nested CV) | Largest training set; upper bound |

### 5.2 Cross-corpus regression results (PHQ-8 prediction, CCC primary metric)

| Config | Exp A: EmpkinS→E-DAIC | Exp B: E-DAIC→EmpkinS | Exp C1: EmpkinS-train→E-DAIC | Exp C2: E-DAIC-train→EmpkinS |
|---|---|---|---|---|
| **LAT_ADK** | CCC=0.015, r=0.037 | CCC=0.135, r=0.428 | CCC=0.006, r=0.053 | CCC=0.135, r=0.428 |
| **LAT_SHAM** | CCC=0.019, r=0.032 | CCC=0.256, r=0.449 | CCC=0.106, r=0.135 | CCC=0.256, r=0.449 |
| **EI1_ADK** | CCC=0.025, r=0.050 | CCC=0.028, r=0.109 | CCC=0.016, r=0.026 | CCC=0.028, r=0.109 |
| **ALL** | CCC=0.092, r=0.148 | CCC=0.052, r=0.138 | CCC=0.109, r=0.128 | CCC=0.052, r=0.138 |

Best models: XGB, KNN, LGBM, Lasso depending on config.

**Best MAE in cross-corpus**: LAT_SHAM Exp B: MAE=5.50, RMSE=6.65 (E-DAIC→EmpkinS)

**Key finding**: Cross-corpus transfer for PHQ-8 regression is **near zero** in the
EmpkinS→E-DAIC direction (Exp A/C1). The reverse (E-DAIC→EmpkinS, Exp B/C2) shows
modest signal (r≈0.4–0.45 for LAT configs). This asymmetry is informative.

---

## 6. Cross-Corpus Classification (EmpkinS ↔ E-DAIC, new for KDD)

### 6.1 Results summary (AUC / F1 / Balanced Accuracy)

| Config | Exp A: EmpkinS→E-DAIC | Exp B: E-DAIC→EmpkinS | Exp C1: EmpkinS-train→E-DAIC | Exp C2: E-DAIC-train→EmpkinS |
|---|---|---|---|---|
| **EI1_ADK** | AUC=0.634, F1=0.486 | AUC=0.806, F1=0.286 | AUC=0.594, F1=0.466 | AUC=0.722, F1=0.000 |
| **POS_ADK** | AUC=0.606, F1=0.450 | AUC=0.935, F1=0.256 | AUC=0.683, F1=0.466 | AUC=0.676, F1=0.199 |
| **POS_CRADK** | **AUC=0.732**, **F1=0.540** | AUC=0.661, F1=0.631 | AUC=0.519, F1=0.500 | AUC=0.648, F1=0.064 |
| **ALL** | AUC=0.561, F1=0.475 | AUC=0.630, F1=0.499 | AUC=0.634, F1=0.479 | AUC=0.553, F1=0.131 |

Best models: LGBM, MLP, GB, SVC, RF depending on config.

**Key finding**: POS_CRADK is the best config for EmpkinS→E-DAIC transfer
(AUC=0.73, F1=0.54 in Exp A). This is the same phase/condition that achieved F1=0.86
in within-corpus holdout — suggesting partial generalization.

However, F1 scores often degenerate due to **class imbalance** in E-DAIC (24% positive)
and **threshold effects**: many classifiers default to predicting majority class, yielding
F1≈0.47 (all-negative baseline). AUC is the more reliable metric here.

---

## 7. Interpretation: Why Does Transfer Fail?

| Challenge | Detail | Impact |
|---|---|---|
| Label mismatch | SCID-5-CV (EmpkinS) vs PHQ-8≥10 threshold (E-DAIC) | Both train and test PHQ-8 are self-report, yet regression fails — suggests feature mismatch is the primary cause |
| Context mismatch | 4 defined experimental phases (EmpkinS) vs unstructured 20-min interview (E-DAIC) | E-DAIC has no equivalent to "Positive Training" or "Mood Induction" phases |
| Class imbalance | E-DAIC: 24% positive vs EmpkinS: 50% | Degrades F1; use AUC as primary |
| Language/culture | German participants (EmpkinS) vs English-speaking US (E-DAIC) | Facial expression norms may differ |
| Sample size | E-DAIC test = 56 participants | Low statistical power; high variance in estimates |

**Key biomarker (AU14)**: AU14 (buccinator) shows erratic intensity peaks + rigid low-variability
dynamics in MDD — a psychomotor dysregulation signature. Gaze entropy is restricted; head pitch
variability is reduced. These features are phase-specific and may not manifest during
unstructured interview.

**Reverse transfer (B/C2) is sometimes better**: E-DAIC→EmpkinS shows modest signal
(r≈0.4–0.45 for latency configs). This may be because EmpkinS latency phase (passive baseline)
is *functionally similar* to E-DAIC interview (both are low-demand, naturalistic contexts).

---

## 8. Paper Structure (4–6 pages)

| Section | Content | ~Pages |
|---|---|---|
| **1. Introduction** | Context-dependency hypothesis; AI4Mental motivation; contribution | 0.5 |
| **2. Datasets & Methods** | EmpkinS RCT summary; E-DAIC; OpenFace 882-feature pipeline; ML pipeline (Spearman select, Pearson prune, GridSearchCV) | 1.0 |
| **3. Within-Corpus Results** | Phase×condition F1/AUC heatmap from journal; PHQ-8 regression (nested CV CCC≈0.07 full cohort; holdout CCC≈0.67 best condition restricted features); HRSD results as contrast | 1.0 |
| **4. Cross-Corpus Results** | Table: Reg Exp A–C across configs (CCC≈0); Table: Cls Exp A–C (AUC 0.52–0.73); POS_CRADK best for classification transfer | 1.5 |
| **5. Discussion** | Why transfer fails; asymmetry B>A; AU14 paradox not preserved in interview; label/context mismatch; implication for system design | 0.75 |
| **6. Conclusion** | Context matters; partial classification transfer shows promise; design implication for AI4Mental | 0.25 |

---

## 9. Key Numbers for Abstract / Highlights

- Within-corpus classification: F1 jumps from **0.56** (passive baseline) to **0.86** (CRADK, Positive Training)
- Within-corpus PHQ-8 regression: CCC = **0.67** (latency/ADK, restricted features, holdout); CCC = **0.07** (nested CV, all conditions) — report both with caveat
- Best within-corpus HRSD-6: CCC = **0.74**, Pearson r = **0.79** (neg_training/CR)
- Cross-corpus regression: CCC ≈ **0** (EmpkinS→E-DAIC), modest r ≈ **0.43** (E-DAIC→EmpkinS latency)
- Cross-corpus classification: best AUC = **0.73** (POS_CRADK, EmpkinS-ALL→E-DAIC)

---

## 10. What to Leave Out (keep for journal)

- Full regression tables for PHQ-9, CES-D, HRSD-17/21/24
- Deep learning comparisons (SEG, SPG, AVEC2019, OpenTSLM)
- All DL architecture details and hyperparameter tables
- Full SHAP beeswarm plots per condition
- Clinical treatment implications (cognitive restructuring efficacy)
- Landmark/composite expression features (not in cross-corpus set)

---

## 11. Pending / Open Questions

- [ ] Nested CV with restricted 882 features — currently only have latency/ALL_CONDITIONS (CCC=0.065). Need other phases for table completeness.
- [ ] PHQ-9 rerun with data-loading fix (text value bug fixed; results pending)
- [ ] Verify: does 882-feature restriction also improve within-corpus *classification*? (`holdout_v2.py` with feature filter)
- [ ] Decide primary metric for regression: CCC (concordance correlation) vs Pearson r vs MAE — CCC preferred as it penalizes both correlation and scale/offset mismatch
- [ ] Address class imbalance in cross-corpus classification write-up (F1 vs AUC; SMOTE not used — note in limitations)

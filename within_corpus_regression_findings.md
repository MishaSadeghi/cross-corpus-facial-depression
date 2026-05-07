# Within-Corpus Regression Findings — EmpkinS-EKSpression
## For KDD AI4Mental 2026 Paper

Generated: 2026-04-17. Results from holdout and nested CV experiments in
`within_corpus_regression/results/`.

---

## 1. Scale Reliability — Why PHQ-8 is Harder Than Other Scales

**Data collection protocol (important context)**:
- **PHQ-8**: Collected **remotely** by participants at home before the lab visit.
  Self-administered questionnaire with no clinical supervision. Subject to recall bias
  and reduced motivation.
- **PHQ-9, CES-D (ADS), HRSD-6/17/21/24**: Collected **in the lab** after the SCID-based
  clinical interview, with the psychologist present. More accurate, clinically supervised.
- **SCID binary label** (used for classification): Gold standard. Derived from structured
  clinical interview by trained psychologist. Most reliable label in the dataset.

**Implication for interpretation**: The ordering SCID (classification) > HRSD/CES-D > PHQ-8
reflects data quality and reliability, not just prediction difficulty. This is why:
- Classification (SCID label) results are the best across the board
- HRSD/CES-D regression within-corpus is stronger than PHQ-8
- PHQ-8 cross-corpus failure is particularly unsurprising

**This should be stated explicitly in the paper.** The psychologists on the team confirm this
interpretation.

---

## 2. Full Features vs 882 Cross-Corpus Features

**Key finding: the restricted 882 feature set outperforms full features for PHQ-8.**

| Config | PHQ-8 CCC (full features) | PHQ-8 CCC (882 restricted) |
|--------|--------------------------|---------------------------|
| latency/ADK   | 0.03 | **0.66** |
| latency/CRADK | 0.49 | **0.66** |
| positive_training/CR | 0.28 | **0.63** |
| negative_training/CR | 0.28 | **0.57** |
| latency/SHAM  | 0.19 | **0.55** |
| EI1/ADK       | 0.22 | **0.46** |

**Why**: The 882 features are a targeted subset of facial AUs, gaze angles, and head pose
(OpenFace standard outputs). Restricting to these removes thousands of noisy features from
EmpkinS (e.g. landmark positions, composite expressions) that do not transfer.
The restricted set acts as an implicit regularisation.

**Important for the paper**: The cross-corpus experiments ALREADY use the 882 restricted
features (they must be the same features in both datasets). So the fair within-corpus
comparison to cross-corpus is:
- Within-corpus PHQ-8 CCC (882 features) = 0.55–0.66 (best configs)
- Cross-corpus PHQ-8 CCC (same 882 features) ≈ 0

This is a much stronger result than comparing full-feature within-corpus to restricted
cross-corpus.

**The 882 feature names are saved in**: `cross_corpus_882_features.txt`

**Note for classification**: These 882 features may also improve within-corpus
classification results (the existing classification experiments used the full feature set).
This is worth testing — if the restricted features also improve classification, it provides
a cleaner and more consistent analysis across the paper.

---

## 3. Best Within-Corpus Results by Scale (Holdout, Restricted 882 Features)

### Top 3 configs per scale:

**PHQ-8 (phq_8)** — remote self-report, least reliable
| Phase | Condition | CCC | MAE | Pearson r | n_test |
|-------|-----------|-----|-----|-----------|--------|
| latency | ADK | 0.665 | 3.94 | 0.665 | 12 |
| latency | CRADK | 0.661 | 4.47 | 0.674 | 11 |
| positive_training | CR | 0.626 | 3.43 | 0.678 | 12 |

**CES-D / ADS (ads.1)** — lab-rated, supervised
| Phase | Condition | CCC | MAE | Pearson r | n_test |
|-------|-----------|-----|-----|-----------|--------|
| positive_training | SHAM | 0.582 | 10.1 | 0.582 | 13 |
| latency | ADK | 0.556 | 12.3 | 0.624 | 12 |
| latency | SHAM | 0.479 | 9.4 | 0.571 | 13 |

**HRSD-6 (HRSD_6.1)** — clinician-rated, most intensive
| Phase | Condition | CCC | MAE | Pearson r | n_test |
|-------|-----------|-----|-----|-----------|--------|
| negative_training | CR | 0.736 | 2.88 | 0.792 | 12 |
| latency | CR | 0.570 | 3.63 | 0.624 | 12 |
| positive_training | CR | 0.519 | 4.27 | 0.645 | 12 |

**HRSD-17 (HRSD_17.1)**
| Phase | Condition | CCC | MAE | Pearson r | n_test |
|-------|-----------|-----|-----|-----------|--------|
| latency | CR | 0.632 | 5.62 | 0.740 | 12 |
| negative_training | CR | 0.590 | 7.34 | 0.647 | 12 |
| positive_training | CR | 0.403 | 8.08 | 0.504 | 12 |

**HRSD-21 (HRSD_21.1)**
| Phase | Condition | CCC | MAE | Pearson r | n_test |
|-------|-----------|-----|-----|-----------|--------|
| negative_training | CR | 0.572 | 7.46 | 0.706 | 12 |
| emotion_induction_1 | CRADK | 0.516 | 5.68 | 0.678 | 11 |
| positive_training | CR | 0.462 | 9.22 | 0.546 | 12 |

**HRSD-24 (HRSD_24.1)**
| Phase | Condition | CCC | MAE | Pearson r | n_test |
|-------|-----------|-----|-----|-----------|--------|
| latency | CR | 0.569 | 9.05 | 0.667 | 12 |
| negative_training | CR | 0.557 | 10.59 | 0.652 | 12 |
| positive_training | CR | 0.501 | 11.21 | 0.549 | 12 |

**PHQ-9 (phq_9)** — ⚠️ Data quality issue: 1 row in master_data has text value
("10 (bei Trainingstermin ausgefüllt)"). Fixed in code (coerce to numeric, drop).
PHQ-9 needs to be re-run with the fix.

---

## 4. Best Within-Corpus Results by Scale (Holdout, Full Features)

| Scale | Best phase | Condition | CCC | MAE | Pearson r |
|-------|------------|-----------|-----|-----|-----------|
| HRSD-6 | latency | CR | 0.688 | 3.37 | 0.782 |
| CES-D | negative_training | CR | 0.626 | 8.45 | 0.680 |
| HRSD-24 | latency | CR | 0.617 | 8.84 | 0.755 |
| HRSD-17 | emotion_induction_1 | CRADK | 0.581 | 5.01 | 0.658 |
| HRSD-21 | latency | CR | 0.575 | 6.78 | 0.707 |
| PHQ-8 | latency | CRADK | 0.488 | 4.78 | 0.582 |
| PHQ-9 | — | — | needs re-run | | |

---

## 5. Phase Hierarchy for PHQ-8 Regression

With restricted 882 features:
1. **latency** (passive baseline observation) — best for PHQ-8, all conditions
2. **positive_training** (CR condition) — second best
3. **negative_training** (CR condition) — third

With HRSD scales:
1. **negative_training / CR** — best for HRSD (high-intensity emotional expression)
2. **latency / CR** — second best
3. **positive_training / CR** — third

Pattern: **CR (Cognitive Reappraisal) condition** appears consistently in the best
HRSD configs. This may reflect that CR elicits more natural facial expressions of emotional
regulation, which correlates with clinician severity ratings.

---

## 6. Config Selection for Cross-Corpus Regression

**Current cross-corpus configs** (selected for classification F1, not PHQ-8 CCC):
- EI1_ADK, POS_ADK, POS_CRADK, ALL

**Recommended configs** (selected by within-corpus PHQ-8 CCC with restricted features):
| Tag | Phase | Condition | Within-corpus CCC | Rationale |
|-----|-------|-----------|-------------------|-----------|
| LAT_ADK | latency | ADK | 0.665 | Best PHQ-8; passive observation; AFE condition |
| LAT_SHAM | latency | SHAM | 0.554 | PHQ-8 CCC=0.55; control condition (no intervention) |
| EI1_ADK | emotion_induction_1 | ADK | 0.457 | Cross-method comparison with classification |
| ALL | all phases | all conds | ~0.18 | Largest training set; upper bound baseline |

**On consistency between classification and regression configs**:
It is methodologically sound to use *different* configs for classification vs regression
because the optimal phase/condition is task-specific:
- Classification is driven by SCID binary labels → EI1/ADK (emotion induction, closest
  to clinical interview context) is theoretically justified
- PHQ-8 regression is driven by remote self-report → latency/ADK (passive baseline)
  is better because it is the cleanest facial baseline, free from intervention effects

This difference is itself an interesting finding: the phase that best captures
*severity* (latency, passive baseline) differs from the phase that best captures
*diagnosis* (emotion induction, closer to interview). Worth 1–2 sentences in the paper.

---

## 7. Nested CV vs Holdout Discrepancy

For individual conditions (n_test ≈ 11–13), holdout results can be inflated due to
small test set size. Cross-reference with nested CV:

| Config | Holdout CCC (restricted) | Nested CV mean CCC (full) |
|--------|--------------------------|--------------------------|
| latency/ADK | 0.665 | 0.268 |
| latency/SHAM | 0.554 | 0.257 |
| EI1/ADK | 0.457 | 0.243 |

The nested CV (full features, n_folds=5) gives lower but more conservative estimates.
For the paper, report **nested CV** as the primary within-corpus metric (more robust
to small test set variance), with holdout as supplementary.

---

## 8. Results Directory Structure

```
within_corpus_regression/results/
    holdout_full_features/      ← full feature set holdout
        holdout_{phase}_{cond}_{timestamp}.csv
        holdout_{phase}_SUMMARY_{timestamp}.csv
    holdout_restricted_features/ ← 882 cross-corpus feature set holdout
        holdout_{phase}_{cond}_crossfeats_{timestamp}.csv
        holdout_{phase}_SUMMARY_crossfeats_{timestamp}.csv
    nested_cv/                  ← 5-fold nested CV (full features)
        nested_cv_folds_{phase}_{cond}_{timestamp}.csv
        nested_cv_summary_{phase}_{cond}_{timestamp}.csv
        nested_cv_ALL_{phase}_5folds_{timestamp}.csv
```

---

## 9. Pending Work

- [ ] Re-run holdout with `--targets phq_9` (data loading fix now in place)
- [ ] Re-run nested CV with restricted features (`--restrict-to-cross-corpus`)
  to get conservative restricted-feature estimates for nested CV
- [ ] Update cross-corpus regression script (`06_cross_corpus_regression_v2.py`)
  to use LAT_ADK / LAT_SHAM / EI1_ADK / ALL configs
- [ ] Test if 882 restricted features also improve within-corpus classification
  (use `holdout_v2.py` with a restricted feature pre-filter)

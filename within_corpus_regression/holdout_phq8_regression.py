"""
holdout_phq8_regression.py
==========================
Within-corpus multi-scale regression on ProposedCorpus using a fixed holdout split.

Supports all 7 depression severity scales:
  PHQ-8 (phq_8), PHQ-9 (phq_9), CES-D/ADS (ads.1),
  HRSD-6 (HRSD_6.1), HRSD-17 (HRSD_17.1), HRSD-21 (HRSD_21.1), HRSD-24 (HRSD_24.1)

Strict no-leakage pipeline (per target × condition):
  1. Load all data for the phase/condition.
  2. Aggregate to ONE ROW PER PARTICIPANT (mean across sessions).
  3. Filter to shared train/test participant IDs (same splits used in classification).
  4. [Optional] Restrict features to the 882-feature E-DAIC intersection
     (--restrict-to-cross-corpus).
  5. Drop NaN-only columns on train; reindex test to train columns.
  6. Impute: SimpleImputer(median) fit on train only, transform both.
  7. For each scaler {StandardScaler, MinMaxScaler, RobustScaler}:
       a. Fit scaler on train only, transform both.
       b. Spearman FDR feature selection on train (q=0.20, fallback top-20).
       c. RFE (Ridge surrogate) on train.
       d. For each regressor: GridSearchCV on train (5-fold inner CV).
       e. Evaluate on test: MAE, RMSE, R², Pearson r, Spearman ρ, CCC.
  8. Save per-condition CSV + print summary table.

Phases: latency, emotion_induction_1, negative_training, positive_training
Conditions: ADK, CR, CRADK, SHAM, ALL_CONDITIONS

Usage:
  python holdout_phq8_regression.py --phase latency
  python holdout_phq8_regression.py --phase positive_training --conditions ADK CR
  python holdout_phq8_regression.py --phase latency --targets phq_8 phq_9 ads.1
  python holdout_phq8_regression.py --phase latency --restrict-to-cross-corpus
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── Shared pipeline ───────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from shared_regression_pipeline import (
    ALL_TARGET_COLS,
    EDAIC_AGGREGATED_PATH,
    PROPOSED_PHASE_FEATURES,
    EXCLUDE_COLS,
    TARGET_SHORT_NAMES,
    evaluate_regressor_ext,
    get_param_grids,
    get_regressors,
    get_scalers,
    grid_search_regressor,
    load_proposed_data,
    load_shared_split,
    rfe_select_regression,
    spearman_select_fdr,
)

RESULTS_BASE = os.path.join(SCRIPT_DIR, "results")

CONDITIONS_DEFAULT = ["ADK", "CR", "CRADK", "SHAM", "ALL_CONDITIONS"]
PHASES_AVAILABLE   = list(PROPOSED_PHASE_FEATURES.keys())
TARGETS_DEFAULT    = ALL_TARGET_COLS   # all 7 scales by default


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CORPUS FEATURE RESTRICTION
# ─────────────────────────────────────────────────────────────────────────────
_cross_corpus_cols_cache = None

def get_cross_corpus_features() -> list:
    """Return column names available in E-DAIC aggregated CSV (excluding metadata)."""
    global _cross_corpus_cols_cache
    if _cross_corpus_cols_cache is not None:
        return _cross_corpus_cols_cache
    df_header = pd.read_csv(EDAIC_AGGREGATED_PATH, nrows=0)
    cols = [c for c in df_header.columns if c not in EXCLUDE_COLS]
    print("  E-DAIC header loaded: {} feature columns".format(len(cols)))
    _cross_corpus_cols_cache = cols
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-TARGET HOLDOUT
# ─────────────────────────────────────────────────────────────────────────────
def run_one_target(phase, condition, target_col, feat_label,
                   regressors, param_grids, scalers, cross_corpus_cols):
    """Run holdout regression for one (phase, condition, target_col) triple.
    Returns list of result dicts (one per (scaler, model) combo)."""
    short = TARGET_SHORT_NAMES.get(target_col, target_col)
    split_tag = "ALL" if condition == "ALL_CONDITIONS" else condition

    print("\n  [Target: {}]".format(short))

    # 1. Load data
    try:
        X_full, y_full, _ = load_proposed_data(phase, condition, target_col)
    except Exception as e:
        print("  ERROR loading: {}".format(e))
        return []

    if X_full.empty or y_full.empty:
        print("  Skipping: no data.")
        return []

    # 2. Aggregate to one row per participant
    n_rows_before = len(X_full)
    X_full = X_full.groupby(X_full.index).mean()
    y_full = y_full.groupby(y_full.index).first()
    print("  Aggregated {} rows → {} participants".format(n_rows_before, len(X_full)))

    # 3. Shared split
    try:
        train_ids, test_ids = load_shared_split(split_tag)
    except FileNotFoundError as e:
        print("  ERROR: {}".format(e))
        return []

    available   = set(X_full.index.astype(str))
    phase_train = sorted(train_ids & available)
    phase_test  = sorted(test_ids  & available)

    missing_tr = len(train_ids - available)
    missing_te = len(test_ids  - available)
    if missing_tr:
        print("  NOTE: {} train IDs absent from this phase/condition.".format(missing_tr))
    if missing_te:
        print("  NOTE: {} test IDs absent from this phase/condition.".format(missing_te))

    X_train = X_full.loc[X_full.index.isin(phase_train)].copy()
    X_test  = X_full.loc[X_full.index.isin(phase_test)].copy()
    y_train = y_full.loc[y_full.index.isin(phase_train)].copy()
    y_test  = y_full.loc[y_full.index.isin(phase_test)].copy()

    print("  Train: {}  |  Test: {}  |  {} train: mean={:.1f} std={:.1f}  "
          "test: mean={:.1f} std={:.1f}".format(
        len(y_train), len(y_test), short,
        y_train.mean(), y_train.std(), y_test.mean(), y_test.std()))

    if len(y_train) < 5 or len(y_test) < 2:
        print("  Skipping: insufficient samples.")
        return []

    # 4. Optional cross-corpus restriction
    if cross_corpus_cols is not None:
        overlap = [c for c in cross_corpus_cols if c in X_train.columns]
        if len(overlap) >= 5:
            X_train = X_train[overlap].copy()
            X_test  = X_test.reindex(columns=overlap)
            print("  Restricted to {} cross-corpus features.".format(len(overlap)))
        else:
            print("  WARNING: only {} overlap features; skipping restriction.".format(
                len(overlap)))

    # 5. Drop NaN-only columns (train), reindex test
    nan_only = [c for c in X_train.columns if X_train[c].isna().all()]
    if nan_only:
        X_train = X_train.drop(columns=nan_only)
    X_test = X_test.reindex(columns=X_train.columns)

    # 6. Impute (fit on train only)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    ).astype(np.float64)
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns, index=X_test.index
    ).astype(np.float64)

    rows = []

    # 7. Scaler → feature selection → models
    for scaler_name, scaler in scalers.items():
        X_tr_sc = pd.DataFrame(
            scaler.fit_transform(X_train_imp),
            columns=X_train_imp.columns, index=X_train_imp.index
        )
        X_te_sc = pd.DataFrame(
            scaler.transform(X_test_imp),
            columns=X_test_imp.columns, index=X_test_imp.index
        )

        top_feats = spearman_select_fdr(X_tr_sc, y_train, q=0.20, fallback_top_n=20)
        if not top_feats:
            continue

        X_tr_fs = X_tr_sc[top_feats]
        X_te_fs = X_te_sc[top_feats]

        rfe_range = tuple(sorted(set(
            k for k in (5, 10, 15, min(20, len(top_feats)))
            if k <= len(top_feats)
        ))) or (len(top_feats),)

        selected = rfe_select_regression(X_tr_fs, y_train,
                                         num_features_range=rfe_range)
        selected = [f for f in selected if f in X_tr_fs.columns] or top_feats

        X_tr_rfe = X_tr_fs[selected]
        X_te_rfe = X_te_fs[selected]

        for model_name, model in regressors.items():
            pg = param_grids.get(model_name, {})
            try:
                fitted  = grid_search_regressor(clone(model), pg,
                                                X_tr_rfe, y_train, cv=5)
                metrics = evaluate_regressor_ext(fitted, X_te_rfe, y_test)
            except Exception as e:
                print("    {}/{}/{}: FAILED ({})".format(
                    short, scaler_name, model_name, e))
                continue

            rows.append({
                "phase":      phase,
                "condition":  condition,
                "target":     target_col,
                "target_short": short,
                "features":   feat_label,
                "scaler":     scaler_name,
                "model":      model_name,
                "n_train":    len(y_train),
                "n_test":     len(y_test),
                "n_features": len(selected),
                "CCC":        metrics["CCC"],
                "MAE":        metrics["MAE"],
                "RMSE":       metrics["RMSE"],
                "Pearson_r":  metrics["Pearson_r"],
                "R2":         metrics["R2"],
                "Spearman_r": metrics["Spearman_r"],
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HOLDOUT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_holdout(phase: str, conditions: list, targets: list,
                restrict_cross_corpus: bool):
    regressors  = get_regressors()
    param_grids = get_param_grids()
    scalers     = get_scalers()

    cross_corpus_cols = get_cross_corpus_features() if restrict_cross_corpus else None
    feat_label = "cross-corpus-882" if restrict_cross_corpus else "full"

    subdir = "holdout_restricted_features" if restrict_cross_corpus else "holdout_full_features"
    RESULTS_DIR = os.path.join(RESULTS_BASE, subdir)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_rows = []

    for condition in conditions:
        print("\n{}".format("#" * 80))
        print("  HOLDOUT REGRESSION  |  Phase: {}  |  Condition: {}  |  Features: {}".format(
            phase.upper(), condition.upper(), feat_label))
        print("  Targets: {}".format(", ".join(
            TARGET_SHORT_NAMES.get(t, t) for t in targets)))
        print("{}\n".format("#" * 80))

        condition_rows = []
        for target_col in targets:
            rows = run_one_target(
                phase, condition, target_col, feat_label,
                regressors, param_grids, scalers, cross_corpus_cols
            )
            condition_rows.extend(rows)
            all_rows.extend(rows)

        if not condition_rows:
            print("  No results for {}/{}.".format(phase, condition))
            continue

        # Save per-condition CSV (all targets merged)
        df = pd.DataFrame(condition_rows).sort_values(
            ["target_short", "CCC"], ascending=[True, False])
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = "_crossfeats" if restrict_cross_corpus else ""
        csv_path = os.path.join(
            RESULTS_DIR,
            "holdout_{}_{}{}_{}.csv".format(phase, condition, suffix, timestamp)
        )
        df.to_csv(csv_path, index=False)
        print("\n  Saved: {}".format(csv_path))

        # Best model per (target, scaler)
        print("\n  Best per target × scaler ({}/{}):".format(phase, condition))
        print("  {:8s}  {:20s}  {:12s}  {:>7s}  {:>7s}  {:>7s}".format(
            "Target", "Scaler", "Model", "CCC", "MAE", "r"))
        for tgt in df["target_short"].unique():
            sub_t = df[df["target_short"] == tgt]
            for sn in sub_t["scaler"].unique():
                best = sub_t[sub_t["scaler"] == sn].iloc[0]
                print("  {:8s}  {:20s}  {:12s}  {:7.3f}  {:7.3f}  {:7.3f}".format(
                    tgt, sn, best["model"],
                    best["CCC"], best["MAE"], best["Pearson_r"]))

    # Overall summary: best per (condition, target)
    if all_rows:
        print("\n\n{}".format("=" * 80))
        print("  OVERALL SUMMARY | Phase: {} | Features: {}".format(phase, feat_label))
        print("{}\n".format("=" * 80))
        overall_df = pd.DataFrame(all_rows)
        summary = (
            overall_df
            .sort_values("CCC", ascending=False)
            .groupby(["condition", "target_short"])
            .first()
            .reset_index()
            [["condition", "target_short", "model", "scaler",
              "n_train", "n_test", "CCC", "MAE", "RMSE", "Pearson_r"]]
        )
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)
        print(summary.to_string(index=False))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = "_crossfeats" if restrict_cross_corpus else ""
        combined_path = os.path.join(
            RESULTS_DIR,
            "holdout_{}_SUMMARY{}_{}.csv".format(phase, suffix, timestamp)
        )
        overall_df.to_csv(combined_path, index=False)
        print("\n  Combined CSV: {}".format(combined_path))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Within-corpus multi-scale holdout regression for ProposedCorpus"
    )
    parser.add_argument("--phase", required=True, choices=PHASES_AVAILABLE,
                        help="Experimental phase")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS_DEFAULT,
                        help="Conditions to run (default: all 5)")
    parser.add_argument(
        "--targets", nargs="+", default=TARGETS_DEFAULT,
        choices=ALL_TARGET_COLS,
        help="Regression targets (default: all 7 scales). "
             "Choices: {}".format(" ".join(ALL_TARGET_COLS))
    )
    parser.add_argument("--restrict-to-cross-corpus", action="store_true",
                        dest="restrict_cross_corpus",
                        help="Use only the 882 features shared with E-DAIC")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_holdout(args.phase, args.conditions, args.targets,
                args.restrict_cross_corpus)

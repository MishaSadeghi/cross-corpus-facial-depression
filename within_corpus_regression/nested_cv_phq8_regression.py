"""
nested_cv_phq8_regression.py
=============================
Within-corpus multi-scale nested CV regression on EmpkinS-EKSpression.

Supports all 7 depression severity scales:
  PHQ-8 (phq_8), PHQ-9 (phq_9), CES-D/ADS (ads.1),
  HRSD-6 (HRSD_6.1), HRSD-17 (HRSD_17.1), HRSD-21 (HRSD_21.1), HRSD-24 (HRSD_24.1)

Structure: 5-fold outer CV × 3-fold inner (GridSearchCV).
Outer folds stratified by SCID binary label so each fold preserves dep/HC balance.
All operations (impute, scale, feature selection, RFE) fit on the outer TRAIN fold
only — no leakage from test fold.

No fixed split is used here; ALL data for the phase/condition enters the CV.

Usage:
  python nested_cv_phq8_regression.py --phase latency
  python nested_cv_phq8_regression.py --phase positive_training --conditions ADK CR
  python nested_cv_phq8_regression.py --phase latency --targets phq_8 phq_9 ads.1
  python nested_cv_phq8_regression.py --phase latency --restrict-to-cross-corpus
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
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── Shared pipeline ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from shared_regression_pipeline import (
    ALL_TARGET_COLS,
    EDAIC_AGGREGATED_PATH,
    EMPKINS_PHASE_FEATURES,
    EXCLUDE_COLS,
    MASTER_DATA_PATH,
    TARGET_SHORT_NAMES,
    evaluate_regressor_ext,
    get_param_grids,
    get_regressors,
    get_scalers,
    grid_search_regressor,
    load_empkins_data,
    normalize_id,
    rfe_select_regression,
    spearman_select_fdr,
)

RESULTS_BASE = os.path.join(SCRIPT_DIR, "results")

CONDITIONS_DEFAULT = ["ADK", "CR", "CRADK", "SHAM", "ALL_CONDITIONS"]
PHASES_AVAILABLE   = list(EMPKINS_PHASE_FEATURES.keys())
TARGETS_DEFAULT    = ALL_TARGET_COLS
N_OUTER_FOLDS      = 5
OUTER_SEED         = 42


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CORPUS FEATURE RESTRICTION
# ─────────────────────────────────────────────────────────────────────────────
_cross_corpus_cols_cache = None

def get_cross_corpus_features() -> list:
    global _cross_corpus_cols_cache
    if _cross_corpus_cols_cache is not None:
        return _cross_corpus_cols_cache
    df_header = pd.read_csv(EDAIC_AGGREGATED_PATH, nrows=0)
    cols = [c for c in df_header.columns if c not in EXCLUDE_COLS]
    print("  E-DAIC header loaded: {} feature columns".format(len(cols)))
    _cross_corpus_cols_cache = cols
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SCID LABELS  (for stratified outer folds)
# ─────────────────────────────────────────────────────────────────────────────
def load_scid_labels(condition: str) -> pd.Series:
    master = pd.read_csv(MASTER_DATA_PATH)
    master["ID"] = master["ID"].apply(normalize_id)
    master.rename(columns={"label": "ground_truth_label"}, inplace=True)
    if condition != "ALL_CONDITIONS":
        master = master[master["condition"] == condition]
    master = master.dropna(subset=["ground_truth_label"])
    return master.set_index("ID")["ground_truth_label"].astype(int)


def load_stratification_labels(condition: str) -> pd.Series:
    """For ALL_CONDITIONS: returns condition+SCID combined stratum label
    so StratifiedKFold balances both condition and dep/HC simultaneously.
    For single conditions: falls back to SCID binary label only."""
    master = pd.read_csv(MASTER_DATA_PATH)
    master["ID"] = master["ID"].apply(normalize_id)
    master.rename(columns={"label": "ground_truth_label"}, inplace=True)
    if condition != "ALL_CONDITIONS":
        master = master[master["condition"] == condition]
    master = master.dropna(subset=["ground_truth_label"])
    if condition == "ALL_CONDITIONS":
        # Combined stratum: e.g. "ADK_1", "SHAM_0" — 8 strata total
        stratum = pd.Series(
            (master["condition"] + "_" + master["ground_truth_label"].astype(int).astype(str)).values,
            index=master["ID"].values
        )
        return stratum
    return master.set_index("ID")["ground_truth_label"].astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-TARGET NESTED CV
# ─────────────────────────────────────────────────────────────────────────────
def run_one_target_cv(phase, condition, target_col, feat_label,
                      regressors, param_grids, scalers,
                      cross_corpus_cols, X_preloaded=None,
                      scid_preloaded=None):
    """Run 5-fold nested CV for one (phase, condition, target_col) triple.
    X_preloaded / scid_preloaded can be passed to avoid re-loading features
    when iterating over multiple targets in the same condition.

    Returns (fold_results_list, agg_df).
    """
    short = TARGET_SHORT_NAMES.get(target_col, target_col)
    print("\n  [Target: {}]".format(short))

    # 1. Load and aggregate
    try:
        X_full, y_full, _ = load_empkins_data(phase, condition, target_col)
    except Exception as e:
        print("  ERROR loading: {}".format(e))
        return [], pd.DataFrame()

    if X_full.empty or y_full.empty:
        print("  Skipping: no data.")
        return [], pd.DataFrame()

    n_rows_before = len(X_full)
    X_full = X_full.groupby(X_full.index).mean()
    y_full = y_full.groupby(y_full.index).first()
    print("  Aggregated {} rows → {} participants".format(n_rows_before, len(X_full)))

    # Stratification labels: condition+SCID for ALL_CONDITIONS, SCID only otherwise
    strat_all  = load_stratification_labels(condition)
    strat_full = strat_all.reindex(X_full.index)
    if condition == "ALL_CONDITIONS":
        strat_full = strat_full.fillna("UNKNOWN_0")
    else:
        strat_full = strat_full.fillna(0).astype(int)
    if strat_full.nunique() < 2:
        print("  Skipping: only one stratum class.")
        return [], pd.DataFrame()

    participant_ids = X_full.index.tolist()
    n_participants  = len(participant_ids)
    if condition == "ALL_CONDITIONS":
        print("  Participants: {}  strata: {}".format(
            n_participants, dict(strat_full.value_counts())))
    else:
        print("  Participants: {}  (dep: {}  HC: {})".format(
            n_participants, int((strat_full == 1).sum()), int((strat_full == 0).sum())))

    if n_participants < N_OUTER_FOLDS * 2:
        print("  Skipping: too few participants.")
        return [], pd.DataFrame()

    # 2. Optional cross-corpus restriction
    if cross_corpus_cols is not None:
        overlap = [c for c in cross_corpus_cols if c in X_full.columns]
        if len(overlap) >= 5:
            X_full = X_full[overlap].copy()
            print("  Restricted to {} cross-corpus features.".format(len(overlap)))

    # 3. Outer 5-fold CV — stratified by condition+SCID (ALL) or SCID only (single)
    skf        = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                                 random_state=OUTER_SEED)
    pid_array  = np.array(participant_ids)
    strat_array = strat_full.loc[pid_array].values

    fold_results = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(pid_array, strat_array)):
        train_pids = set(pid_array[tr_idx])
        test_pids  = set(pid_array[te_idx])

        X_train = X_full.loc[X_full.index.isin(train_pids)].copy()
        X_test  = X_full.loc[X_full.index.isin(test_pids)].copy()
        y_train = y_full.loc[y_full.index.isin(train_pids)].copy()
        y_test  = y_full.loc[y_full.index.isin(test_pids)].copy()

        print("  Fold {}/{}: train={} test={}".format(
            fold_idx + 1, N_OUTER_FOLDS, len(y_train), len(y_test)))

        if len(y_train) < 5 or len(y_test) < 2:
            continue

        # Drop NaN-only cols (train), reindex test
        nan_only = [c for c in X_train.columns if X_train[c].isna().all()]
        if nan_only:
            X_train = X_train.drop(columns=nan_only)
        X_test = X_test.reindex(columns=X_train.columns)

        # Impute (fit on train)
        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index
        ).astype(np.float64)
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns, index=X_test.index
        ).astype(np.float64)

        for scaler_name, scaler in scalers.items():
            X_tr_sc = pd.DataFrame(
                scaler.fit_transform(X_train_imp),
                columns=X_train_imp.columns, index=X_train_imp.index
            )
            X_te_sc = pd.DataFrame(
                scaler.transform(X_test_imp),
                columns=X_test_imp.columns, index=X_test_imp.index
            )

            top_feats = spearman_select_fdr(X_tr_sc, y_train,
                                            q=0.20, fallback_top_n=20)
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
                                                    X_tr_rfe, y_train, cv=3)
                    metrics = evaluate_regressor_ext(fitted, X_te_rfe, y_test)
                except Exception as e:
                    print("    {}/{}/{}: FAILED ({})".format(
                        short, scaler_name, model_name, e))
                    continue

                fold_results.append({
                    "fold":         fold_idx + 1,
                    "phase":        phase,
                    "condition":    condition,
                    "target":       target_col,
                    "target_short": short,
                    "features":     feat_label,
                    "scaler":       scaler_name,
                    "model":        model_name,
                    "n_train":      len(y_train),
                    "n_test":       len(y_test),
                    "n_features":   len(selected),
                    "CCC":          metrics["CCC"],
                    "MAE":          metrics["MAE"],
                    "RMSE":         metrics["RMSE"],
                    "Pearson_r":    metrics["Pearson_r"],
                    "R2":           metrics["R2"],
                    "Spearman_r":   metrics["Spearman_r"],
                })

    if not fold_results:
        return [], pd.DataFrame()

    # Aggregate across folds
    folds_df = pd.DataFrame(fold_results)
    agg = (
        folds_df
        .groupby(["scaler", "model"])
        .agg(
            mean_CCC=("CCC",      "mean"),
            std_CCC=("CCC",       "std"),
            mean_MAE=("MAE",      "mean"),
            std_MAE=("MAE",       "std"),
            mean_RMSE=("RMSE",    "mean"),
            mean_r=("Pearson_r",  "mean"),
            std_r=("Pearson_r",   "std"),
            mean_R2=("R2",        "mean"),
            n_folds=("fold",      "count"),
        )
        .reset_index()
        .sort_values("mean_CCC", ascending=False)
    )
    agg["phase"]        = phase
    agg["condition"]    = condition
    agg["target"]       = target_col
    agg["target_short"] = short
    agg["features"]     = feat_label

    # Print top-3
    print("\n  Top 3 (mean CCC) | {} | {}:".format(short, condition))
    print("  {:20s}  {:12s}  {:>8s} ±{:>6s}  {:>8s}  {:>6s}".format(
        "Scaler", "Model", "CCC", "std", "MAE", "r"))
    for _, row in agg.head(3).iterrows():
        print("  {:20s}  {:12s}  {:8.3f} ±{:6.3f}  {:8.3f}  {:6.3f}".format(
            row["scaler"], row["model"],
            row["mean_CCC"], row.get("std_CCC", 0) or 0,
            row["mean_MAE"], row["mean_r"]))

    return fold_results, agg


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NESTED CV PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_nested_cv(phase: str, conditions: list, targets: list,
                  restrict_cross_corpus: bool):
    regressors  = get_regressors()
    param_grids = get_param_grids()
    scalers     = get_scalers()

    cross_corpus_cols = get_cross_corpus_features() if restrict_cross_corpus else None
    feat_label = "cross-corpus-882" if restrict_cross_corpus else "full"

    subdir = "nested_cv_restricted_features" if restrict_cross_corpus else "nested_cv_full_features"
    RESULTS_DIR = os.path.join(RESULTS_BASE, subdir)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_agg_rows = []

    for condition in conditions:
        print("\n{}".format("#" * 80))
        print("  NESTED CV REGRESSION  |  Phase: {}  |  Condition: {}  |  Features: {}".format(
            phase.upper(), condition.upper(), feat_label))
        print("  Targets: {}".format(", ".join(
            TARGET_SHORT_NAMES.get(t, t) for t in targets)))
        print("{}\n".format("#" * 80))

        condition_folds = []
        condition_aggs  = []

        for target_col in targets:
            fold_results, agg_df = run_one_target_cv(
                phase, condition, target_col, feat_label,
                regressors, param_grids, scalers, cross_corpus_cols
            )
            condition_folds.extend(fold_results)
            if not agg_df.empty:
                condition_aggs.append(agg_df)
                all_agg_rows.append(agg_df)

        if not condition_folds:
            print("  No fold results for {}/{}.".format(phase, condition))
            continue

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix    = "_crossfeats" if restrict_cross_corpus else ""

        # Per-fold CSV
        folds_df  = pd.DataFrame(condition_folds)
        fold_csv  = os.path.join(
            RESULTS_DIR,
            "nested_cv_folds_{}_{}{}_{}.csv".format(phase, condition, suffix, timestamp)
        )
        folds_df.to_csv(fold_csv, index=False)
        print("\n  Per-fold CSV: {}".format(fold_csv))

        # Summary CSV (all targets × scalers × models)
        if condition_aggs:
            summary_df  = pd.concat(condition_aggs, ignore_index=True)
            summary_csv = os.path.join(
                RESULTS_DIR,
                "nested_cv_summary_{}_{}{}_{}.csv".format(
                    phase, condition, suffix, timestamp)
            )
            summary_df.to_csv(summary_csv, index=False)
            print("  Summary CSV: {}".format(summary_csv))

    # Combined across all conditions
    if all_agg_rows:
        combined    = pd.concat(all_agg_rows, ignore_index=True)
        timestamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix      = "_crossfeats" if restrict_cross_corpus else ""
        combined_csv = os.path.join(
            RESULTS_DIR,
            "nested_cv_ALL_{}{}_{}folds_{}.csv".format(
                phase, suffix, N_OUTER_FOLDS, timestamp)
        )
        combined.to_csv(combined_csv, index=False)
        print("\n\n  All-condition summary: {}".format(combined_csv))

        # Print cross-condition overview: best CCC per (condition, target)
        print("\n  BEST CCC PER (CONDITION, TARGET) | Phase: {} | Features: {}".format(
            phase, feat_label))
        overview = (
            combined
            .sort_values("mean_CCC", ascending=False)
            .groupby(["condition", "target_short"])
            .first()
            .reset_index()
            [["condition", "target_short", "model", "scaler",
              "mean_CCC", "std_CCC", "mean_MAE", "mean_r"]]
        )
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)
        print(overview.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Within-corpus multi-scale nested CV regression for EmpkinS"
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
    run_nested_cv(args.phase, args.conditions, args.targets,
                  args.restrict_cross_corpus)

"""
03_cross_corpus_video_regression.py
=====================================
Cross-corpus PHQ-8 regression: ProposedCorpus RCT video features ↔ E-DAIC.

Source domain: ProposedCorpus emotion_induction_1 / ADK condition
  — chosen for highest classification F1 (0.83) in a passive-observation phase
    that best matches the E-DAIC clinical interview context.

Requires:
  - 01_aggregate_edaic_video_features.py must have been run (E-DAIC aggregated CSV)

Experiments:
  A. ProposedCorpus (ALL) → E-DAIC test   (all ProposedCorpus as train, E-DAIC official test)
  B. E-DAIC (train+dev) → ProposedCorpus  (E-DAIC as train, ProposedCorpus official holdout)
  C1. ProposedCorpus train split → E-DAIC test   (uses the fixed 80/20 ProposedCorpus split)
  C2. E-DAIC train+dev → ProposedCorpus test     (symmetric)

Feature selection (FDR+RFE — same as repeated holdout v2):
  - FDR-corrected Spearman filter (BH q=0.20, fallback top-20) on train only
  - RFE with Ridge surrogate, CV-optimal k ∈ {5,10,15,20} on train only
  - Applied after feature alignment (column intersection of both corpora)

Usage:
  python 03_cross_corpus_video_regression.py
  python 03_cross_corpus_video_regression.py --phase emotion_induction_1 --condition ADK
  python 03_cross_corpus_video_regression.py --experiments A B
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from shared_regression_pipeline import (
from config import RESULTS_ROOT, EDAIC_AGGREGATED_PATH
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    align_features,
    evaluate_regressor,
    get_param_grids,
    get_regressors,
    get_scalers,
    grid_search_regressor,
    load_edaic_data,
    load_proposed_data,
    load_shared_split,
    print_metrics,
    rfe_select_regression,
    spearman_select_fdr,
)

if CATBOOST_AVAILABLE:
    from catboost import CatBoostRegressor
if LIGHTGBM_AVAILABLE:
    from lightgbm import LGBMRegressor

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_BASE = (
    RESULTS_ROOT
    "KDD_paper/cross_corpus_regression"
)

# Pipeline flags — FDR+RFE feature selection (mirrors repeated_holdout_v2)
FDR_Q          = 0.20
FDR_FALLBACK_N = 20
RFE_K_RANGE    = (5, 10, 15, 20)
TARGET_SCALERS = ["StandardScaler", "RobustScaler"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def select_features(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame):
    """
    FDR-corrected Spearman → RFE (Ridge surrogate) on train only.
    Returns (X_train_selected, X_test_selected, kept_feature_names).
    """
    fdr_feats = spearman_select_fdr(X_train, y_train,
                                    q=FDR_Q, fallback_top_n=FDR_FALLBACK_N)
    kept = rfe_select_regression(X_train[fdr_feats], y_train,
                                 num_features_range=RFE_K_RANGE)
    if not kept:
        kept = fdr_feats
    return X_train[kept], X_test[kept], kept


def fit_and_eval(models, param_grids, scalers,
                 X_train_imp: pd.DataFrame, y_train: pd.Series,
                 X_test_imp: pd.DataFrame, y_test: pd.Series,
                 experiment_tag: str, save_dir: str, timestamp: str):
    """
    Run all scaler × model combos. Returns (results_df, best_model_info_dict).
    """
    all_results  = {}
    all_trained  = {}
    all_features = {}

    for scaler_name in TARGET_SCALERS:
        scaler_obj = scalers[scaler_name]
        X_tr_sc = pd.DataFrame(
            scaler_obj.fit_transform(X_train_imp),
            columns=X_train_imp.columns, index=X_train_imp.index
        )
        X_te_sc = pd.DataFrame(
            scaler_obj.transform(X_test_imp),
            columns=X_test_imp.columns, index=X_test_imp.index
        )

        X_tr_fs, X_te_fs, kept = select_features(X_tr_sc, y_train, X_te_sc)

        for model_name, model in models.items():
            key = (model_name, scaler_name)
            print(f"  {model_name} | {scaler_name}")

            if model_name in param_grids:
                fitted = grid_search_regressor(
                    clone(model), param_grids[model_name],
                    X_tr_fs, y_train, cv=5
                )
            else:
                fitted = clone(model)
                fitted.fit(X_tr_fs, y_train)

            metrics = evaluate_regressor(fitted, X_te_fs, y_test)
            all_results[key]  = metrics
            all_trained[key]  = fitted
            all_features[key] = kept
            print_metrics(metrics, label=f"{model_name}/{scaler_name}")

    if not all_results:
        return pd.DataFrame(), {}

    rows = [{"model": m, "scaler": s, **v}
            for (m, s), v in all_results.items()]
    results_df = pd.DataFrame(rows).sort_values("MAE")

    best_key  = min(all_results, key=lambda k: all_results[k]["MAE"])
    bm, bs    = best_key
    print(f"\n  >> Best [{experiment_tag}]: {bm}/{bs}  "
          f"MAE={all_results[best_key]['MAE']:.3f}  "
          f"Pearson r={all_results[best_key]['Pearson_r']:.3f}  "
          f"R²={all_results[best_key]['R2']:.3f}")

    csv_path = os.path.join(
        save_dir, f"crosscorpus_{experiment_tag}_{timestamp}.csv"
    )
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    model_path = os.path.join(
        save_dir, f"best_model_{experiment_tag}_{timestamp}.joblib"
    )
    dump({
        "model":    all_trained[best_key],
        "features": all_features[best_key],
        "metrics":  all_results[best_key],
        "tag":      experiment_tag,
    }, model_path)

    return results_df, {
        "model_name": bm, "scaler": bs,
        "metrics": all_results[best_key],
        "features": all_features[best_key],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_cross_corpus(phase: str, condition: str, experiments: list):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir  = os.path.join(OUTPUT_BASE, f"{phase}_{condition}")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print(f"  CROSS-CORPUS PHQ-8 REGRESSION")
    print(f"  ProposedCorpus phase: {phase} | condition: {condition}")
    print(f"  Experiments: {experiments}")
    print("=" * 80)

    models      = get_regressors()
    param_grids = get_param_grids()
    scalers     = get_scalers()

    # ── Load ProposedCorpus data ─────────────────────────────────────────────────────
    print("\n[1] Loading ProposedCorpus data …")
    X_emp, y_emp, ids_emp = load_proposed_data(phase, condition)

    split_tag = "ALL" if condition == "ALL_CONDITIONS" else condition
    try:
        train_ids_emp, test_ids_emp = load_shared_split(split_tag)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    available_emp   = set(ids_emp.index)
    emp_train_idx   = sorted(train_ids_emp & available_emp)
    emp_test_idx    = sorted(test_ids_emp  & available_emp)

    X_emp_train_raw = X_emp.loc[X_emp.index.isin(emp_train_idx)]
    X_emp_test_raw  = X_emp.loc[X_emp.index.isin(emp_test_idx)]
    y_emp_train     = y_emp.loc[y_emp.index.isin(emp_train_idx)]
    y_emp_test      = y_emp.loc[y_emp.index.isin(emp_test_idx)]

    print(f"  ProposedCorpus train: n={len(y_emp_train)}  "
          f"PHQ-8 mean={y_emp_train.mean():.1f}")
    print(f"  ProposedCorpus test:  n={len(y_emp_test)}  "
          f"PHQ-8 mean={y_emp_test.mean():.1f}")

    # ── Load E-DAIC data ──────────────────────────────────────────────────────
    print("\n[2] Loading E-DAIC data …")
    try:
        X_daic_all, y_daic_all, ids_daic, split_col = load_edaic_data(
            split_filter=None  # load all splits
        )
    except FileNotFoundError:
        print(f"  ERROR: E-DAIC aggregated features not found at:\n"
              f"  {EDAIC_AGGREGATED_PATH}\n"
              f"  Run 01_aggregate_edaic_video_features.py first!")
        return

    X_daic_train_raw = X_daic_all[split_col.isin(["train", "dev"])]
    y_daic_train     = y_daic_all[split_col.isin(["train", "dev"])]
    X_daic_test_raw  = X_daic_all[split_col == "test"]
    y_daic_test      = y_daic_all[split_col == "test"]

    print(f"  E-DAIC train+dev: n={len(y_daic_train)}  "
          f"PHQ-8 mean={y_daic_train.mean():.1f}")
    print(f"  E-DAIC test:      n={len(y_daic_test)}  "
          f"PHQ-8 mean={y_daic_test.mean():.1f}")

    # ── Summary table accumulator ─────────────────────────────────────────────
    summary_rows = []

    # ═════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A: Train ProposedCorpus (ALL) → Test E-DAIC test set
    # ═════════════════════════════════════════════════════════════════════════
    if "A" in experiments:
        print(f"\n{'─'*80}")
        print(f"  EXPERIMENT A: ProposedCorpus (ALL) → E-DAIC test")
        print(f"  Transfers structured task biomarkers → clinical interview")
        print(f"{'─'*80}")

        # Align features
        X_emp_all_aligned, X_daic_test_aligned = align_features(
            X_emp, X_daic_test_raw
        )
        y_emp_all = y_emp  # use all ProposedCorpus as training

        # Impute (fit on ProposedCorpus ALL)
        imputer_A = SimpleImputer(strategy="median")
        X_emp_imp = pd.DataFrame(
            imputer_A.fit_transform(X_emp_all_aligned),
            columns=X_emp_all_aligned.columns, index=X_emp_all_aligned.index
        ).astype(np.float32)
        X_daic_test_imp = pd.DataFrame(
            imputer_A.transform(X_daic_test_aligned),
            columns=X_daic_test_aligned.columns, index=X_daic_test_aligned.index
        ).astype(np.float32)

        res_df_A, best_A = fit_and_eval(
            models, param_grids, scalers,
            X_emp_imp, y_emp_all,
            X_daic_test_imp, y_daic_test,
            experiment_tag=f"A_emp2daic_{phase}_{condition}",
            save_dir=save_dir, timestamp=timestamp
        )
        if best_A:
            summary_rows.append({
                "Experiment": "A: ProposedCorpus→E-DAIC",
                "Train": f"ProposedCorpus all (n={len(y_emp_all)})",
                "Test":  f"E-DAIC test (n={len(y_daic_test)})",
                "Best_model": f"{best_A['model_name']}/{best_A['scaler']}",
                **best_A["metrics"],
            })

    # ═════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B: Train E-DAIC (train+dev) → Test ProposedCorpus test split
    # ═════════════════════════════════════════════════════════════════════════
    if "B" in experiments:
        print(f"\n{'─'*80}")
        print(f"  EXPERIMENT B: E-DAIC (train+dev) → ProposedCorpus test")
        print(f"  Tests which ProposedCorpus phase interview-trained model works on")
        print(f"{'─'*80}")

        X_daic_tr_aligned, X_emp_test_aligned = align_features(
            X_daic_train_raw, X_emp_test_raw
        )
        y_daic_tr = y_daic_train

        imputer_B = SimpleImputer(strategy="median")
        X_daic_tr_imp = pd.DataFrame(
            imputer_B.fit_transform(X_daic_tr_aligned),
            columns=X_daic_tr_aligned.columns, index=X_daic_tr_aligned.index
        ).astype(np.float32)
        X_emp_test_imp = pd.DataFrame(
            imputer_B.transform(X_emp_test_aligned),
            columns=X_emp_test_aligned.columns, index=X_emp_test_aligned.index
        ).astype(np.float32)

        res_df_B, best_B = fit_and_eval(
            models, param_grids, scalers,
            X_daic_tr_imp, y_daic_tr,
            X_emp_test_imp, y_emp_test,
            experiment_tag=f"B_daic2emp_{phase}_{condition}",
            save_dir=save_dir, timestamp=timestamp
        )
        if best_B:
            summary_rows.append({
                "Experiment": "B: E-DAIC→ProposedCorpus",
                "Train": f"E-DAIC train+dev (n={len(y_daic_tr)})",
                "Test":  f"ProposedCorpus test (n={len(y_emp_test)})",
                "Best_model": f"{best_B['model_name']}/{best_B['scaler']}",
                **best_B["metrics"],
            })

    # ═════════════════════════════════════════════════════════════════════════
    # EXPERIMENT C: Leave-one-corpus-out (pooled)
    # ═════════════════════════════════════════════════════════════════════════
    if "C" in experiments:
        print(f"\n{'─'*80}")
        print(f"  EXPERIMENT C: Leave-one-corpus-out (ProposedCorpus train→DAIC test "
              f"+ DAIC train→ProposedCorpus test, average)")
        print(f"{'─'*80}")

        # Both directions with the official splits
        X_emp_tr_aligned, X_daic_te_aligned = align_features(
            X_emp_train_raw, X_daic_test_raw
        )
        y_emp_tr = y_emp_train

        imputer_C1 = SimpleImputer(strategy="median")
        X_emp_tr_imp = pd.DataFrame(
            imputer_C1.fit_transform(X_emp_tr_aligned),
            columns=X_emp_tr_aligned.columns
        ).astype(np.float32)
        X_daic_te_imp = pd.DataFrame(
            imputer_C1.transform(X_daic_te_aligned),
            columns=X_daic_te_aligned.columns
        ).astype(np.float32)

        print("\n  C1: ProposedCorpus train → E-DAIC test")
        _, best_C1 = fit_and_eval(
            models, param_grids, scalers,
            X_emp_tr_imp, y_emp_tr,
            X_daic_te_imp, y_daic_test,
            experiment_tag=f"C1_emptr2daicte_{phase}_{condition}",
            save_dir=save_dir, timestamp=timestamp
        )
        if best_C1:
            summary_rows.append({
                "Experiment": "C1: ProposedCorpus-train→E-DAIC-test",
                "Train": f"ProposedCorpus train (n={len(y_emp_tr)})",
                "Test":  f"E-DAIC test (n={len(y_daic_test)})",
                "Best_model": f"{best_C1['model_name']}/{best_C1['scaler']}",
                **best_C1["metrics"],
            })

        X_daic_tr2_aligned, X_emp_te_aligned = align_features(
            X_daic_train_raw, X_emp_test_raw
        )

        imputer_C2 = SimpleImputer(strategy="median")
        X_daic_tr2_imp = pd.DataFrame(
            imputer_C2.fit_transform(X_daic_tr2_aligned),
            columns=X_daic_tr2_aligned.columns
        ).astype(np.float32)
        X_emp_te_imp = pd.DataFrame(
            imputer_C2.transform(X_emp_te_aligned),
            columns=X_emp_te_aligned.columns
        ).astype(np.float32)

        print("\n  C2: E-DAIC train+dev → ProposedCorpus test")
        _, best_C2 = fit_and_eval(
            models, param_grids, scalers,
            X_daic_tr2_imp, y_daic_train,
            X_emp_te_imp, y_emp_test,
            experiment_tag=f"C2_daictr2empte_{phase}_{condition}",
            save_dir=save_dir, timestamp=timestamp
        )
        if best_C2:
            summary_rows.append({
                "Experiment": "C2: E-DAIC-train→ProposedCorpus-test",
                "Train": f"E-DAIC train+dev (n={len(y_daic_train)})",
                "Test":  f"ProposedCorpus test (n={len(y_emp_test)})",
                "Best_model": f"{best_C2['model_name']}/{best_C2['scaler']}",
                **best_C2["metrics"],
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(
            save_dir, f"SUMMARY_{phase}_{condition}_{timestamp}.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}")
        print("  CROSS-CORPUS SUMMARY")
        print(f"{'='*80}")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(summary_df[["Experiment", "Best_model",
                           "MAE", "RMSE", "R2",
                           "Pearson_r", "Spearman_r"]].to_string(index=False))
        print(f"\n  Summary saved: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-corpus PHQ-8 regression: ProposedCorpus ↔ E-DAIC"
    )
    parser.add_argument(
        "--phase", default="emotion_induction_1",
        choices=["latency", "emotion_induction_1",
                 "negative_training", "positive_training"],
        help="ProposedCorpus phase to use as source (default: emotion_induction_1)"
    )
    parser.add_argument(
        "--condition", default="ADK",
        choices=["ADK", "CR", "CRADK", "SHAM", "ALL_CONDITIONS"],
        help="ProposedCorpus condition to use (default: ADK)"
    )
    parser.add_argument(
        "--experiments", nargs="+", default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="Which experiments to run (default: A B C)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cross_corpus(
        phase=args.phase,
        condition=args.condition,
        experiments=args.experiments
    )

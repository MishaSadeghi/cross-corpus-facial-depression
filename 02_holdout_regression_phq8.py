"""
02_holdout_regression_phq8.py
==============================
PHQ-8 score regression on EmpkinS RCT video features — fixed holdout evaluation.
Mirrors the classification holdout_v2.py pipeline but predicts continuous PHQ-8.

Pipeline (no data leakage):
  1. Load pre-aggregated EmpkinS video features for a phase.
  2. Filter to participants in the SHARED train/test split.
  3. Imputer    : fit on X_train, transform X_test.
  4. Scaler     : fit on X_train, transform X_test.
  5. Feature selection (Spearman |r| top 50) on X_train only.
  6. Pearson redundancy pruning (r >= 0.90) on X_train only.
  7. GridSearchCV (5-fold, neg_MAE) on X_train only.
  8. Evaluate on X_test → MAE, RMSE, R², Pearson r, Spearman ρ.
  9. Save best model + results CSV.

Usage:
  python 02_holdout_regression_phq8.py --phase positive_training
  python 02_holdout_regression_phq8.py --phase latency --conditions ADK CR CRADK SHAM
  python 02_holdout_regression_phq8.py --phase positive_training --conditions ALL_CONDITIONS
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
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    ALL_CONDITIONS,
    align_features,
    evaluate_regressor,
    get_param_grids,
    get_regressors,
    get_scalers,
    grid_search_regressor,
    load_empkins_data,
    load_shared_split,
    pearson_redundancy_prune,
    print_metrics,
    spearman_select,
)

if CATBOOST_AVAILABLE:
    from catboost import CatBoostRegressor
if LIGHTGBM_AVAILABLE:
    from lightgbm import LGBMRegressor

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_BASE = (
    "/home/woody/empk/empk004h/D02_dataset/D02_final_results_and_models/"
    "KDD_paper/regression_phq8"
)

# Pipeline flags
SPEARMAN_TOP_N     = 50    # top features by |Spearman r| with PHQ-8
PEARSON_THRESHOLD  = 0.90  # redundancy pruning threshold
PERFORM_GRID_SEARCH = True
TARGET_SCALERS      = ["StandardScaler", "MinMaxScaler", "RobustScaler"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HOLDOUT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_regression_holdout(phase: str, conditions: list):
    save_dir = os.path.join(OUTPUT_BASE, phase)
    os.makedirs(save_dir, exist_ok=True)

    models      = get_regressors()
    param_grids = get_param_grids()
    scalers     = get_scalers()

    for condition in conditions:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n{'#'*80}")
        print(f"  PHQ-8 REGRESSION HOLDOUT | Phase: {phase.upper()} | "
              f"Condition: {condition.upper()}")
        print(f"{'#'*80}")

        # ── 1. Load shared split ──────────────────────────────────────────────
        split_tag = "ALL" if condition == "ALL_CONDITIONS" else condition
        try:
            train_ids, test_ids = load_shared_split(split_tag)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        # ── 2. Load EmpkinS features (PHQ-8 as target) ───────────────────────
        X_full, y_full, subject_ids = load_empkins_data(phase, condition)
        if X_full.empty:
            print(f"  Skipping {condition}: no data.")
            continue

        # ── 3. Filter to shared split participants ────────────────────────────
        available   = set(subject_ids.index)
        phase_train = sorted(train_ids & available)
        phase_test  = sorted(test_ids  & available)

        missing_tr = train_ids - available
        missing_te = test_ids  - available
        if missing_tr:
            print(f"  NOTE: {len(missing_tr)} train IDs absent from this phase.")
        if missing_te:
            print(f"  NOTE: {len(missing_te)} test IDs absent from this phase.")

        X_train_raw = X_full.loc[X_full.index.isin(phase_train)]
        X_test_raw  = X_full.loc[X_full.index.isin(phase_test)]
        y_train     = y_full.loc[y_full.index.isin(phase_train)]
        y_test      = y_full.loc[y_full.index.isin(phase_test)]

        print(f"  Train: n={len(y_train)}  "
              f"PHQ-8 mean={y_train.mean():.1f} std={y_train.std():.1f}")
        print(f"  Test:  n={len(y_test)}  "
              f"PHQ-8 mean={y_test.mean():.1f} std={y_test.std():.1f}")

        if len(y_train) < 10 or len(y_test) < 5:
            print(f"  Skipping {condition}: too few samples.")
            continue

        # ── 4. Drop all-NaN columns (train), then impute ──────────────────────
        # SimpleImputer silently drops columns that are all-NaN in train,
        # causing a shape mismatch when transforming test. Pre-filter to avoid.
        valid_cols  = X_train_raw.columns[~X_train_raw.isna().all(axis=0).values]
        X_train_raw = X_train_raw[valid_cols]
        X_test_raw  = X_test_raw[valid_cols]

        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train_raw),
            columns=valid_cols, index=X_train_raw.index
        ).astype(np.float32)
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test_raw),
            columns=valid_cols, index=X_test_raw.index
        ).astype(np.float32)

        # ── 5. Loop scalers × models ─────────────────────────────────────────
        all_results  = {}
        all_trained  = {}
        all_features = {}

        for scaler_name in TARGET_SCALERS:
            print(f"\n{'='*60}")
            print(f"  Scaler: {scaler_name}")
            print(f"{'='*60}")

            scaler_obj = scalers[scaler_name]
            X_train_sc = pd.DataFrame(
                scaler_obj.fit_transform(X_train_imp),
                columns=X_train_imp.columns, index=X_train_imp.index
            )
            X_test_sc = pd.DataFrame(
                scaler_obj.transform(X_test_imp),
                columns=X_test_imp.columns, index=X_test_imp.index
            )

            # Feature selection on TRAIN only
            top_feats = spearman_select(X_train_sc, y_train,
                                        top_n=SPEARMAN_TOP_N)
            X_tr_sel  = X_train_sc[top_feats]
            kept      = pearson_redundancy_prune(X_tr_sel,
                                                 threshold=PEARSON_THRESHOLD)
            if not kept:
                kept = top_feats

            X_tr_fs  = X_train_sc[kept]
            X_te_fs  = X_test_sc[kept]

            for model_name, model in models.items():
                print(f"\n  --- {model_name} | {scaler_name} ---")
                key = (model_name, scaler_name)

                # Grid search on TRAIN
                if PERFORM_GRID_SEARCH and model_name in param_grids:
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
            print(f"  No results for {condition}.")
            continue

        # ── 6. Best model by MAE ──────────────────────────────────────────────
        best_key = min(all_results, key=lambda k: all_results[k]["MAE"])
        bm, bs   = best_key
        print(f"\n  Best: model={bm}, scaler={bs}  "
              f"MAE={all_results[best_key]['MAE']:.3f}  "
              f"Pearson r={all_results[best_key]['Pearson_r']:.3f}")

        # ── 7. Save results CSV ───────────────────────────────────────────────
        rows = []
        for (mname, sname), metrics in all_results.items():
            rows.append({"model": mname, "scaler": sname, **metrics})
        results_df = pd.DataFrame(rows).sort_values("MAE")

        csv_path = os.path.join(
            save_dir, f"regression_{phase}_{condition}_{timestamp}.csv"
        )
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Results saved: {csv_path}")

        # Print top-5 summary
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(results_df[["model", "scaler", "MAE", "RMSE",
                           "R2", "Pearson_r", "Spearman_r"]].head(5).to_string(
            index=False
        ))

        # ── 8. Save best model ────────────────────────────────────────────────
        model_path = os.path.join(
            save_dir,
            f"best_model_{phase}_{condition}_{timestamp}.joblib"
        )
        dump({
            "model":     all_trained[best_key],
            "features":  all_features[best_key],
            "phase":     phase,
            "condition": condition,
            "metrics":   all_results[best_key],
        }, model_path)
        print(f"  Best model saved: {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="PHQ-8 regression holdout on EmpkinS video features"
    )
    parser.add_argument(
        "--phase", required=True,
        choices=["latency", "emotion_induction_1",
                 "negative_training", "positive_training"],
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Conditions to run. Defaults: ALL_CONDITIONS + per-condition for "
             "latency/EI1; per-condition only for training phases."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.conditions is None:
        if args.phase in ("positive_training", "negative_training"):
            args.conditions = ALL_CONDITIONS            # per-condition only
        else:
            args.conditions = ALL_CONDITIONS + ["ALL_CONDITIONS"]
    run_regression_holdout(args.phase, args.conditions)

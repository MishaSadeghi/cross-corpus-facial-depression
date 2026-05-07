"""
04_repeated_holdout_regression.py
===================================
Repeated holdout regression for PHQ-8 — addresses the lucky-split problem.

Why this is needed
------------------
Each RCT condition has ~64 participants (between-subjects design, completely
different people per condition). An 80/20 split leaves only ~13 test subjects.
With n=13, one or two outlier PHQ-8 scores in the test set can shift MAE by
1–2 points, making a condition look better or worse purely by chance.

Solution: repeat the 80/20 stratified split K times with different random seeds.
Report mean ± std over seeds. This:
  - Quantifies split variance (how much results vary by luck)
  - Gives a reliable point estimate (mean across seeds)
  - Stays within holdout framework (NOT nested CV)
  - Lets you compare conditions fairly: if CR MAE is lower than ADK across
    ALL seeds, it's a real difference; if it's only lower on some seeds, it's luck.

Stratification: stratified by binary SCID label (HC=0 / dep=1) so that each
split preserves the 50/50 class balance within each condition.

Pipeline per seed (improved feature selection — v2):
  1. Stratified 80/20 split by SCID label
  2. Drop all-NaN train columns
  3. Impute (median, fit on train)
  4. Scale (fit on train)
  5. FDR-corrected Spearman filter (BH q=0.20, fallback top-20) — train only
  6. RFE with Ridge surrogate, CV-optimal k ∈ {5,10,15,20} — train only
  7. GridSearchCV 5-fold (train only)
  8. Evaluate on test

Output per phase/condition:
  - Per-seed results CSV (one row per model × scaler × seed)
  - Summary CSV: mean ± std of MAE and Pearson r across seeds, per model
  - Best-model summary across all conditions

Usage:
  python 04_repeated_holdout_regression.py --phase positive_training
  python 04_repeated_holdout_regression.py --phase emotion_induction_1 --n_seeds 5
  python 04_repeated_holdout_regression.py --phase latency --conditions ADK CR CRADK SHAM
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
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from shared_regression_pipeline import (
from config import RESULTS_ROOT
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    ALL_CONDITIONS,
    evaluate_regressor,
    get_param_grids,
    get_regressors,
    get_scalers,
    grid_search_regressor,
    load_proposed_data,
    print_metrics,
    rfe_select_regression,
    spearman_select_fdr,
    MASTER_DATA_PATH,
    normalize_id,
)

if CATBOOST_AVAILABLE:
    from catboost import CatBoostRegressor
if LIGHTGBM_AVAILABLE:
    from lightgbm import LGBMRegressor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_BASE       = (
    RESULTS_ROOT
    "KDD_paper/repeated_holdout_phq8_v2"
)
TEST_SIZE         = 0.20   # 80/20 split
TARGET_SCALERS    = ["StandardScaler", "RobustScaler"]
PERFORM_GRID_SEARCH = True
# Feature selection: FDR-corrected Spearman → RFE (Ridge surrogate)
FDR_Q             = 0.20   # Benjamini-Hochberg threshold
FDR_FALLBACK_N    = 20     # fallback top-N if <5 survive FDR
RFE_K_RANGE       = (5, 10, 15, 20)  # candidate feature counts for RFE


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_repeated_holdout(phase: str, conditions: list, n_seeds: int):
    save_dir = os.path.join(OUTPUT_BASE, phase)
    os.makedirs(save_dir, exist_ok=True)

    models      = get_regressors()
    param_grids = get_param_grids()
    scalers     = get_scalers()

    # Load SCID binary label for stratification (separate from PHQ-8 target)
    master_df = pd.read_csv(MASTER_DATA_PATH)
    master_df["ID"] = master_df["ID"].apply(normalize_id)

    all_condition_summaries = []

    for condition in conditions:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n{'#'*80}")
        print(f"  REPEATED HOLDOUT PHQ-8 | Phase: {phase.upper()} | "
              f"Condition: {condition.upper()} | Seeds: {n_seeds}")
        print(f"{'#'*80}")

        # Load features and PHQ-8 target
        X_full, y_full, subject_ids = load_proposed_data(phase, condition)
        if X_full.empty:
            print(f"  Skipping {condition}: no data.")
            continue

        # Get unique participant IDs (index may repeat for multi-session phases)
        unique_ids = X_full.index.unique()

        # Merge SCID label for stratification
        if condition == "ALL_CONDITIONS":
            master_filt = master_df.copy()
        else:
            master_filt = master_df[master_df["condition"] == condition]

        label_map = master_filt.set_index("ID")["label"]

        # Work at the PARTICIPANT level for splitting,
        # then expand back to all sessions for train/test
        # (prevents participant-level data leakage across sessions)
        pid_labels = label_map.reindex(unique_ids).dropna()
        valid_pids = pid_labels.index.tolist()
        pid_arr    = np.array(valid_pids)
        lbl_arr    = pid_labels.values

        if len(valid_pids) < 20:
            print(f"  Skipping {condition}: only {len(valid_pids)} participants.")
            continue

        print(f"  Participants: {len(valid_pids)}  "
              f"(HC={int((lbl_arr==0).sum())}, dep={int((lbl_arr==1).sum())})")
        print(f"  PHQ-8: mean={y_full.mean():.1f}, std={y_full.std():.1f}")

        all_seed_rows = []
        seeds = list(range(n_seeds))

        for seed in seeds:
            print(f"\n  -- Seed {seed+1}/{n_seeds} --")

            # Split at participant level
            sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                          random_state=seed)
            train_pids_idx, test_pids_idx = list(
                sss.split(pid_arr, lbl_arr)
            )[0]
            train_pids = set(pid_arr[train_pids_idx])
            test_pids  = set(pid_arr[test_pids_idx])

            # Expand to session level
            X_cond_train = X_full[X_full.index.isin(train_pids)]
            X_cond_test  = X_full[X_full.index.isin(test_pids)]
            y_train      = y_full[y_full.index.isin(train_pids)]
            y_test       = y_full[y_full.index.isin(test_pids)]

            print(f"    Train: {len(train_pids)} participants / "
                  f"{len(y_train)} sessions  "
                  f"PHQ-8 mean={y_train.mean():.1f}")
            print(f"    Test:  {len(test_pids)} participants / "
                  f"{len(y_test)} sessions  "
                  f"PHQ-8 mean={y_test.mean():.1f}")

            # Drop all-NaN train columns
            valid_cols  = X_cond_train.columns[
                ~X_cond_train.isna().all(axis=0).values
            ]
            X_cond_train = X_cond_train[valid_cols]
            X_cond_test  = X_cond_test[valid_cols]

            # Impute
            imputer = SimpleImputer(strategy="median")
            X_tr_imp = pd.DataFrame(
                imputer.fit_transform(X_cond_train),
                columns=valid_cols, index=X_cond_train.index
            ).astype(np.float32)
            X_te_imp = pd.DataFrame(
                imputer.transform(X_cond_test),
                columns=valid_cols, index=X_cond_test.index
            ).astype(np.float32)

            for scaler_name in TARGET_SCALERS:
                scaler_obj = scalers[scaler_name]
                X_tr_sc = pd.DataFrame(
                    scaler_obj.fit_transform(X_tr_imp),
                    columns=valid_cols, index=X_tr_imp.index
                )
                X_te_sc = pd.DataFrame(
                    scaler_obj.transform(X_te_imp),
                    columns=valid_cols, index=X_te_imp.index
                )

                fdr_feats = spearman_select_fdr(
                    X_tr_sc, y_train,
                    q=FDR_Q, fallback_top_n=FDR_FALLBACK_N
                )
                kept = rfe_select_regression(
                    X_tr_sc[fdr_feats], y_train,
                    num_features_range=RFE_K_RANGE
                )
                if not kept:
                    kept = fdr_feats

                X_tr_fs = X_tr_sc[kept]
                X_te_fs = X_te_sc[kept]

                for model_name, model in models.items():
                    if PERFORM_GRID_SEARCH and model_name in param_grids:
                        fitted = grid_search_regressor(
                            clone(model), param_grids[model_name],
                            X_tr_fs, y_train, cv=5
                        )
                    else:
                        fitted = clone(model)
                        fitted.fit(X_tr_fs, y_train)

                    metrics = evaluate_regressor(fitted, X_te_fs, y_test)
                    print_metrics(metrics,
                                  label=f"seed={seed} {model_name}/{scaler_name}")

                    all_seed_rows.append({
                        "seed":      seed,
                        "model":     model_name,
                        "scaler":    scaler_name,
                        "n_train_participants": len(train_pids),
                        "n_test_participants":  len(test_pids),
                        "n_train_sessions":     len(y_train),
                        "n_test_sessions":      len(y_test),
                        "y_test_mean": float(y_test.mean()),
                        "y_test_std":  float(y_test.std()),
                        **metrics,
                    })

        if not all_seed_rows:
            continue

        # ── Save per-seed results ─────────────────────────────────────────────
        per_seed_df = pd.DataFrame(all_seed_rows)
        per_seed_path = os.path.join(
            save_dir,
            f"per_seed_{phase}_{condition}_{timestamp}.csv"
        )
        per_seed_df.to_csv(per_seed_path, index=False)

        # ── Summary: mean ± std across seeds per (model, scaler) ─────────────
        metric_cols = ["MAE", "RMSE", "R2", "Pearson_r", "Spearman_r"]
        summary_rows = []
        for (mname, sname), grp in per_seed_df.groupby(["model", "scaler"]):
            row = {"model": mname, "scaler": sname, "n_seeds": len(grp)}
            for m in metric_cols:
                row[f"{m}_mean"] = round(grp[m].mean(), 4)
                row[f"{m}_std"]  = round(grp[m].std(),  4)
                row[f"{m}_min"]  = round(grp[m].min(),  4)
                row[f"{m}_max"]  = round(grp[m].max(),  4)
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows).sort_values("MAE_mean")
        summary_path = os.path.join(
            save_dir,
            f"summary_{phase}_{condition}_{timestamp}.csv"
        )
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*80}")
        print(f"  SUMMARY | {phase.upper()} | {condition.upper()} "
              f"(mean ± std over {n_seeds} seeds)")
        print(f"{'='*80}")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(summary_df[["model", "scaler",
                           "MAE_mean", "MAE_std",
                           "Pearson_r_mean", "Pearson_r_std",
                           "R2_mean", "R2_std"]].head(5).to_string(index=False))
        print(f"\n  Per-seed results : {per_seed_path}")
        print(f"  Summary          : {summary_path}")

        best = summary_df.iloc[0]
        all_condition_summaries.append({
            "phase":      phase,
            "condition":  condition,
            "best_model": f"{best['model']}/{best['scaler']}",
            "MAE_mean":   best["MAE_mean"],
            "MAE_std":    best["MAE_std"],
            "Pearson_r_mean": best["Pearson_r_mean"],
            "Pearson_r_std":  best["Pearson_r_std"],
            "R2_mean":    best["R2_mean"],
            "n_seeds":    n_seeds,
        })

    # ── Cross-condition comparison ────────────────────────────────────────────
    if all_condition_summaries:
        comp_df = pd.DataFrame(all_condition_summaries)
        comp_path = os.path.join(
            save_dir,
            f"condition_comparison_{phase}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        )
        comp_df.to_csv(comp_path, index=False)
        print(f"\n{'='*80}")
        print(f"  CONDITION COMPARISON | {phase.upper()} | best model per condition")
        print(f"  (If MAE_std is large relative to difference between conditions,")
        print(f"   the gap is likely due to split variance, not a real effect.)")
        print(f"{'='*80}")
        print(comp_df[["condition", "best_model",
                        "MAE_mean", "MAE_std",
                        "Pearson_r_mean", "Pearson_r_std"]].to_string(index=False))
        print(f"\n  Comparison saved: {comp_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeated holdout PHQ-8 regression (multiple random seeds)"
    )
    parser.add_argument(
        "--phase", required=True,
        choices=["latency", "emotion_induction_1",
                 "negative_training", "positive_training"],
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Conditions to run (default: all 4 + ALL_CONDITIONS for latency/EI1)"
    )
    parser.add_argument(
        "--n_seeds", type=int, default=5,
        help="Number of random seeds / repeated splits (default: 5)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.conditions is None:
        if args.phase in ("positive_training", "negative_training"):
            args.conditions = ALL_CONDITIONS
        else:
            args.conditions = ALL_CONDITIONS + ["ALL_CONDITIONS"]
    run_repeated_holdout(args.phase, args.conditions, args.n_seeds)

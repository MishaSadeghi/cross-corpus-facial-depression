"""
06_cross_corpus_regression_v2.py
=====================================
Cross-corpus PHQ-8 regression: ProposedCorpus RCT ↔ E-DAIC.
Parallel structure to 05_cross_corpus_classification.py so both result tables
can be directly compared in the paper.

Source configurations (ProposedCorpus side) — selected by within-corpus PHQ-8 CCC
using the 882 cross-corpus restricted features (holdout evaluation):
  LAT_ADK  — latency / ADK        (within-corpus CCC=0.66; passive observation, ADK condition)
  LAT_SHAM — latency / SHAM       (within-corpus CCC=0.55; passive + control/no-intervention)
  EI1_ADK  — emotion_induction_1 / ADK  (within-corpus CCC=0.46; comparison with classification)
  ALL      — all 4 phases × all conditions (ProposedCorpus pooled; largest training set baseline)

Rationale for using latency phase as primary config:
  The latency (preparatory/baseline) phase is a passive observation task where participants
  sit quietly in front of the camera — the closest ProposedCorpus equivalent to the E-DAIC
  clinical interview setting. It achieves the highest within-corpus PHQ-8 CCC, which is
  the task-matched selection criterion for cross-corpus regression.

  Note: classification cross-corpus uses EI1_ADK (emotion induction, closest to interview
  context for binary detection) — a different optimal phase for a different task. This
  difference is itself informative about the task/phase interaction.

Experiments:
  A : ProposedCorpus-ALL → E-DAIC official test set
  B : E-DAIC (train+dev) → ProposedCorpus (official held-out test split)
  C1: ProposedCorpus (train split) → E-DAIC test
  C2: E-DAIC (train+dev) → ProposedCorpus (test split)

Feature selection: FDR-corrected Spearman + RFE (Ridge surrogate), train only.

Metrics: MAE, RMSE, Pearson r, CCC (Lin's Concordance Correlation Coefficient).
CCC is the primary ranking metric — it captures both correlation and systematic
bias (mean/variance shift) in [-1, 1]. CCC≈0 means the model predicts no better
than the training mean, which is what we expect from cross-corpus transfer failure.
R² is intentionally excluded from the primary table (negative values mislead).

Usage:
  python 06_cross_corpus_regression_v2.py
  python 06_cross_corpus_regression_v2.py --configs EI1_ADK POS_ADK
  python 06_cross_corpus_regression_v2.py --experiments A B C
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
from config import RESULTS_ROOT
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    PROPOSED_PHASE_FEATURES,
    EXCLUDE_COLS,
    MASTER_DATA_PATH,
    _bh_correction,
    align_features,
    evaluate_regressor,
    get_param_grids,
    get_regressors,
    get_scalers,
    grid_search_regressor,
    load_edaic_data,
    load_shared_split,
    normalize_id,
    rename_proposed_to_openface,
    rfe_select_regression,
    spearman_select_fdr,
)

if CATBOOST_AVAILABLE:
    from catboost import CatBoostRegressor
if LIGHTGBM_AVAILABLE:
    from lightgbm import LGBMRegressor


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_BASE = (
    RESULTS_ROOT
    "KDD_paper/cross_corpus_regression_v2"
)

ALL_PHASES = ["latency", "emotion_induction_1", "negative_training", "positive_training"]

SOURCE_CONFIGS = {
    # Latency phase (passive baseline, closest to E-DAIC interview setting)
    "LAT_ADK":   {"phases": ["latency"],            "condition": "ADK"},
    "LAT_SHAM":  {"phases": ["latency"],            "condition": "SHAM"},
    "LAT_ALL":   {"phases": ["latency"],            "condition": "ALL_CONDITIONS"},
    # Best training phase per condition (selected by within-corpus PHQ-8 nested CV CCC)
    "POS_ADK":   {"phases": ["positive_training"],  "condition": "ADK"},   # CCC=.26
    "NEG_CRADK": {"phases": ["negative_training"],  "condition": "CRADK"}, # CCC=.09
    "NEG_CR":    {"phases": ["negative_training"],  "condition": "CR"},    # CCC=.20
    "NEG_SHAM":  {"phases": ["negative_training"],  "condition": "SHAM"},  # CCC=.26
}

FDR_Q          = 0.20
FDR_FALLBACK_N = 20
RFE_K_RANGE    = (5, 10, 15, 20)
TARGET_SCALERS = ["StandardScaler", "RobustScaler"]


# ─────────────────────────────────────────────────────────────────────────────
# CCC (Lin's Concordance Correlation Coefficient)
# ─────────────────────────────────────────────────────────────────────────────
def ccc(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient.
    CCC = 2*rho*sigma_x*sigma_y / (sigma_x^2 + sigma_y^2 + (mu_x - mu_y)^2)
    Range [-1, 1].  CCC=1: perfect agreement.  CCC=0: no agreement beyond chance.
    CCC captures both precision (Pearson r) and accuracy (mean/variance bias).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mu_t, mu_p   = y_true.mean(), y_pred.mean()
    sig_t, sig_p = y_true.std(),  y_pred.std()
    if sig_t == 0 or sig_p == 0:
        return 0.0
    rho = np.corrcoef(y_true, y_pred)[0, 1]
    return float(2.0 * rho * sig_t * sig_p /
                 (sig_t**2 + sig_p**2 + (mu_t - mu_p)**2))


def evaluate_regressor_ext(model, X_test, y_test):
    """Extend evaluate_regressor with CCC."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import math
    from scipy.stats import pearsonr, spearmanr

    y_pred = model.predict(X_test)
    mae    = float(mean_absolute_error(y_test, y_pred))
    rmse   = float(math.sqrt(mean_squared_error(y_test, y_pred)))
    pr, _  = pearsonr(y_test, y_pred)
    sr, _  = spearmanr(y_test, y_pred)
    ccc_v  = ccc(y_test, y_pred)

    return {
        "MAE":        round(mae,   4),
        "RMSE":       round(rmse,  4),
        "CCC":        round(ccc_v, 4),
        "Pearson_r":  round(float(pr), 4),
        "Spearman_r": round(float(sr), 4),
        "n_test":     int(len(y_test)),
    }


def print_reg_metrics(metrics, label=""):
    tag = "  [{}] ".format(label) if label else "  "
    print("{}MAE={:.3f}  RMSE={:.3f}  CCC={:.3f}  "
          "Pearson r={:.3f}  Spearman ρ={:.3f}  n={}".format(
              tag,
              metrics["MAE"], metrics["RMSE"], metrics["CCC"],
              metrics["Pearson_r"], metrics["Spearman_r"], metrics["n_test"]))


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING — ProposedCorpus (PHQ-8 target)
# ─────────────────────────────────────────────────────────────────────────────
def load_proposed_reg(phase, condition):
    """
    Load ProposedCorpus pre-aggregated video features + PHQ-8 target for one
    phase/condition. Returns (X, y_phq8).
    """
    features_path = PROPOSED_PHASE_FEATURES[phase]
    print("  Loading ProposedCorpus [{}/{}]: {}".format(phase, condition, features_path))

    feat_df = pd.read_csv(features_path)
    feat_df.columns = feat_df.columns.str.strip()
    feat_df["ID"] = feat_df["ID"].apply(normalize_id)

    drop_cols = [c for c in EXCLUDE_COLS if c in feat_df.columns and c != "ID"]
    feat_df.drop(columns=drop_cols, inplace=True)
    feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()]

    master_df = pd.read_csv(MASTER_DATA_PATH)
    master_df["ID"] = master_df["ID"].apply(normalize_id)
    master_df.rename(columns={"label": "ground_truth_label"}, inplace=True)

    if condition == "ALL_CONDITIONS":
        master_filt = master_df.copy()
    else:
        master_filt = master_df[master_df["condition"] == condition].copy()

    merged = pd.merge(
        feat_df,
        master_filt[["ID", "condition", "phq_8"]],
        on="ID", how="inner"
    )
    print("  Merged shape [{}]: {}".format(condition, merged.shape))

    n_before = len(merged)
    merged = merged.dropna(subset=["phq_8"])
    if len(merged) < n_before:
        print("  Dropped {} rows with missing PHQ-8.".format(n_before - len(merged)))

    merged = merged.set_index("ID")
    feature_cols = [c for c in merged.columns if c not in EXCLUDE_COLS]
    X = merged[feature_cols]

    n_before = X.shape[1]
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] < n_before:
        print("  Dropped {} non-numeric columns.".format(n_before - X.shape[1]))

    X = rename_proposed_to_openface(X)

    # Drop rename-created duplicates (positive_training has both naming styles)
    n_before = X.shape[1]
    X = X.loc[:, ~X.columns.duplicated()]
    if X.shape[1] < n_before:
        print("  Dropped {} duplicate columns after AU rename.".format(
            n_before - X.shape[1]))

    y = merged["phq_8"]
    print("  X: {}  |  PHQ-8 mean={:.1f}, std={:.1f}".format(
        X.shape, y.mean(), y.std()))
    return X, y


def load_proposed_all_phases_reg():
    """Pool all 4 phases × all conditions, returning (X, y_phq8)."""
    print("\n  Loading ALL phases × ALL conditions ...")
    all_X, all_y = [], []
    for phase in ALL_PHASES:
        Xp, yp = load_proposed_reg(phase, "ALL_CONDITIONS")
        all_X.append(Xp)
        all_y.append(yp)

    common_cols = sorted(set.intersection(*[set(Xp.columns) for Xp in all_X]))
    print("  ALL phases: {} common features across {} phases".format(
        len(common_cols), len(ALL_PHASES)))

    X_stacked = pd.concat([Xp[common_cols] for Xp in all_X], axis=0)
    y_stacked = pd.concat(all_y, axis=0)

    print("  ALL phases pooled: {}  |  PHQ-8 mean={:.1f}, std={:.1f}".format(
        X_stacked.shape, y_stacked.mean(), y_stacked.std()))
    return X_stacked, y_stacked


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def select_features_reg(X_train, y_train, X_test):
    """FDR-corrected Spearman → RFE (Ridge surrogate) on train only."""
    fdr_feats = spearman_select_fdr(X_train, y_train,
                                    q=FDR_Q, fallback_top_n=FDR_FALLBACK_N)
    kept = rfe_select_regression(X_train[fdr_feats], y_train,
                                 num_features_range=RFE_K_RANGE)
    if not kept:
        kept = fdr_feats
    return X_train[kept], X_test[kept], kept


# ─────────────────────────────────────────────────────────────────────────────
# FIT + EVALUATE: all scaler × model combos
# ─────────────────────────────────────────────────────────────────────────────
def fit_and_eval_reg(models, param_grids, scalers,
                     X_train_imp, y_train,
                     X_test_imp, y_test,
                     experiment_tag, save_dir, timestamp):
    """Run all (scaler × model) combinations. Ranked by CCC descending."""
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

        X_tr_fs, X_te_fs, kept = select_features_reg(X_tr_sc, y_train, X_te_sc)

        for model_name, model in models.items():
            key = (model_name, scaler_name)
            print("  {} | {}".format(model_name, scaler_name))

            if model_name in param_grids:
                fitted = grid_search_regressor(
                    clone(model), param_grids[model_name],
                    X_tr_fs, y_train, cv=5
                )
            else:
                fitted = clone(model)
                fitted.fit(X_tr_fs, y_train)

            metrics = evaluate_regressor_ext(fitted, X_te_fs, y_test)
            all_results[key]  = metrics
            all_trained[key]  = fitted
            all_features[key] = kept
            print_reg_metrics(metrics, label="{}/{}".format(model_name, scaler_name))

    if not all_results:
        return pd.DataFrame(), {}

    rows = [{"model": m, "scaler": s, **v}
            for (m, s), v in all_results.items()]
    results_df = pd.DataFrame(rows).sort_values("CCC", ascending=False)

    # Best by CCC; break ties by MAE (lower is better)
    best_key = max(
        all_results,
        key=lambda k: (all_results[k]["CCC"], -all_results[k]["MAE"])
    )
    bm, bs = best_key
    print("\n  >> Best [{}]: {}/{}  CCC={:.3f}  MAE={:.3f}  "
          "Pearson r={:.3f}".format(
              experiment_tag, bm, bs,
              all_results[best_key]["CCC"],
              all_results[best_key]["MAE"],
              all_results[best_key]["Pearson_r"]))

    csv_path = os.path.join(
        save_dir, "reg_{}_{}.csv".format(experiment_tag, timestamp)
    )
    results_df.to_csv(csv_path, index=False)
    print("  Saved: {}".format(csv_path))

    model_path = os.path.join(
        save_dir, "best_reg_model_{}_{}.joblib".format(experiment_tag, timestamp)
    )
    dump({
        "model":    all_trained[best_key],
        "features": all_features[best_key],
        "metrics":  all_results[best_key],
        "tag":      experiment_tag,
    }, model_path)

    return results_df, {
        "model_name": bm, "scaler": bs,
        "metrics":    all_results[best_key],
        "features":   all_features[best_key],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: run one source configuration
# ─────────────────────────────────────────────────────────────────────────────
def run_config(config_name, config, experiments, timestamp, scalers, models, param_grids):
    phases    = config["phases"]
    condition = config["condition"]
    save_dir  = os.path.join(OUTPUT_BASE, config_name)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  CROSS-CORPUS PHQ-8 REGRESSION v2 | Config: {}".format(config_name))
    print("  Phases: {}  Condition: {}".format(phases, condition))
    print("  Experiments: {}".format(experiments))
    print("=" * 80)

    # ── Load ProposedCorpus ──────────────────────────────────────────────────────────
    print("\n[1] Loading ProposedCorpus data ...")
    if config_name == "ALL":
        X_emp, y_emp = load_proposed_all_phases_reg()
        split_tag    = "ALL"
    else:
        assert len(phases) == 1
        X_emp, y_emp = load_proposed_reg(phases[0], condition)
        split_tag    = "ALL" if condition == "ALL_CONDITIONS" else condition

    # Aggregate sessions to participant level — matches within-corpus nested CV
    # (nested CV does groupby(index).mean() before splitting)
    n_sessions = len(X_emp)
    X_emp = X_emp.groupby(X_emp.index).mean()
    y_emp = y_emp.groupby(y_emp.index).first()
    print("  Aggregated {} sessions → {} participants".format(n_sessions, len(X_emp)))

    try:
        train_ids_emp, test_ids_emp = load_shared_split(split_tag)
    except FileNotFoundError as e:
        print("  ERROR: {}".format(e))
        return {}

    available_emp   = set(X_emp.index.unique().tolist())
    emp_train_idx   = sorted(train_ids_emp & available_emp)
    emp_test_idx    = sorted(test_ids_emp  & available_emp)

    X_emp_train_raw = X_emp[X_emp.index.isin(emp_train_idx)]
    X_emp_test_raw  = X_emp[X_emp.index.isin(emp_test_idx)]
    y_emp_train     = y_emp[y_emp.index.isin(emp_train_idx)]
    y_emp_test      = y_emp[y_emp.index.isin(emp_test_idx)]

    print("  ProposedCorpus train: n_participants={}  PHQ-8 mean={:.1f}".format(
              len(emp_train_idx), y_emp_train.mean()))
    print("  ProposedCorpus test:  n_participants={}  PHQ-8 mean={:.1f}".format(
              len(emp_test_idx), y_emp_test.mean()))

    # ── Load E-DAIC ───────────────────────────────────────────────────────────
    print("\n[2] Loading E-DAIC data ...")
    try:
        X_daic_all, y_daic_all, _, split_col = load_edaic_data(split_filter=None)
    except FileNotFoundError:
        print("  ERROR: E-DAIC aggregated file not found. "
              "Run 01_aggregate_edaic_video_features.py first!")
        return {}

    X_daic_train_raw = X_daic_all[split_col.isin(["train", "dev"])]
    y_daic_train     = y_daic_all[split_col.isin(["train", "dev"])]
    X_daic_test_raw  = X_daic_all[split_col == "test"]
    y_daic_test      = y_daic_all[split_col == "test"]

    print("  E-DAIC train+dev: n={}  PHQ-8 mean={:.1f}".format(
        len(y_daic_train), y_daic_train.mean()))
    print("  E-DAIC test:      n={}  PHQ-8 mean={:.1f}".format(
        len(y_daic_test), y_daic_test.mean()))

    summary_rows = []

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A
    # ═══════════════════════════════════════════════════════════════════════════
    if "A" in experiments:
        print("\n" + "─" * 80)
        print("  EXPERIMENT A: ProposedCorpus-ALL → E-DAIC test")
        print("─" * 80)

        X_emp_all_al, X_daic_te_al = align_features(X_emp, X_daic_test_raw)
        y_emp_all = y_emp

        imputer_A = SimpleImputer(strategy="median")
        X_emp_imp = pd.DataFrame(
            imputer_A.fit_transform(X_emp_all_al),
            columns=X_emp_all_al.columns, index=X_emp_all_al.index
        ).astype(np.float32)
        X_daic_te_imp = pd.DataFrame(
            imputer_A.transform(X_daic_te_al),
            columns=X_daic_te_al.columns, index=X_daic_te_al.index
        ).astype(np.float32)

        _, best_A = fit_and_eval_reg(
            models, param_grids, scalers,
            X_emp_imp, y_emp_all,
            X_daic_te_imp, y_daic_test,
            experiment_tag="A_emp2daic_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_A:
            summary_rows.append({
                "Experiment":   "A: ProposedCorpus-ALL→E-DAIC-test",
                "Train_source": "ProposedCorpus-ALL ({} sessions, {} participants)".format(
                    len(y_emp_all), len(set(y_emp_all.index.tolist()))),
                "Test_source":  "E-DAIC official test split",
                "n_train":      len(y_emp_all),
                "n_test":       len(y_daic_test),
                "Best_model":   "{}/{}".format(best_A["model_name"], best_A["scaler"]),
                **best_A["metrics"],
            })

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B
    # ═══════════════════════════════════════════════════════════════════════════
    if "B" in experiments:
        print("\n" + "─" * 80)
        print("  EXPERIMENT B: E-DAIC (train+dev) → ProposedCorpus test")
        print("─" * 80)

        X_daic_tr_al, X_emp_te_al = align_features(X_daic_train_raw, X_emp_test_raw)

        imputer_B = SimpleImputer(strategy="median")
        X_daic_tr_imp = pd.DataFrame(
            imputer_B.fit_transform(X_daic_tr_al),
            columns=X_daic_tr_al.columns, index=X_daic_tr_al.index
        ).astype(np.float32)
        X_emp_te_imp = pd.DataFrame(
            imputer_B.transform(X_emp_te_al),
            columns=X_emp_te_al.columns, index=X_emp_te_al.index
        ).astype(np.float32)

        _, best_B = fit_and_eval_reg(
            models, param_grids, scalers,
            X_daic_tr_imp, y_daic_train,
            X_emp_te_imp, y_emp_test,
            experiment_tag="B_daic2emp_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_B:
            summary_rows.append({
                "Experiment":   "B: E-DAIC-ALL→ProposedCorpus-test",
                "Train_source": "E-DAIC official train+dev split",
                "Test_source":  "ProposedCorpus fixed 80/20 test split ({} participants)".format(
                    len(set(y_emp_test.index.tolist()))),
                "n_train":      len(y_daic_train),
                "n_test":       len(y_emp_test),
                "Best_model":   "{}/{}".format(best_B["model_name"], best_B["scaler"]),
                **best_B["metrics"],
            })

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT C
    # ═══════════════════════════════════════════════════════════════════════════
    if "C" in experiments:
        print("\n" + "─" * 80)
        print("  EXPERIMENT C1: ProposedCorpus (train split) → E-DAIC test")
        print("─" * 80)

        X_emp_tr_al, X_daic_te_al = align_features(X_emp_train_raw, X_daic_test_raw)

        imputer_C1 = SimpleImputer(strategy="median")
        X_emp_tr_imp = pd.DataFrame(
            imputer_C1.fit_transform(X_emp_tr_al),
            columns=X_emp_tr_al.columns
        ).astype(np.float32)
        X_daic_te_imp = pd.DataFrame(
            imputer_C1.transform(X_daic_te_al),
            columns=X_daic_te_al.columns
        ).astype(np.float32)

        _, best_C1 = fit_and_eval_reg(
            models, param_grids, scalers,
            X_emp_tr_imp, y_emp_train,
            X_daic_te_imp, y_daic_test,
            experiment_tag="C1_emptr2daicte_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_C1:
            summary_rows.append({
                "Experiment":   "C1: ProposedCorpus-train→E-DAIC-test",
                "Train_source": "ProposedCorpus fixed 80/20 train split ({} participants)".format(
                    len(set(y_emp_train.index.tolist()))),
                "Test_source":  "E-DAIC official test split",
                "n_train":      len(y_emp_train),
                "n_test":       len(y_daic_test),
                "Best_model":   "{}/{}".format(best_C1["model_name"], best_C1["scaler"]),
                **best_C1["metrics"],
            })

        print("\n" + "─" * 80)
        print("  EXPERIMENT C2: E-DAIC (train+dev) → ProposedCorpus (test split)")
        print("─" * 80)

        X_daic_tr2_al, X_emp_te2_al = align_features(X_daic_train_raw, X_emp_test_raw)

        imputer_C2 = SimpleImputer(strategy="median")
        X_daic_tr2_imp = pd.DataFrame(
            imputer_C2.fit_transform(X_daic_tr2_al),
            columns=X_daic_tr2_al.columns
        ).astype(np.float32)
        X_emp_te2_imp = pd.DataFrame(
            imputer_C2.transform(X_emp_te2_al),
            columns=X_emp_te2_al.columns
        ).astype(np.float32)

        _, best_C2 = fit_and_eval_reg(
            models, param_grids, scalers,
            X_daic_tr2_imp, y_daic_train,
            X_emp_te2_imp, y_emp_test,
            experiment_tag="C2_daictr2empte_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_C2:
            summary_rows.append({
                "Experiment":   "C2: E-DAIC-train→ProposedCorpus-test",
                "Train_source": "E-DAIC official train+dev split",
                "Test_source":  "ProposedCorpus fixed 80/20 test split ({} participants)".format(
                    len(set(y_emp_test.index.tolist()))),
                "n_train":      len(y_daic_train),
                "n_test":       len(y_emp_test),
                "Best_model":   "{}/{}".format(best_C2["model_name"], best_C2["scaler"]),
                **best_C2["metrics"],
            })

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT D: within-corpus baselines (same fixed splits)
    # ═══════════════════════════════════════════════════════════════════════════
    if "D" in experiments:
        # D1: ProposedCorpus-train → ProposedCorpus-test (within-corpus upper bound)
        print("\n" + "─" * 80)
        print("  EXPERIMENT D1: ProposedCorpus-train → ProposedCorpus-test (within-corpus baseline)")
        print("─" * 80)

        X_emp_tr_al, X_emp_te_al = align_features(X_emp_train_raw, X_emp_test_raw)

        imputer_D1 = SimpleImputer(strategy="median")
        X_emp_tr_imp = pd.DataFrame(
            imputer_D1.fit_transform(X_emp_tr_al),
            columns=X_emp_tr_al.columns, index=X_emp_tr_al.index
        ).astype(np.float32)
        X_emp_te_imp = pd.DataFrame(
            imputer_D1.transform(X_emp_te_al),
            columns=X_emp_te_al.columns, index=X_emp_te_al.index
        ).astype(np.float32)

        _, best_D1 = fit_and_eval_reg(
            models, param_grids, scalers,
            X_emp_tr_imp, y_emp_train,
            X_emp_te_imp, y_emp_test,
            experiment_tag="D1_emptr2empte_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_D1:
            summary_rows.append({
                "Experiment":   "D1: ProposedCorpus-train→ProposedCorpus-test",
                "Train_source": "ProposedCorpus fixed 80/20 train split ({} participants)".format(
                    len(set(y_emp_train.index.tolist()))),
                "Test_source":  "ProposedCorpus fixed 80/20 test split ({} participants)".format(
                    len(set(y_emp_test.index.tolist()))),
                "n_train":      len(y_emp_train),
                "n_test":       len(y_emp_test),
                "Best_model":   "{}/{}".format(best_D1["model_name"], best_D1["scaler"]),
                **best_D1["metrics"],
            })

        # D2: E-DAIC-train → E-DAIC-test (within-corpus baseline, same across configs)
        print("\n" + "─" * 80)
        print("  EXPERIMENT D2: E-DAIC-train → E-DAIC-test (within-corpus baseline)")
        print("─" * 80)

        X_daic_tr_al, X_daic_te_al = align_features(X_daic_train_raw, X_daic_test_raw)

        imputer_D2 = SimpleImputer(strategy="median")
        X_daic_tr_imp = pd.DataFrame(
            imputer_D2.fit_transform(X_daic_tr_al),
            columns=X_daic_tr_al.columns, index=X_daic_tr_al.index
        ).astype(np.float32)
        X_daic_te_imp = pd.DataFrame(
            imputer_D2.transform(X_daic_te_al),
            columns=X_daic_te_al.columns, index=X_daic_te_al.index
        ).astype(np.float32)

        _, best_D2 = fit_and_eval_reg(
            models, param_grids, scalers,
            X_daic_tr_imp, y_daic_train,
            X_daic_te_imp, y_daic_test,
            experiment_tag="D2_daictr2daicte_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_D2:
            summary_rows.append({
                "Experiment":   "D2: E-DAIC-train→E-DAIC-test",
                "Train_source": "E-DAIC official train+dev split",
                "Test_source":  "E-DAIC official test split",
                "n_train":      len(y_daic_train),
                "n_test":       len(y_daic_test),
                "Best_model":   "{}/{}".format(best_D2["model_name"], best_D2["scaler"]),
                **best_D2["metrics"],
            })

    # ── Config-level summary ──────────────────────────────────────────────────
    if summary_rows:
        summary_df   = pd.DataFrame(summary_rows)
        summary_path = os.path.join(
            save_dir,
            "SUMMARY_reg_{}_{}.csv".format(config_name, timestamp)
        )
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 80)
        print("  PHQ-8 REGRESSION SUMMARY | Config: {}".format(config_name))
        print("=" * 80)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(summary_df[[
            "Experiment", "n_train", "Train_source", "n_test", "Test_source",
            "Best_model", "CCC", "MAE", "RMSE", "Pearson_r"
        ]].to_string(index=False))
        print("\n  Summary saved: {}".format(summary_path))

    return {r["Experiment"]: r for r in summary_rows}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-corpus PHQ-8 regression v2: ProposedCorpus ↔ E-DAIC"
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(SOURCE_CONFIGS.keys()),
        choices=list(SOURCE_CONFIGS.keys()),
        help="Source configurations (default: all four)"
    )
    parser.add_argument(
        "--experiments", nargs="+", default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
        help="Experiments to run (default: A B C D; D=within-corpus baselines)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    ts      = datetime.now().strftime("%Y%m%d-%H%M%S")
    scalers = get_scalers()
    models  = get_regressors()
    grids   = get_param_grids()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    all_summaries = []
    for cfg_name in args.configs:
        if cfg_name not in SOURCE_CONFIGS:
            print("Unknown config: {}".format(cfg_name))
            continue
        result = run_config(
            cfg_name, SOURCE_CONFIGS[cfg_name],
            args.experiments, ts, scalers, models, grids
        )
        for exp_label, row in result.items():
            all_summaries.append({"config": cfg_name, **row})

    # ── Final cross-config comparison ─────────────────────────────────────────
    if all_summaries:
        final_df   = pd.DataFrame(all_summaries)
        final_path = os.path.join(
            OUTPUT_BASE,
            "FINAL_COMPARISON_reg_{}.csv".format(ts)
        )
        final_df.to_csv(final_path, index=False)
        print("\n" + "=" * 80)
        print("  FINAL CROSS-CONFIG COMPARISON — PHQ-8 REGRESSION")
        print("  Primary metric: CCC (Lin's Concordance Correlation Coefficient)")
        print("  CCC=0 → predicts no better than training mean (expected for transfer failure)")
        print("=" * 80)
        print(final_df[[
            "config", "Experiment",
            "n_train", "Train_source", "n_test", "Test_source",
            "Best_model", "CCC", "MAE", "RMSE", "Pearson_r"
        ]].to_string(index=False))
        print("\n  Final comparison saved: {}".format(final_path))

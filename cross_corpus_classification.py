"""
cross_corpus_classification.py
=====================================
Cross-corpus depression CLASSIFICATION: ProposedCorpus RCT ↔ E-DAIC.
Binary labels: ProposedCorpus SCID-5-CV (0=HC, 1=MDD), E-DAIC depressed (0/1).

Source configurations (ProposedCorpus side):
  EI1_ADK     — emotion_induction_1 / ADK  (best classification F1=0.83)
  POS_ADK     — positive_training / ADK
  POS_CRADK   — positive_training / CRADK
  ALL         — all 4 phases × all conditions (ProposedCorpus pooled)

Experiments (same structure as regression script 03):
  A : ProposedCorpus (ALL data for this config) → E-DAIC official test set
  B : E-DAIC (train+dev)                 → ProposedCorpus (official held-out test split)
  C1: ProposedCorpus (train split)              → E-DAIC test
  C2: E-DAIC (train+dev)                 → ProposedCorpus (test split)

Feature selection per train set (no leakage):
  1. Mann-Whitney U + BH FDR correction (q=0.20, fallback top-20)
  2. RFE with LogisticRegression surrogate, k ∈ {5,10,15,20}

Metrics: AUC-ROC, F1 (binary), balanced accuracy, sensitivity, specificity

Usage:
  python 05_cross_corpus_classification.py
  python 05_cross_corpus_classification.py --configs EI1_ADK POS_ADK
  python 05_cross_corpus_classification.py --experiments A B C
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from shared_regression_pipeline import (
from config import RESULTS_ROOT
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    ALL_CONDITIONS,
    EDAIC_AGGREGATED_PATH,
    PROPOSED_PHASE_FEATURES,
    EXCLUDE_COLS,
    MASTER_DATA_PATH,
    _bh_correction,
    align_features,
    get_scalers,
    load_shared_split,
    normalize_id,
    rename_proposed_to_openface,
)

if CATBOOST_AVAILABLE:
    from catboost import CatBoostClassifier
if LIGHTGBM_AVAILABLE:
    from lightgbm import LGBMClassifier


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_BASE = (
    RESULTS_ROOT
    "KDD_paper/cross_corpus_classification"
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
# DATA LOADING — ProposedCorpus (binary label)
# ─────────────────────────────────────────────────────────────────────────────
def load_proposed_cls(phase, condition):
    """
    Load ProposedCorpus pre-aggregated video features + binary SCID label
    (ground_truth_label: 0=HC, 1=MDD) for one phase/condition.

    Returns
    -------
    X  : pd.DataFrame  index=ID (non-unique for multi-session phases)
    y  : pd.Series     binary labels  index=ID
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
        master_filt[["ID", "condition", "ground_truth_label"]],
        on="ID", how="inner"
    )
    print("  Merged shape [{}]: {}".format(condition, merged.shape))

    n_before = len(merged)
    merged = merged.dropna(subset=["ground_truth_label"])
    if len(merged) < n_before:
        print("  Dropped {} rows with missing labels.".format(n_before - len(merged)))

    merged = merged.set_index("ID")
    feature_cols = [c for c in merged.columns if c not in EXCLUDE_COLS]
    X = merged[feature_cols]

    n_before = X.shape[1]
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] < n_before:
        print("  Dropped {} non-numeric columns.".format(n_before - X.shape[1]))

    X = rename_proposed_to_openface(X)

    # Deduplicate columns created by the rename (positive_training has both
    # fac_AU01int__* AND AU01_r__* — after rename they collide; keep first).
    n_before = X.shape[1]
    X = X.loc[:, ~X.columns.duplicated()]
    if X.shape[1] < n_before:
        print("  Dropped {} duplicate columns after AU rename.".format(
            n_before - X.shape[1]))

    y = merged["ground_truth_label"].astype(int)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    print("  X: {}  |  HC={}, MDD={}".format(X.shape, n0, n1))
    return X, y


def load_proposed_all_phases_cls():
    """
    Pool all 4 phases × all conditions into one feature matrix.
    Uses column intersection across phases so all rows share the same features.
    Participant-level index is preserved (non-unique — multiple sessions per participant).

    Returns
    -------
    X : pd.DataFrame  (many sessions × common features)  index=ID (non-unique)
    y : pd.Series     binary labels  index=ID (non-unique)
    """
    print("\n  Loading ALL phases × ALL conditions ...")
    all_X = []
    all_y = []
    for phase in ALL_PHASES:
        Xp, yp = load_proposed_cls(phase, "ALL_CONDITIONS")
        all_X.append(Xp)
        all_y.append(yp)

    # Feature intersection across phases
    common_cols = sorted(set.intersection(*[set(Xp.columns) for Xp in all_X]))
    print("  ALL phases: {} common features across {} phases".format(
        len(common_cols), len(ALL_PHASES)))

    X_stacked = pd.concat([Xp[common_cols] for Xp in all_X], axis=0)
    y_stacked = pd.concat(all_y, axis=0)

    n0 = int((y_stacked == 0).sum())
    n1 = int((y_stacked == 1).sum())
    print("  ALL phases pooled: {}  |  HC={}, MDD={}".format(
        X_stacked.shape, n0, n1))
    return X_stacked, y_stacked


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING — E-DAIC (binary label)
# ─────────────────────────────────────────────────────────────────────────────
def load_edaic_cls(split_filter=None):
    """
    Load E-DAIC aggregated OpenFace features + binary depression label
    (depressed: 0=non-depressed, 1=depressed).

    Returns
    -------
    X         : pd.DataFrame  index=participant_id
    y         : pd.Series     binary labels  index=participant_id
    split_col : pd.Series     train/dev/test labels
    """
    print("  Loading E-DAIC classification data: {}".format(EDAIC_AGGREGATED_PATH))
    df = pd.read_csv(EDAIC_AGGREGATED_PATH)
    df.columns = df.columns.str.strip()
    df["participant_id"] = df["participant_id"].astype(str)

    if split_filter is not None:
        df = df[df["split"].isin(split_filter)].copy()
        print("  Filtered to splits={}: {} participants".format(split_filter, len(df)))

    df = df.dropna(subset=["depressed"])
    df = df.set_index("participant_id")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df["depressed"].astype(int)
    split_col = df["split"]

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    print("  X: {}  |  non-dep={}, dep={}".format(X.shape, n0, n1))
    return X, y, split_col


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION (binary-appropriate)
# ─────────────────────────────────────────────────────────────────────────────
def mannwhitney_select_fdr(X, y, q=0.20, fallback_top_n=20):
    """
    Feature selection via Mann-Whitney U test + Benjamini-Hochberg FDR correction.
    Appropriate for binary classification targets.
    Effect size: rank-biserial correlation |r|.

    Fit on training data only — no leakage.
    """
    y0_idx = set(y[y == 0].index.tolist())
    y1_idx = set(y[y == 1].index.tolist())

    effect_sizes, pvals, names = [], [], []
    for col in X.columns:
        try:
            x0 = X.loc[X.index.isin(y0_idx), col].dropna().values
            x1 = X.loc[X.index.isin(y1_idx), col].dropna().values
            if len(x0) < 3 or len(x1) < 3:
                continue
            stat, p = mannwhitneyu(x0, x1, alternative="two-sided")
            n0n1 = float(len(x0) * len(x1))
            r = abs(1.0 - (2.0 * stat) / n0n1) if n0n1 > 0 else 0.0
            effect_sizes.append(r)
            pvals.append(float(p))
            names.append(col)
        except Exception:
            continue

    if not names:
        return list(X.columns[:fallback_top_n])

    mask = _bh_correction(pvals, q=q)
    selected = [names[i] for i in range(len(names)) if mask[i]]

    if len(selected) < 5:
        order = sorted(range(len(names)),
                       key=lambda i: effect_sizes[i], reverse=True)
        selected = [names[i] for i in order[:fallback_top_n]]
        print("  Mann-Whitney FDR: <5 features survived; fallback to top-{} "
              "(max |r|={:.3f})".format(fallback_top_n, max(effect_sizes)))
    else:
        max_r = max(effect_sizes[i] for i in range(len(names)) if mask[i])
        print("  Mann-Whitney FDR (q={:.2f}): {} features kept "
              "(max |r|={:.3f})".format(q, len(selected), max_r))
    return selected


def rfe_select_cls(X, y, num_features_range=(5, 10, 15, 20)):
    """
    RFE with LogisticRegression surrogate, CV-optimised feature count.
    Uses 3-fold StratifiedKFold; picks k with best mean CV AUC.
    """
    surrogate = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_support, best_score, best_k = None, -np.inf, 0

    for k in num_features_range:
        if k > X.shape[1]:
            continue
        try:
            sel = RFE(estimator=clone(surrogate),
                      n_features_to_select=k, step=0.1)
            X_rfe = sel.fit_transform(X, y)
            scores = cross_val_score(
                clone(surrogate), X_rfe, y,
                cv=skf, scoring="roc_auc", n_jobs=1
            )
            score = scores.mean()
            if score > best_score:
                best_score = score
                best_k = k
                best_support = sel.support_
        except Exception as e:
            print("  RFE: k={} failed ({}), skipping".format(k, e))
            continue

    if best_support is None:
        print("  RFE: all sizes failed, returning all features")
        return list(X.columns)

    kept = [X.columns[i] for i in range(len(X.columns)) if best_support[i]]
    print("  RFE (LR): {} features selected "
          "(k={}, CV AUC={:.3f})".format(len(kept), best_k, best_score))
    return kept


def select_features_cls(X_train, y_train, X_test):
    """Mann-Whitney FDR → RFE (LR surrogate) on train only."""
    mw_feats = mannwhitney_select_fdr(
        X_train, y_train, q=FDR_Q, fallback_top_n=FDR_FALLBACK_N
    )
    kept = rfe_select_cls(X_train[mw_feats], y_train,
                          num_features_range=RFE_K_RANGE)
    if not kept:
        kept = mw_feats
    return X_train[kept], X_test[kept], kept


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIERS & HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────────────────────────────────────
def get_classifiers():
    models = {
        "LR":  LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
        "SVC": SVC(kernel="rbf", probability=True, random_state=42),
        "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
        "ET":  ExtraTreesClassifier(n_estimators=100, random_state=42),
        "GB":  GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGB": XGBClassifier(n_estimators=100, random_state=42,
                             verbosity=0, eval_metric="logloss"),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=500, random_state=42),
    }
    if LIGHTGBM_AVAILABLE:
        models["LGBM"] = LGBMClassifier(
            n_estimators=100, random_state=42, verbose=-1
        )
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=100, random_state=42, verbose=0
        )
    return models


def get_cls_param_grids():
    grids = {
        "LR":  {"C": [0.01, 0.1, 1.0, 10.0]},
        "SVC": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
        "RF":  {"n_estimators": [100, 200],
                "max_features": ["sqrt", 0.3],
                "min_samples_leaf": [1, 3]},
        "ET":  {"n_estimators": [100, 200], "max_features": ["sqrt", 0.3]},
        "GB":  {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]},
        "XGB": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]},
        "KNN": {"n_neighbors": [3, 5, 7, 9]},
        "MLP": {"hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "alpha": [1e-4, 1e-3]},
        "LGBM": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                 "num_leaves": [15, 31]},
        "CatBoost": {"iterations": [100, 200], "learning_rate": [0.05, 0.1],
                     "depth": [4, 6]},
    }
    return grids


def grid_search_cls(model, param_grid, X_train, y_train, cv=5):
    """GridSearchCV with roc_auc scoring, n_jobs=1 for Python 3.6 HPC compat."""
    if not param_grid:
        model.fit(X_train, y_train)
        return model
    gs = GridSearchCV(model, param_grid, scoring="roc_auc",
                      cv=cv, n_jobs=1, refit=True)
    gs.fit(X_train, y_train)
    print("    Best params: {}  (CV AUC={:.3f})".format(
        gs.best_params_, gs.best_score_))
    return gs.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_cls(model, X_test, y_test):
    """AUC-ROC, F1 (binary), balanced accuracy, sensitivity, specificity."""
    y_pred = model.predict(X_test)

    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = float("nan")

    f1  = float(f1_score(y_test, y_pred, average="binary", zero_division=0))
    bac = float(balanced_accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = float("nan")
        specificity = float("nan")

    return {
        "AUC":         round(auc,         4),
        "F1":          round(f1,          4),
        "BalancedAcc": round(bac,         4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "n_test":      int(len(y_test)),
    }


def print_cls_metrics(metrics, label=""):
    tag = "  [{}] ".format(label) if label else "  "
    print("{}AUC={:.3f}  F1={:.3f}  BalAcc={:.3f}  "
          "Sens={:.3f}  Spec={:.3f}  n={}".format(
              tag,
              metrics["AUC"], metrics["F1"], metrics["BalancedAcc"],
              metrics["Sensitivity"], metrics["Specificity"], metrics["n_test"]))


# ─────────────────────────────────────────────────────────────────────────────
# FIT + EVALUATE: all scaler × model combos
# ─────────────────────────────────────────────────────────────────────────────
def fit_and_eval_cls(models, param_grids, scalers,
                     X_train_imp, y_train,
                     X_test_imp, y_test,
                     experiment_tag, save_dir, timestamp):
    """
    Run all (scaler × model) combinations.
    Returns (results_df, best_info_dict) sorted by AUC descending.
    """
    if len(np.unique(y_train)) < 2:
        print("  WARNING: only one class in train — skipping {}".format(experiment_tag))
        return pd.DataFrame(), {}

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

        X_tr_fs, X_te_fs, kept = select_features_cls(X_tr_sc, y_train, X_te_sc)

        for model_name, model in models.items():
            key = (model_name, scaler_name)
            print("  {} | {}".format(model_name, scaler_name))

            if model_name in param_grids:
                fitted = grid_search_cls(
                    clone(model), param_grids[model_name],
                    X_tr_fs, y_train, cv=5
                )
            else:
                fitted = clone(model)
                fitted.fit(X_tr_fs, y_train)

            metrics = evaluate_cls(fitted, X_te_fs, y_test)
            all_results[key]  = metrics
            all_trained[key]  = fitted
            all_features[key] = kept
            print_cls_metrics(metrics, label="{}/{}".format(model_name, scaler_name))

    if not all_results:
        return pd.DataFrame(), {}

    rows = [{"model": m, "scaler": s, **v}
            for (m, s), v in all_results.items()]
    results_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)

    # Best by AUC; break ties by F1
    best_key = max(
        all_results,
        key=lambda k: (all_results[k]["AUC"], all_results[k]["F1"])
    )
    bm, bs = best_key
    print("\n  >> Best [{}]: {}/{}  AUC={:.3f}  F1={:.3f}  "
          "BalAcc={:.3f}  Sens={:.3f}  Spec={:.3f}".format(
              experiment_tag, bm, bs,
              all_results[best_key]["AUC"],
              all_results[best_key]["F1"],
              all_results[best_key]["BalancedAcc"],
              all_results[best_key]["Sensitivity"],
              all_results[best_key]["Specificity"]))

    csv_path = os.path.join(
        save_dir, "cls_{}_{}.csv".format(experiment_tag, timestamp)
    )
    results_df.to_csv(csv_path, index=False)
    print("  Saved: {}".format(csv_path))

    model_path = os.path.join(
        save_dir, "best_cls_model_{}_{}.joblib".format(experiment_tag, timestamp)
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
    """
    Run experiments A / B / C for one ProposedCorpus source configuration.
    Returns a dict {experiment_label: result_row}.
    """
    phases    = config["phases"]
    condition = config["condition"]
    save_dir  = os.path.join(OUTPUT_BASE, config_name)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  CROSS-CORPUS CLASSIFICATION | Config: {}".format(config_name))
    print("  Phases: {}  Condition: {}".format(phases, condition))
    print("  Experiments: {}".format(experiments))
    print("=" * 80)

    # ── Load ProposedCorpus ──────────────────────────────────────────────────────────
    print("\n[1] Loading ProposedCorpus data ...")
    if config_name == "ALL":
        X_emp, y_emp = load_proposed_all_phases_cls()
        split_tag    = "ALL"
    else:
        assert len(phases) == 1, "Single-phase config must have exactly one phase"
        X_emp, y_emp = load_proposed_cls(phases[0], condition)
        split_tag    = "ALL" if condition == "ALL_CONDITIONS" else condition

    # Aggregate sessions to participant level — matches within-corpus nested CV
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

    n0_tr = int((y_emp_train == 0).sum()); n1_tr = int((y_emp_train == 1).sum())
    n0_te = int((y_emp_test  == 0).sum()); n1_te = int((y_emp_test  == 1).sum())
    print("  ProposedCorpus train: n_participants={} (HC={}, MDD={})".format(
              len(emp_train_idx), n0_tr, n1_tr))
    print("  ProposedCorpus test:  n_participants={} (HC={}, MDD={})".format(
              len(emp_test_idx), n0_te, n1_te))

    # ── Load E-DAIC ───────────────────────────────────────────────────────────
    print("\n[2] Loading E-DAIC data ...")
    try:
        X_daic_all, y_daic_all, split_col = load_edaic_cls(split_filter=None)
    except FileNotFoundError:
        print("  ERROR: E-DAIC aggregated file not found at:\n"
              "  {}\n"
              "  Run 01_aggregate_edaic_video_features.py first!".format(
                  EDAIC_AGGREGATED_PATH))
        return {}

    X_daic_train_raw = X_daic_all[split_col.isin(["train", "dev"])]
    y_daic_train     = y_daic_all[split_col.isin(["train", "dev"])]
    X_daic_test_raw  = X_daic_all[split_col == "test"]
    y_daic_test      = y_daic_all[split_col == "test"]

    n0_tr = int((y_daic_train == 0).sum()); n1_tr = int((y_daic_train == 1).sum())
    n0_te = int((y_daic_test  == 0).sum()); n1_te = int((y_daic_test  == 1).sum())
    print("  E-DAIC train+dev: n={} (non-dep={}, dep={})".format(
        len(y_daic_train), n0_tr, n1_tr))
    print("  E-DAIC test:      n={} (non-dep={}, dep={})".format(
        len(y_daic_test), n0_te, n1_te))

    summary_rows = []

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A: ProposedCorpus ALL → E-DAIC official test set
    # ═══════════════════════════════════════════════════════════════════════════
    if "A" in experiments:
        print("\n" + "─" * 80)
        print("  EXPERIMENT A: ProposedCorpus (ALL) → E-DAIC test")
        print("  Trains on entire ProposedCorpus source config, tests on E-DAIC test")
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

        _, best_A = fit_and_eval_cls(
            models, param_grids, scalers,
            X_emp_imp, y_emp_all,
            X_daic_te_imp, y_daic_test,
            experiment_tag="A_emp2daic_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_A:
            summary_rows.append({
                "Experiment":    "A: ProposedCorpus-ALL→E-DAIC-test",
                "Train_source":  "ProposedCorpus-ALL ({} sessions, {} participants, fixed split N/A)".format(
                    len(y_emp_all), len(set(y_emp_all.index.tolist()))),
                "Test_source":   "E-DAIC official test split",
                "n_train":       len(y_emp_all),
                "n_test":        len(y_daic_test),
                "Best_model":    "{}/{}".format(best_A["model_name"], best_A["scaler"]),
                **best_A["metrics"],
            })

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B: E-DAIC (train+dev) → ProposedCorpus test
    # ═══════════════════════════════════════════════════════════════════════════
    if "B" in experiments:
        print("\n" + "─" * 80)
        print("  EXPERIMENT B: E-DAIC (train+dev) → ProposedCorpus test")
        print("  Tests whether clinical-interview model transfers to RCT passive task")
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

        _, best_B = fit_and_eval_cls(
            models, param_grids, scalers,
            X_daic_tr_imp, y_daic_train,
            X_emp_te_imp, y_emp_test,
            experiment_tag="B_daic2emp_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_B:
            summary_rows.append({
                "Experiment":    "B: E-DAIC-ALL→ProposedCorpus-test",
                "Train_source":  "E-DAIC official train+dev split",
                "Test_source":   "ProposedCorpus fixed 80/20 test split ({} participants)".format(
                    len(set(y_emp_test.index.tolist()))),
                "n_train":       len(y_daic_train),
                "n_test":        len(y_emp_test),
                "Best_model":    "{}/{}".format(best_B["model_name"], best_B["scaler"]),
                **best_B["metrics"],
            })

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT C: leave-one-corpus-out (both directions with official splits)
    # ═══════════════════════════════════════════════════════════════════════════
    if "C" in experiments:
        # C1: ProposedCorpus train split → E-DAIC test
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

        _, best_C1 = fit_and_eval_cls(
            models, param_grids, scalers,
            X_emp_tr_imp, y_emp_train,
            X_daic_te_imp, y_daic_test,
            experiment_tag="C1_emptr2daicte_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_C1:
            summary_rows.append({
                "Experiment":    "C1: ProposedCorpus-train→E-DAIC-test",
                "Train_source":  "ProposedCorpus fixed 80/20 train split ({} participants)".format(
                    len(set(y_emp_train.index.tolist()))),
                "Test_source":   "E-DAIC official test split",
                "n_train":       len(y_emp_train),
                "n_test":        len(y_daic_test),
                "Best_model":    "{}/{}".format(best_C1["model_name"], best_C1["scaler"]),
                **best_C1["metrics"],
            })

        # C2: E-DAIC train+dev → ProposedCorpus test split
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

        _, best_C2 = fit_and_eval_cls(
            models, param_grids, scalers,
            X_daic_tr2_imp, y_daic_train,
            X_emp_te2_imp, y_emp_test,
            experiment_tag="C2_daictr2empte_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_C2:
            summary_rows.append({
                "Experiment":    "C2: E-DAIC-train→ProposedCorpus-test",
                "Train_source":  "E-DAIC official train+dev split",
                "Test_source":   "ProposedCorpus fixed 80/20 test split ({} participants)".format(
                    len(set(y_emp_test.index.tolist()))),
                "n_train":       len(y_daic_train),
                "n_test":        len(y_emp_test),
                "Best_model":    "{}/{}".format(best_C2["model_name"], best_C2["scaler"]),
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

        _, best_D1 = fit_and_eval_cls(
            models, param_grids, scalers,
            X_emp_tr_imp, y_emp_train,
            X_emp_te_imp, y_emp_test,
            experiment_tag="D1_emptr2empte_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_D1:
            summary_rows.append({
                "Experiment":    "D1: ProposedCorpus-train→ProposedCorpus-test",
                "Train_source":  "ProposedCorpus fixed 80/20 train split ({} participants)".format(
                    len(set(y_emp_train.index.tolist()))),
                "Test_source":   "ProposedCorpus fixed 80/20 test split ({} participants)".format(
                    len(set(y_emp_test.index.tolist()))),
                "n_train":       len(y_emp_train),
                "n_test":        len(y_emp_test),
                "Best_model":    "{}/{}".format(best_D1["model_name"], best_D1["scaler"]),
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

        _, best_D2 = fit_and_eval_cls(
            models, param_grids, scalers,
            X_daic_tr_imp, y_daic_train,
            X_daic_te_imp, y_daic_test,
            experiment_tag="D2_daictr2daicte_{}".format(config_name),
            save_dir=save_dir, timestamp=timestamp
        )
        if best_D2:
            summary_rows.append({
                "Experiment":    "D2: E-DAIC-train→E-DAIC-test",
                "Train_source":  "E-DAIC official train+dev split",
                "Test_source":   "E-DAIC official test split",
                "n_train":       len(y_daic_train),
                "n_test":        len(y_daic_test),
                "Best_model":    "{}/{}".format(best_D2["model_name"], best_D2["scaler"]),
                **best_D2["metrics"],
            })

    # ── Config-level summary ──────────────────────────────────────────────────
    if summary_rows:
        summary_df   = pd.DataFrame(summary_rows)
        summary_path = os.path.join(
            save_dir,
            "SUMMARY_cls_{}_{}.csv".format(config_name, timestamp)
        )
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 80)
        print("  CLASSIFICATION SUMMARY | Config: {}".format(config_name))
        print("=" * 80)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(summary_df[[
            "Experiment", "n_train", "Train_source", "n_test", "Test_source",
            "Best_model", "AUC", "F1", "BalancedAcc", "Sensitivity", "Specificity"
        ]].to_string(index=False))
        print("\n  Summary saved: {}".format(summary_path))

    return {r["Experiment"]: r for r in summary_rows}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-corpus depression classification: ProposedCorpus ↔ E-DAIC"
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(SOURCE_CONFIGS.keys()),
        choices=list(SOURCE_CONFIGS.keys()),
        help="Source configurations to run (default: all four)"
    )
    parser.add_argument(
        "--experiments", nargs="+", default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
        help="Which experiments to run (default: A B C D; D=within-corpus baselines)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    ts      = datetime.now().strftime("%Y%m%d-%H%M%S")
    scalers = get_scalers()
    models  = get_classifiers()
    grids   = get_cls_param_grids()

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
            "FINAL_COMPARISON_cls_{}.csv".format(ts)
        )
        final_df.to_csv(final_path, index=False)
        print("\n" + "=" * 80)
        print("  FINAL CROSS-CONFIG COMPARISON (best model per config/experiment)")
        print("=" * 80)
        print(final_df[[
            "config", "Experiment",
            "n_train", "Train_source", "n_test", "Test_source",
            "Best_model", "AUC", "F1", "BalancedAcc", "Sensitivity", "Specificity"
        ]].to_string(index=False))
        print("\n  Final comparison saved: {}".format(final_path))

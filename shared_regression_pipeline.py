"""
shared_regression_pipeline.py
==============================
Shared helpers for PHQ-8 regression and cross-corpus video analysis.
Self-contained — no dependency on the classification pipeline.

Regression targets:
  ProposedCorpus : phq_8  (from participant_master_data.csv)
  E-DAIC  : PHQ_score  (from edaic_labels.csv, same instrument)

Statistical aggregation (18 functionals per OpenFace column):
  Original signal   : mean, std, min, max, skew, kurt, range, entropy
  Frame differences : mean, std, min, max, skew, kurt, range, entropy
  Dynamic           : rate_of_change, peaks_count
  → column naming: {signal}__{stat}  (matches ProposedCorpus processed CSVs)
"""

import math
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PATHS  —  imported from config.py (set your paths there)
# ─────────────────────────────────────────────────────────────────────────────
from config import (
    MASTER_DATA_PATH,
    EDAIC_LABELS_PATH,
    EDAIC_OPENFACE_ROOT,
    EDAIC_AGGREGATED_PATH,
    PROPOSED_SPLITS_DIR,
    PROPOSED_PHASE_FEATURES,
    RESULTS_ROOT,
)

# Metadata columns to exclude from feature matrix
EXCLUDE_COLS = {
    "ID", "participant_id", "gender", "age", "condition",
    "ground_truth_label", "label", "depressed",
    "phq_9", "phq_8", "PHQ_score", "ads.1",
    "HRSD_17.1", "HRSD_21.1", "HRSD_24.1", "HRSD_6.1",
    "training_group", "training_session_idx", "split",
    "Geschlecht", "Alter", "Bedingung",
    "PHQ8-Score", "PHQ9-Score", "previous_depression_diagnosis",
    "EMG-Messung(ja/nein)", "condition_group", "sentiment",
    "training1_rating", "training2_rating", "session_rating",
}

# Conditions in ProposedCorpus RCT
ALL_CONDITIONS = ["ADK", "CR", "CRADK", "SHAM"]


# ─────────────────────────────────────────────────────────────────────────────
# PROPOSED → OPENFACE COLUMN RENAME
# ─────────────────────────────────────────────────────────────────────────────
def rename_proposed_to_openface(X: pd.DataFrame) -> pd.DataFrame:
    """
    Rename ProposedCorpus custom column names to standard OpenFace names so that
    feature alignment with E-DAIC (which uses raw OpenFace output) works
    correctly.

    ProposedCorpus preprocessing uses a different naming convention:
      fac_AU{XX}int   →  AU{XX}_r   (AU intensity)
      fac_AU{XX}pres  →  AU{XX}_c   (AU presence / confidence)

    Columns that have no OpenFace equivalent (fac_LMK*, fac_hap*, X_N, Y_N …)
    are left unchanged — they will be dropped by align_features() when
    intersecting with E-DAIC.
    """
    rename_map = {}
    for col in X.columns:
        if "__" not in col:
            continue
        sig, stat = col.split("__", 1)

        m_int = re.match(r"fac_AU(\d+)int$", sig)
        if m_int:
            rename_map[col] = "AU{}_r__{}".format(m_int.group(1), stat)
            continue

        m_pres = re.match(r"fac_AU(\d+)pres$", sig)
        if m_pres:
            rename_map[col] = "AU{}_c__{}".format(m_pres.group(1), stat)
            continue

    if rename_map:
        X = X.rename(columns=rename_map)
        print("  Column rename: {} ProposedCorpus AU cols → OpenFace naming"
              " (AU{{XX}}_r / AU{{XX}}_c)".format(len(rename_map)))
    return X


# ─────────────────────────────────────────────────────────────────────────────
# ID NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def normalize_id(pid) -> str:
    if pd.isna(pid) or (isinstance(pid, float) and math.isnan(pid)):
        return str(np.nan)
    try:
        return str(int(float(pid)))
    except (ValueError, TypeError):
        return str(pid)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SPLIT LOADER (reuses the pre-computed ProposedCorpus splits)
# ─────────────────────────────────────────────────────────────────────────────
def load_shared_split(tag: str):
    """Load pre-computed ProposedCorpus train/test participant IDs (tag: ALL/ADK/CR/CRADK/SHAM)."""
    train_path = os.path.join(PROPOSED_SPLITS_DIR, f"shared_train_ids_{tag}.csv")
    test_path  = os.path.join(PROPOSED_SPLITS_DIR, f"shared_test_ids_{tag}.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Split file not found: {train_path}\n"
            "Run 00_generate_shared_split.py first."
        )
    train_ids = set(pd.read_csv(train_path)["ID"].astype(str).tolist())
    test_ids  = set(pd.read_csv(test_path)["ID"].astype(str).tolist())
    print(f"  Loaded shared split [{tag}]: {len(train_ids)} train / {len(test_ids)} test")
    return train_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# PROPOSED DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
# All continuous target columns available in master data
ALL_TARGET_COLS = ["phq_8", "phq_9", "ads.1",
                   "HRSD_6.1", "HRSD_17.1", "HRSD_21.1", "HRSD_24.1"]

# Human-readable short names for output filenames / tables
TARGET_SHORT_NAMES = {
    "phq_8":     "PHQ8",
    "phq_9":     "PHQ9",
    "ads.1":     "CES-D",
    "HRSD_6.1":  "HRSD6",
    "HRSD_17.1": "HRSD17",
    "HRSD_21.1": "HRSD21",
    "HRSD_24.1": "HRSD24",
}


def load_proposed_data(phase: str, condition: str, target_col: str = "phq_8"):
    """
    Load ProposedCorpus pre-aggregated video features for one phase/condition.

    Parameters
    ----------
    phase      : one of PROPOSED_PHASE_FEATURES keys
    condition  : one of ALL_CONDITIONS or 'ALL_CONDITIONS'
    target_col : regression target column from master data
                 (default 'phq_8'; also supports 'phq_9', 'ads.1',
                  'HRSD_6.1', 'HRSD_17.1', 'HRSD_21.1', 'HRSD_24.1')

    Returns
    -------
    X            : pd.DataFrame  [n_subjects × n_features]  index=ID
    y            : pd.Series     [n_subjects]  target scores  index=ID
    subject_ids  : pd.Series     of string IDs
    """
    if target_col not in ALL_TARGET_COLS:
        raise ValueError(
            "Unknown target_col '{}'. Choose from: {}".format(
                target_col, ALL_TARGET_COLS))

    features_path = PROPOSED_PHASE_FEATURES[phase]
    print("  Loading ProposedCorpus features: {}".format(features_path))

    feat_df = pd.read_csv(features_path)
    feat_df.columns = feat_df.columns.str.strip()
    feat_df["ID"]   = feat_df["ID"].apply(normalize_id)

    # Drop non-feature columns that may appear in the CSV
    drop_cols = [c for c in EXCLUDE_COLS if c in feat_df.columns and c != "ID"]
    feat_df.drop(columns=drop_cols, inplace=True)

    # Deduplicate column names — duplicate cols cause select_dtypes and boolean
    # masking to behave incorrectly in older pandas (Python 3.6 env).
    feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()]

    # Master data (labels + demographics + scores)
    master_df = pd.read_csv(MASTER_DATA_PATH)
    master_df["ID"] = master_df["ID"].apply(normalize_id)
    master_df.rename(columns={"label": "ground_truth_label"}, inplace=True)

    # Filter master by condition
    if condition == "ALL_CONDITIONS":
        master_filt = master_df.copy()
    else:
        master_filt = master_df[master_df["condition"] == condition].copy()

    # Coerce target column to numeric — some cells contain text annotations
    # (e.g. phq_9 has "10 (bei Trainingstermin ausgefüllt)"); coerce to NaN.
    if target_col in master_filt.columns:
        master_filt = master_filt.copy()
        master_filt[target_col] = pd.to_numeric(master_filt[target_col],
                                                 errors="coerce")

    merge_cols = ["ID", "condition", target_col, "ground_truth_label"]
    merged = pd.merge(feat_df, master_filt[merge_cols], on="ID", how="inner")
    print("  Merged shape for condition '{}': {}".format(condition, merged.shape))

    # Drop rows where target is missing (or was coerced to NaN)
    n_before = len(merged)
    merged   = merged.dropna(subset=[target_col])
    if len(merged) < n_before:
        print("  Dropped {} rows with missing/invalid {}.".format(
            n_before - len(merged), target_col))

    merged = merged.set_index("ID")
    feature_cols = [c for c in merged.columns if c not in EXCLUDE_COLS]
    X = merged[feature_cols]

    # Drop any non-numeric columns that slipped through (e.g. 'Ja'/'Nein' booleans)
    n_before = X.shape[1]
    X = X.select_dtypes(include=[np.number])
    dropped = n_before - X.shape[1]
    if dropped:
        print("  Dropped {} non-numeric columns.".format(dropped))

    # Rename ProposedCorpus AU columns to standard OpenFace names so that
    # align_features() with E-DAIC gives the full AU/gaze/pose overlap
    # instead of just the gaze+pose subset.
    X = rename_proposed_to_openface(X)

    # Deduplicate columns after rename (some phase CSVs contain both the
    # original fac_AU*int__* names AND the standard AU*_r__* names; after
    # rename they become duplicate column names → keep first occurrence).
    n_before = X.shape[1]
    X = X.loc[:, ~X.columns.duplicated()]
    if X.shape[1] < n_before:
        print("  Dropped {} duplicate columns after AU rename.".format(
            n_before - X.shape[1]))

    y = merged[target_col]
    subject_ids = merged.index.to_series()

    short = TARGET_SHORT_NAMES.get(target_col, target_col)
    print("  X: {}  |  {}: mean={:.1f}, std={:.1f}, range=[{:.0f},{:.0f}]".format(
        X.shape, short, y.mean(), y.std(), y.min(), y.max()))
    return X, y, subject_ids


# ─────────────────────────────────────────────────────────────────────────────
# E-DAIC DATA LOADING (from pre-aggregated CSV)
# ─────────────────────────────────────────────────────────────────────────────
def load_edaic_data(split_filter: list = None):
    """
    Load E-DAIC aggregated OpenFace features.
    Requires aggregate_edaic_video_features.py to have been run first.

    Parameters
    ----------
    split_filter : list or None
        If given (e.g. ['train'], ['dev'], ['test', 'dev']), keep only those splits.
        If None, all splits are returned.

    Returns
    -------
    X            : pd.DataFrame  [n_subjects × n_features]  index=participant_id
    y            : pd.Series     [n_subjects]  PHQ-8 scores
    subject_ids  : pd.Series
    split_col    : pd.Series     train/dev/test labels (for stratified use)
    """
    print(f"  Loading E-DAIC aggregated features: {EDAIC_AGGREGATED_PATH}")
    df = pd.read_csv(EDAIC_AGGREGATED_PATH)
    df.columns = df.columns.str.strip()
    df["participant_id"] = df["participant_id"].astype(str)

    if split_filter is not None:
        df = df[df["split"].isin(split_filter)].copy()
        print(f"  Filtered to splits={split_filter}: {len(df)} participants")

    df = df.dropna(subset=["PHQ_score"])
    df = df.set_index("participant_id")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df["PHQ_score"]
    split_col = df["split"]
    subject_ids = df.index.to_series()

    print(f"  X: {X.shape}  |  PHQ-8: mean={y.mean():.1f}, "
          f"std={y.std():.1f}, range=[{y.min():.0f},{y.max():.0f}]")
    return X, y, subject_ids, split_col


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def spearman_select(X: pd.DataFrame, y: pd.Series,
                    top_n: int = 50) -> list:
    """
    Select top_n features by absolute Spearman correlation with target.
    Computed on training data only — no leakage.
    """
    corrs = {}
    for col in X.columns:
        r, p = spearmanr(X[col], y, nan_policy="omit")
        try:
            r_val = float(r)  # scipy 1.x may return array-like in old versions
        except (TypeError, ValueError):
            continue
        if np.isfinite(r_val):
            corrs[col] = abs(r_val)
    if not corrs:
        return list(X.columns[:top_n])
    sorted_feats = sorted(corrs, key=corrs.get, reverse=True)
    selected = sorted_feats[:top_n]
    print(f"  Spearman selection: top {len(selected)} features "
          f"(max |r|={max(corrs.values()):.3f})")
    return selected


def pearson_redundancy_prune(X: pd.DataFrame,
                             threshold: float = 0.90) -> list:
    """
    Remove redundant features: greedily drop the later feature in any pair
    with Pearson |r| >= threshold. Keeps the first (higher-ranked) feature.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [c for c in upper.columns if any(upper[c] >= threshold)]
    kept    = [c for c in X.columns if c not in set(to_drop)]
    print(f"  Pearson pruning (r>={threshold}): "
          f"{len(to_drop)} dropped, {len(kept)} kept")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED FEATURE SELECTION (v2 — borrowed from classification pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def _bh_correction(pvals, q=0.20):
    """Benjamini-Hochberg FDR correction. Returns boolean mask (True = keep)."""
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool)
    sorted_idx = np.argsort(pvals)
    sorted_p   = np.array(pvals)[sorted_idx]
    threshold  = np.arange(1, n + 1) / float(n) * q
    below      = sorted_p <= threshold
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_k  = int(np.max(np.where(below)))
    result = np.zeros(n, dtype=bool)
    result[sorted_idx[:max_k + 1]] = True
    return result


def spearman_select_fdr(X: pd.DataFrame, y: pd.Series,
                        q: float = 0.20,
                        fallback_top_n: int = 20) -> list:
    """
    FDR-corrected Spearman |r| filter for regression.

    Computes Spearman correlation of each feature with the continuous target,
    applies Benjamini-Hochberg correction at level q.  If fewer than 5
    features survive, falls back to the top-fallback_top_n by raw |r|.

    Fit on training data only — no leakage.
    """
    corrs, pvals, names = [], [], []
    for col in X.columns:
        try:
            r, p = spearmanr(X[col], y, nan_policy="omit")
            r_val = float(r)
            p_val = float(p)
        except Exception:
            continue
        if np.isfinite(r_val) and np.isfinite(p_val):
            corrs.append(abs(r_val))
            pvals.append(p_val)
            names.append(col)

    if not names:
        return list(X.columns[:fallback_top_n])

    mask = _bh_correction(pvals, q=q)
    selected = [names[i] for i in range(len(names)) if mask[i]]

    if len(selected) < 5:
        # fall back to top-N by raw |r|
        order     = sorted(range(len(names)), key=lambda i: corrs[i], reverse=True)
        selected  = [names[i] for i in order[:fallback_top_n]]
        print("  Spearman FDR: <5 features survived; fallback to top-{} by |r|"
              " (max |r|={:.3f})".format(fallback_top_n, max(corrs)))
    else:
        max_r = max(corrs[i] for i in range(len(names)) if mask[i])
        print("  Spearman FDR (q={:.2f}): {} features kept"
              " (max |r|={:.3f})".format(q, len(selected), max_r))

    return selected


def rfe_select_regression(X: pd.DataFrame, y: pd.Series,
                          num_features_range=(5, 10, 15, 20)) -> list:
    """
    RFE with Ridge surrogate, CV-optimised feature count.

    Uses 3-fold KFold (not StratifiedKFold — regression target).
    Tries each k in num_features_range; picks the k with best mean CV MAE.
    Falls back to all columns if RFE fails.
    """
    surrogate = Ridge()
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_support, best_mae, best_k = None, np.inf, 0

    for k in num_features_range:
        if k > X.shape[1]:
            continue
        try:
            sel   = RFE(estimator=clone(surrogate),
                        n_features_to_select=k, step=0.1)
            X_rfe = sel.fit_transform(X, y)
            scores = cross_val_score(clone(surrogate), X_rfe, y,
                                     cv=kf,
                                     scoring="neg_mean_absolute_error",
                                     n_jobs=1)
            mae = -scores.mean()
            if mae < best_mae:
                best_mae     = mae
                best_k       = k
                best_support = sel.support_
        except Exception as e:
            print("  RFE: k={} failed ({}), skipping".format(k, e))
            continue

    if best_support is None:
        print("  RFE: all sizes failed, returning all features")
        return list(X.columns)

    kept = [X.columns[i] for i in range(len(X.columns)) if best_support[i]]
    print("  RFE (Ridge): {} features selected"
          " (k={}, CV MAE={:.3f})".format(len(kept), best_k, best_mae))
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# MODELS & HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────────────────────────────────────
def get_regressors():
    models = {
        "Ridge":    Ridge(),
        "Lasso":    Lasso(max_iter=5000),
        "ElasticNet": ElasticNet(max_iter=5000),
        "SVR":      SVR(kernel="rbf"),
        "KNN":      KNeighborsRegressor(),
        "RF":       RandomForestRegressor(n_estimators=100, random_state=42),
        "ET":       ExtraTreesRegressor(n_estimators=100, random_state=42),
        "GB":       GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGB":      XGBRegressor(n_estimators=100, random_state=42,
                                  verbosity=0, eval_metric="rmse"),
        "MLP":      MLPRegressor(max_iter=500, random_state=42),
    }
    if LIGHTGBM_AVAILABLE:
        models["LGBM"] = LGBMRegressor(n_estimators=100, random_state=42,
                                        verbose=-1)
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostRegressor(iterations=100, random_state=42,
                                                verbose=0)
    return models


def get_param_grids():
    grids = {
        "Ridge":    {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "Lasso":    {"alpha": [0.01, 0.1, 1.0, 10.0]},
        "ElasticNet": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        "SVR":      {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", "auto"],
                     "epsilon": [0.1, 0.5, 1.0]},
        "KNN":      {"n_neighbors": [3, 5, 7, 9]},
        "RF":       {"n_estimators": [100, 200],
                     "max_features": ["sqrt", 0.3],
                     "min_samples_leaf": [1, 3]},
        "ET":       {"n_estimators": [100, 200], "max_features": ["sqrt", 0.3]},
        "GB":       {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                     "max_depth": [3, 5]},
        "XGB":      {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                     "max_depth": [3, 5]},
        "MLP":      {"hidden_layer_sizes": [(64,), (128,), (64, 32)],
                     "alpha": [1e-4, 1e-3]},
        "LGBM":     {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                     "num_leaves": [15, 31]},
        "CatBoost": {"iterations": [100, 200], "learning_rate": [0.05, 0.1],
                     "depth": [4, 6]},
    }
    return grids


def get_scalers():
    return {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler":   MinMaxScaler(),
        "RobustScaler":   RobustScaler(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH (regression, 5-fold CV)
# ─────────────────────────────────────────────────────────────────────────────
def grid_search_regressor(model, param_grid: dict,
                          X_train: pd.DataFrame, y_train: pd.Series,
                          cv: int = 5):
    """GridSearchCV with neg_mean_absolute_error, returns best estimator."""
    if not param_grid:
        model.fit(X_train, y_train)
        return model
    # n_jobs=1: avoids pickle protocol mismatch between Python 3.6 worker
    # subprocesses and the parent process on this HPC environment.
    gs = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error",
                      cv=cv, n_jobs=1, refit=True)
    gs.fit(X_train, y_train)
    print(f"    Best params: {gs.best_params_}  "
          f"(CV MAE={-gs.best_score_:.3f})")
    return gs.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def ccc(y_true, y_pred) -> float:
    """
    Lin's Concordance Correlation Coefficient.
    Range [-1, 1]; CCC≈0 means no better than a mean predictor.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mu_t = np.mean(y_true)
    mu_p = np.mean(y_pred)
    s2_t = np.var(y_true)
    s2_p = np.var(y_pred)
    cov  = np.mean((y_true - mu_t) * (y_pred - mu_p))
    denom = s2_t + s2_p + (mu_t - mu_p) ** 2
    return float(2.0 * cov / denom) if denom > 1e-10 else 0.0


def evaluate_regressor(model, X_test: pd.DataFrame,
                       y_test: pd.Series) -> dict:
    """Returns MAE, RMSE, R², Pearson r, Spearman rho."""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    pr, pp = pearsonr(y_test, y_pred)
    sr, sp = spearmanr(y_test, y_pred)
    return {
        "MAE":      round(mae,  4),
        "RMSE":     round(rmse, 4),
        "R2":       round(r2,   4),
        "Pearson_r": round(pr,  4),
        "Pearson_p": round(pp,  4),
        "Spearman_r": round(sr, 4),
        "Spearman_p": round(sp, 4),
        "n_test":   len(y_test),
    }


def evaluate_regressor_ext(model, X_test: pd.DataFrame,
                            y_test: pd.Series) -> dict:
    """Returns MAE, RMSE, R², Pearson r, Spearman rho, and CCC."""
    metrics = evaluate_regressor(model, X_test, y_test)
    y_pred  = model.predict(X_test)
    metrics["CCC"] = round(ccc(y_test.values, y_pred), 4)
    return metrics


def print_metrics(metrics: dict, label: str = ""):
    tag = f"  [{label}] " if label else "  "
    print(f"{tag}MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  "
          f"R²={metrics['R2']:.3f}  Pearson r={metrics['Pearson_r']:.3f}  "
          f"Spearman ρ={metrics['Spearman_r']:.3f}  n={metrics['n_test']}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ALIGNMENT (for cross-corpus)
# ─────────────────────────────────────────────────────────────────────────────
def align_features(X_source: pd.DataFrame,
                   X_target: pd.DataFrame) -> tuple:
    """
    Keep only the columns present in BOTH DataFrames.
    Returns (X_source_aligned, X_target_aligned).
    """
    common = sorted(set(X_source.columns) & set(X_target.columns))
    print(f"  Feature alignment: {len(X_source.columns)} source, "
          f"{len(X_target.columns)} target → {len(common)} common")
    if not common:
        raise ValueError("No common features between source and target datasets.")
    return X_source[common].copy(), X_target[common].copy()


# ─────────────────────────────────────────────────────────────────────────────
# FRAME-LEVEL AGGREGATION (for E-DAIC raw OpenFace CSVs)
# ─────────────────────────────────────────────────────────────────────────────
META_FRAME_COLS = {"frame", "timestamp", "confidence", "success", "face_id"}

def compute_entropy(series: pd.Series, bins: int = 10) -> float:
    data = series.dropna()
    if data.empty or data.nunique() < 2:
        return 0.0
    counts, _ = np.histogram(data, bins=bins)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-9)))


def aggregate_openface_frames(df: pd.DataFrame, bins: int = 10) -> dict:
    """
    18 statistical functionals per numeric OpenFace column.
    Naming: {col}__{stat}  — identical to ProposedCorpus processed CSVs.
    """
    agg = {}
    for col in df.columns:
        col_s = col.strip()
        if col_s in META_FRAME_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        agg[f"{col_s}__mean_orig"]    = series.mean()
        agg[f"{col_s}__std_orig"]     = series.std()
        agg[f"{col_s}__min_orig"]     = series.min()
        agg[f"{col_s}__max_orig"]     = series.max()
        agg[f"{col_s}__skew_orig"]    = series.skew()
        agg[f"{col_s}__kurt_orig"]    = series.kurt()
        agg[f"{col_s}__range_orig"]   = series.max() - series.min()
        agg[f"{col_s}__entropy_orig"] = compute_entropy(series, bins)

        deriv = series.diff().fillna(0)
        agg[f"{col_s}__mean_deriv"]    = deriv.mean()
        agg[f"{col_s}__std_deriv"]     = deriv.std()
        agg[f"{col_s}__min_deriv"]     = deriv.min()
        agg[f"{col_s}__max_deriv"]     = deriv.max()
        agg[f"{col_s}__skew_deriv"]    = deriv.skew()
        agg[f"{col_s}__kurt_deriv"]    = deriv.kurt()
        agg[f"{col_s}__range_deriv"]   = deriv.max() - deriv.min()
        agg[f"{col_s}__entropy_deriv"] = compute_entropy(deriv, bins)

        agg[f"{col_s}__rate_of_change"] = deriv.abs().mean()
        peaks, _ = find_peaks(series.values) if len(series) > 2 else ([], None)
        agg[f"{col_s}__peaks_count"] = len(peaks)

    return agg

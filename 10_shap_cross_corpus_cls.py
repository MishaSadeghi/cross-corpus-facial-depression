"""
10_shap_cross_corpus_cls.py
============================
SHAP analysis for the two best cross-corpus classification directions:

  Plot 1 — Proposed → E-DAIC  (LAT_ALL, Exp A, AUC=0.70)
  Plot 2 — E-DAIC  → Proposed (POS_ADK, Exp B, AUC=1.00)

Both plots use:
  - LGBM trained on ALL shared features (no RFE) — for interpretability
  - SHAP computed on the SOURCE training set (denser, more readable)
  - TreeExplainer (fast, exact)
  - Top 15 features by mean |SHAP| shown

Classification metrics reported in the paper are from the RFE-selected models.
This script trains separate "interpretability" models solely for the SHAP figures.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from shared_regression_pipeline import (
    EMPKINS_PHASE_FEATURES, MASTER_DATA_PATH, EDAIC_AGGREGATED_PATH,
    EMPKINS_SPLITS_DIR, EXCLUDE_COLS,
    load_shared_split, normalize_id, align_features,
    rename_empkins_to_openface,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
CLS_BASE   = (
    "/home/woody/empk/empk004h/D02_dataset/D02_final_results_and_models/"
    "KDD_paper/cross_corpus_classification"
)
OUTPUT_DIR = os.path.join(CLS_BASE, "shap_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_TOP_FEATURES = 15   # features to show in beeswarm

# ── Publication plot settings ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
})

# ── Feature name map ───────────────────────────────────────────────────────────
_AU_LABEL = {
    "AU01": "AU01 (inner brow raise)",  "AU02": "AU02 (outer brow raise)",
    "AU04": "AU04 (brow lowerer)",      "AU05": "AU05 (upper lid raiser)",
    "AU06": "AU06 (cheek raiser)",      "AU07": "AU07 (lid tightener)",
    "AU09": "AU09 (nose wrinkler)",     "AU10": "AU10 (upper lip raiser)",
    "AU12": "AU12 (lip corner puller)", "AU14": "AU14 (dimpler)",
    "AU15": "AU15 (lip corner depr.)",  "AU17": "AU17 (chin raiser)",
    "AU20": "AU20 (lip stretcher)",     "AU23": "AU23 (lip tightener)",
    "AU25": "AU25 (lips part)",         "AU26": "AU26 (jaw drop)",
    "AU45": "AU45 (blink rate)",
}
_STAT_LABEL = {
    "mean_orig":     "mean",      "std_orig":      "std",
    "median_orig":   "median",    "min_orig":      "min",
    "max_orig":      "max",       "range_orig":    "range",
    "entropy_orig":  "entropy",   "kurtosis_orig": "kurtosis",
    "skewness_orig": "skewness",  "rate_of_change":"rate-of-change",
    "mean_deriv":    "mean Δ",    "std_deriv":     "std Δ",
    "entropy_deriv": "entropy Δ", "range_deriv":   "range Δ",
    "kurtosis_deriv":"kurtosis Δ",
}

def clean_name(raw):
    """AU12_r__std_orig → 'AU12 (lip corner puller)  std'"""
    if "__" not in raw:
        return raw.replace("_", " ")
    sig, _, stat_raw = raw.partition("__")
    stat = _STAT_LABEL.get(stat_raw, stat_raw.replace("_", " "))

    # AU intensity / presence
    for prefix in ("AU", ):
        if sig.startswith(prefix):
            code = sig[:4]          # e.g. AU12
            suffix = sig[4:]        # _r or _c
            kind = "int." if "_r" in suffix else "pres."
            label = _AU_LABEL.get(code, code)
            return f"{label} [{kind} {stat}]"

    # gaze / head pose
    sig_clean = (sig
                 .replace("gaze_angle_x", "Gaze angle x")
                 .replace("gaze_angle_y", "Gaze angle y")
                 .replace("gaze_0_x", "Gaze0 x").replace("gaze_0_y", "Gaze0 y")
                 .replace("gaze_0_z", "Gaze0 z")
                 .replace("gaze_1_x", "Gaze1 x").replace("gaze_1_y", "Gaze1 y")
                 .replace("gaze_1_z", "Gaze1 z")
                 .replace("pose_Rx", "Head rot. x").replace("pose_Ry", "Head rot. y")
                 .replace("pose_Rz", "Head rot. z")
                 .replace("pose_Tx", "Head trans. x").replace("pose_Ty", "Head trans. y")
                 .replace("pose_Tz", "Head trans. z")
                 .replace("_", " "))
    return f"{sig_clean} [{stat}]"


# ── Data loaders ───────────────────────────────────────────────────────────────
def _load_empkins_phase(phase, condition):
    feat_df = pd.read_csv(EMPKINS_PHASE_FEATURES[phase])
    feat_df.columns = feat_df.columns.str.strip()
    feat_df["ID"] = feat_df["ID"].apply(normalize_id)
    drop_cols = [c for c in EXCLUDE_COLS if c in feat_df.columns and c != "ID"]
    feat_df.drop(columns=drop_cols, inplace=True)
    feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()]

    master_df = pd.read_csv(MASTER_DATA_PATH)
    master_df.columns = master_df.columns.str.strip()
    master_df["ID"] = master_df["ID"].apply(normalize_id)
    master_df.rename(columns={"label": "ground_truth_label"}, inplace=True)
    if condition == "ALL_CONDITIONS":
        mf = master_df.copy()
    else:
        mf = master_df[master_df["condition"] == condition].copy()

    merged = pd.merge(feat_df, mf[["ID", "ground_truth_label"]], on="ID", how="inner")
    merged = merged.dropna(subset=["ground_truth_label"]).set_index("ID")
    feat_cols = [c for c in merged.columns if c not in EXCLUDE_COLS]
    X = merged[feat_cols].select_dtypes(include=[np.number])
    X = rename_empkins_to_openface(X)
    X = X.loc[:, ~X.columns.duplicated()]
    y = merged["ground_truth_label"].astype(int)
    X = X.groupby(X.index).mean()
    y = y.groupby(y.index).first()
    return X, y


def load_edaic():
    df = pd.read_csv(EDAIC_AGGREGATED_PATH)
    df.columns = df.columns.str.strip()
    df["participant_id"] = df["participant_id"].astype(str)
    split_col = df["split"].str.strip().values
    y = df["depressed"].astype(int)
    feat_cols = [c for c in df.columns
                 if c not in {"participant_id", "split", "depressed",
                              "phq_score", "gender"}]
    X = df[feat_cols].copy()
    X.index = df["participant_id"]
    y.index = df["participant_id"]
    return X, y, split_col


# ── SHAP beeswarm (publication quality) ───────────────────────────────────────
def save_beeswarm(shap_values, X_df, clean_names, title, fname_base, n_top=N_TOP_FEATURES):
    """
    shap_values : np.ndarray (n_samples, n_features)
    X_df        : pd.DataFrame with same columns
    clean_names : list[str]  readable feature labels
    """
    # Sort features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:n_top]

    sv_top   = shap_values[:, order]
    X_top    = X_df.iloc[:, order]
    names_top = [clean_names[i] for i in order]

    fig_h = max(4.0, 0.58 * n_top + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    plt.sca(ax)

    shap.summary_plot(
        sv_top, X_top,
        feature_names=names_top,
        max_display=n_top,
        show=False,
        plot_size=None,
        color_bar=True,
        alpha=0.80,
        plot_type="dot",
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("SHAP value  (→ increases P(depressed))", fontsize=11)
    ax.axvline(0, color="#888888", linewidth=0.9, linestyle="--", zorder=0)
    ax.tick_params(axis="both", labelsize=10)

    # Colorbar tweak
    if len(fig.axes) > 1:
        fig.axes[-1].set_ylabel("Feature value", fontsize=10)

    plt.tight_layout(pad=1.4)
    for ext in ("pdf", "png"):
        out = os.path.join(OUTPUT_DIR, f"{fname_base}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Proposed → E-DAIC  (LAT_ALL, Exp A, LGBM interpretability model)
# SHAP on SOURCE training data (all Proposed/Latency/All-conditions)
# ══════════════════════════════════════════════════════════════════════════════
def run_plot1():
    print("\n" + "="*70)
    print("  Plot 1: Proposed → E-DAIC  (LAT_ALL, Exp A)")
    print("  Full-feature LGBM · SHAP on Proposed training set")
    print("="*70)

    # Load Proposed (latency, ALL_CONDITIONS)
    print("  Loading EmpkinS LAT_ALL ...")
    X_emp, y_emp = _load_empkins_phase("latency", "ALL_CONDITIONS")
    print(f"  EmpkinS all: {X_emp.shape}, dep={y_emp.sum()}/{len(y_emp)}")

    # Load E-DAIC (for feature alignment only)
    print("  Loading E-DAIC ...")
    X_daic, y_daic, split_col = load_edaic()
    X_daic_te = X_daic[split_col == "test"]

    # Align features
    X_emp_al, _ = align_features(X_emp, X_daic_te)

    # Impute + scale (on training data)
    imp    = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_tr   = pd.DataFrame(
        scaler.fit_transform(imp.fit_transform(X_emp_al)),
        columns=X_emp_al.columns, index=X_emp_al.index
    ).astype(np.float32)
    y_tr = y_emp.loc[X_tr.index]

    print(f"  Training features: {X_tr.shape}  dep={y_tr.sum()}/{len(y_tr)}")

    # Train interpretability LGBM (no RFE, all shared features)
    print("  Training LGBM (interpretability model) ...")
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                          num_leaves=31, random_state=42, verbose=-1)
    lgbm.fit(X_tr.values, y_tr.values)

    # SHAP on training set (source domain)
    print("  Computing SHAP (TreeExplainer on training set) ...")
    explainer  = shap.TreeExplainer(lgbm)
    shap_vals  = explainer.shap_values(X_tr.values)
    if isinstance(shap_vals, list):
        sv = shap_vals[1]   # class 1 = depressed
    else:
        sv = shap_vals

    print(f"  SHAP shape: {sv.shape}")
    clean_names = [clean_name(c) for c in X_tr.columns]

    save_beeswarm(
        sv, X_tr, clean_names,
        title="Proposed → E-DAIC  (Lat./All, AUC = 0.70)\n"
              "Source: Proposed corpus training set",
        fname_base="shap_plot1_proposed2edaic_LAT_ALL_v2",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: E-DAIC → Proposed  (POS_ADK, Exp B, LGBM interpretability model)
# SHAP on SOURCE training data (E-DAIC train+dev, n=219)
# ══════════════════════════════════════════════════════════════════════════════
def run_plot2():
    print("\n" + "="*70)
    print("  Plot 2: E-DAIC → Proposed  (POS_ADK, Exp B)")
    print("  Full-feature LGBM · SHAP on E-DAIC training set")
    print("="*70)

    # Load E-DAIC train+dev
    print("  Loading E-DAIC ...")
    X_daic, y_daic, split_col = load_edaic()
    X_daic_tr = X_daic[split_col != "test"]
    y_daic_tr = y_daic[split_col != "test"]
    print(f"  E-DAIC train+dev: {X_daic_tr.shape}, dep={y_daic_tr.sum()}/{len(y_daic_tr)}")

    # Load EmpkinS POS_ADK (for feature alignment only)
    print("  Loading EmpkinS POS_ADK ...")
    X_emp, y_emp = _load_empkins_phase("positive_training", "ADK")
    _, test_ids  = load_shared_split("ADK")
    X_emp_te     = X_emp[X_emp.index.isin(test_ids)]

    # Align features
    X_daic_al, _ = align_features(X_daic_tr, X_emp_te)

    # Impute + scale (on E-DAIC training data)
    imp    = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_tr   = pd.DataFrame(
        scaler.fit_transform(imp.fit_transform(X_daic_al)),
        columns=X_daic_al.columns, index=X_daic_al.index
    ).astype(np.float32)
    y_tr = y_daic_tr.loc[X_tr.index]

    print(f"  Training features: {X_tr.shape}  dep={y_tr.sum()}/{len(y_tr)}")

    # Train interpretability LGBM
    print("  Training LGBM (interpretability model) ...")
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                          num_leaves=31, random_state=42, verbose=-1)
    lgbm.fit(X_tr.values, y_tr.values)

    # SHAP on training set (source domain)
    print("  Computing SHAP (TreeExplainer on training set) ...")
    explainer = shap.TreeExplainer(lgbm)
    shap_vals = explainer.shap_values(X_tr.values)
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals

    print(f"  SHAP shape: {sv.shape}")
    clean_names = [clean_name(c) for c in X_tr.columns]

    save_beeswarm(
        sv, X_tr, clean_names,
        title="E-DAIC → Proposed  (Pos./AFE, AUC = 1.00)\n"
              "Source: E-DAIC training set",
        fname_base="shap_plot2_edaic2proposed_POS_ADK_v2",
    )


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_plot1()
    run_plot2()
    print("\nDone. Plots saved to:", OUTPUT_DIR)

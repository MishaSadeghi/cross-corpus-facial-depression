"""
11_result_visualizations.py
============================
Three publication-quality result figures:

  Fig A — AUC heatmap: 7 configs × 5 experiments, panels for Setting 1 & 2
  Fig B — Domain-gap slope chart: within-corpus vs cross-corpus AUC per config
  Fig C — Setting 1 vs Setting 2 scatter: does label matching help?

All figures saved to OUTPUT_DIR as PDF + PNG.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCID_CSV = (
    "/home/woody/empk/empk004h/D02_dataset/D02_final_results_and_models/"
    "KDD_paper/cross_corpus_classification/FINAL_COMPARISON_cls_20260423-180740.csv"
)
PHQ8_CSV = (
    "/home/woody/empk/empk004h/D02_dataset/D02_final_results_and_models/"
    "KDD_paper/cross_corpus_classification_phq8/FINAL_COMPARISON_phq8cls_20260423-180741.csv"
)
OUTPUT_DIR = (
    "/home/woody/empk/empk004h/D02_dataset/D02_final_results_and_models/"
    "KDD_paper/cross_corpus_classification/result_plots"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Display labels ─────────────────────────────────────────────────────────────
CONFIG_LABELS = {
    "LAT_ADK":   "Lat./AFE",
    "LAT_SHAM":  "Lat./SHAM",
    "LAT_ALL":   "Lat./All",
    "POS_ADK":   "Pos./AFE",
    "NEG_CRADK": "Neg./CR+AFE",
    "NEG_CR":    "Neg./CR",
    "NEG_SHAM":  "Neg./SHAM",
}
CONFIGS = list(CONFIG_LABELS.keys())

EXP_LABELS = {
    "A: EmpkinS-ALL→E-DAIC-test":        "A: Prop.ₐₗₗ→E-DAIC",
    "C1: EmpkinS-train→E-DAIC-test":     "C1: Prop.→E-DAIC",
    "B: E-DAIC-ALL→EmpkinS-test":        "B: E-DAIC→Prop.",
    "D1: EmpkinS-train→EmpkinS-test":    "D1: Prop.→Prop. (within)",
    "D2: E-DAIC-train→E-DAIC-test":      "D2: E-DAIC→E-DAIC (within)",
}
EXP_ORDER = list(EXP_LABELS.keys())

# ── Publication plot settings ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    df1 = pd.read_csv(SCID_CSV);  df1["setting"] = "Setting 1 (SCID)"
    df2 = pd.read_csv(PHQ8_CSV);  df2["setting"] = "Setting 2 (PHQ-8)"
    df  = pd.concat([df1, df2], ignore_index=True)
    df  = df[df["Experiment"].isin(EXP_ORDER)]
    df  = df[df["config"].isin(CONFIGS)]
    return df1, df2, df

df_scid, df_phq8, df_all = load_data()


def save_fig(fig, fname_base):
    for ext in ("pdf", "png"):
        out = os.path.join(OUTPUT_DIR, f"{fname_base}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig A — AUC Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def _draw_heatmap(df_scid, df_phq8, cmap, vmin, vmax, fname_base, text_threshold_lo, text_threshold_hi):
    """Shared heatmap drawing logic for different colormaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, (df_src, title) in zip(axes, [
        (df_scid, "Setting 1: Proposed (SCID) → E-DAIC (PHQ-8≥10)"),
        (df_phq8, "Setting 2: PHQ-8≥10 for both corpora"),
    ]):
        mat = np.full((len(CONFIGS), len(EXP_ORDER)), np.nan)
        for ci, cfg in enumerate(CONFIGS):
            for ei, exp in enumerate(EXP_ORDER):
                row = df_src[(df_src["config"] == cfg) &
                             (df_src["Experiment"] == exp)]
                if not row.empty:
                    mat[ci, ei] = row.iloc[0]["AUC"]

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")

        for ci in range(len(CONFIGS)):
            for ei in range(len(EXP_ORDER)):
                v = mat[ci, ei]
                if not np.isnan(v):
                    color = "white" if (v < text_threshold_lo or v > text_threshold_hi) else "black"
                    ax.text(ei, ci, f"{v:.2f}", ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(range(len(EXP_ORDER)))
        ax.set_xticklabels([EXP_LABELS[e] for e in EXP_ORDER],
                           rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(CONFIGS)))
        ax.set_yticklabels([CONFIG_LABELS[c] for c in CONFIGS], fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)

        ax.axvline(1.5, color="white", linewidth=2)
        ax.axvline(2.5, color="white", linewidth=2)
        plt.colorbar(im, ax=ax, label="AUC", fraction=0.046, pad=0.04)

    fig.suptitle("Cross-Corpus Classification AUC", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, fname_base)


def plot_heatmap():
    print("\n[Fig A] AUC Heatmap — 3 colormap variants ...")

    # Variant 1: Blues (single-hue, professional, white=chance dark=perfect)
    _draw_heatmap(df_scid, df_phq8,
                  cmap="Blues", vmin=0.4, vmax=1.0,
                  fname_base="figA_heatmap_Blues",
                  text_threshold_lo=0.50, text_threshold_hi=0.82)

    # Variant 2: viridis (perceptually uniform, colorblind-safe)
    _draw_heatmap(df_scid, df_phq8,
                  cmap="viridis", vmin=0.4, vmax=1.0,
                  fname_base="figA_heatmap_viridis",
                  text_threshold_lo=0.50, text_threshold_hi=0.82)

    # Variant 3: YlOrRd (yellow-orange-red, warm "heat" reading)
    _draw_heatmap(df_scid, df_phq8,
                  cmap="YlOrRd", vmin=0.4, vmax=1.0,
                  fname_base="figA_heatmap_YlOrRd",
                  text_threshold_lo=0.50, text_threshold_hi=0.82)

    # Variant 4: custom — white at chance (0.5), dark teal at 1.0
    from matplotlib.colors import LinearSegmentedColormap
    teal_cmap = LinearSegmentedColormap.from_list(
        "WhiteTeal", ["#f7f7f7", "#c7eae5", "#35978f", "#003c30"], N=256
    )
    _draw_heatmap(df_scid, df_phq8,
                  cmap=teal_cmap, vmin=0.4, vmax=1.0,
                  fname_base="figA_heatmap_teal",
                  text_threshold_lo=0.50, text_threshold_hi=0.82)


# ══════════════════════════════════════════════════════════════════════════════
# Fig B — Domain-Gap Slope Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_slope():
    print("\n[Fig B] Domain-gap slope chart ...")

    # Proposed direction: D1 (within) vs C1 (cross to E-DAIC)
    # E-DAIC direction:   D2 (within) vs B  (cross to Proposed)
    # Use Setting 1 only for clarity

    EXP_D1  = "D1: EmpkinS-train→EmpkinS-test"
    EXP_C1  = "C1: EmpkinS-train→E-DAIC-test"
    EXP_D2  = "D2: E-DAIC-train→E-DAIC-test"
    EXP_B   = "B: E-DAIC-ALL→EmpkinS-test"

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(CONFIGS)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)

    # Panel 1: Proposed source (D1 within → C1 cross)
    ax = axes[0]
    for ci, cfg in enumerate(CONFIGS):
        d1_row = df_scid[(df_scid["config"] == cfg) &
                         (df_scid["Experiment"] == EXP_D1)]
        c1_row = df_scid[(df_scid["config"] == cfg) &
                         (df_scid["Experiment"] == EXP_C1)]
        if d1_row.empty or c1_row.empty:
            continue
        y_d1 = d1_row.iloc[0]["AUC"]
        y_c1 = c1_row.iloc[0]["AUC"]
        lbl  = CONFIG_LABELS[cfg]
        col  = colors[ci]

        ax.plot([0, 1], [y_d1, y_c1], "o-", color=col, lw=1.8,
                ms=7, label=lbl, zorder=3)
        ax.annotate(f"{y_d1:.2f}", xy=(0, y_d1),
                    xytext=(-28, 0), textcoords="offset points",
                    fontsize=8, color=col, va="center")
        ax.annotate(f"{y_c1:.2f}", xy=(1, y_c1),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=col, va="center")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within-corpus\n(Proposed→Proposed)",
                         "Cross-corpus\n(Proposed→E-DAIC)"], fontsize=10)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(0.28, 1.08)
    ax.set_xlim(-0.3, 1.3)
    ax.set_title("Proposed source  (Setting 1)", fontsize=11, pad=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: E-DAIC source (D2 within → B cross)
    ax = axes[1]
    d2_auc = df_scid[df_scid["Experiment"] == EXP_D2]["AUC"].mean()  # config-independent

    for ci, cfg in enumerate(CONFIGS):
        b_row = df_scid[(df_scid["config"] == cfg) &
                        (df_scid["Experiment"] == EXP_B)]
        if b_row.empty:
            continue
        y_b  = b_row.iloc[0]["AUC"]
        lbl  = CONFIG_LABELS[cfg]
        col  = colors[ci]

        ax.plot([0, 1], [d2_auc, y_b], "o-", color=col, lw=1.8,
                ms=7, label=lbl, zorder=3)
        ax.annotate(f"{y_b:.2f}", xy=(1, y_b),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=col, va="center")

    # D2 annotation (same for all)
    ax.annotate(f"{d2_auc:.2f}", xy=(0, d2_auc),
                xytext=(-32, 0), textcoords="offset points",
                fontsize=9, color="black", va="center", fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within-corpus\n(E-DAIC→E-DAIC)",
                         "Cross-corpus\n(E-DAIC→Proposed)"], fontsize=10)
    ax.set_ylim(0.28, 1.08)
    ax.set_xlim(-0.3, 1.3)
    ax.set_title("E-DAIC source  (Setting 1)", fontsize=11, pad=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Domain Gap: Within-Corpus vs Cross-Corpus AUC",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, "figB_domain_gap_slope")


# ══════════════════════════════════════════════════════════════════════════════
# Fig C — Setting 1 vs Setting 2 Scatter
# ══════════════════════════════════════════════════════════════════════════════
def plot_setting_scatter():
    print("\n[Fig C] Setting 1 vs Setting 2 scatter ...")

    # Merge on (config, Experiment)
    merged = pd.merge(
        df_scid[["config", "Experiment", "AUC"]].rename(columns={"AUC": "AUC_s1"}),
        df_phq8[["config", "Experiment", "AUC"]].rename(columns={"AUC": "AUC_s2"}),
        on=["config", "Experiment"]
    )
    merged = merged[merged["Experiment"].isin(EXP_ORDER)]

    exp_colors = {
        "A: EmpkinS-ALL→E-DAIC-test":     "#2166ac",
        "C1: EmpkinS-train→E-DAIC-test":  "#4dac26",
        "B: E-DAIC-ALL→EmpkinS-test":     "#d6604d",
        "D1: EmpkinS-train→EmpkinS-test": "#8073ac",
        "D2: E-DAIC-train→E-DAIC-test":   "#878787",
    }
    exp_markers = {
        "A: EmpkinS-ALL→E-DAIC-test":     "o",
        "C1: EmpkinS-train→E-DAIC-test":  "s",
        "B: E-DAIC-ALL→EmpkinS-test":     "^",
        "D1: EmpkinS-train→EmpkinS-test": "D",
        "D2: E-DAIC-train→E-DAIC-test":   "P",
    }

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    for exp in EXP_ORDER:
        sub = merged[merged["Experiment"] == exp]
        if sub.empty:
            continue
        col = exp_colors.get(exp, "gray")
        mkr = exp_markers.get(exp, "o")
        ax.scatter(sub["AUC_s1"], sub["AUC_s2"],
                   c=col, marker=mkr, s=70, alpha=0.85, zorder=3,
                   label=EXP_LABELS[exp])

    # Diagonal
    lims = [0.38, 1.02]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Equal performance", zorder=0)
    ax.fill_between(lims, lims, [1.02]*2, color="#d6604d", alpha=0.06)
    ax.fill_between(lims, [0.38]*2, lims, color="#2166ac", alpha=0.06)
    ax.text(0.41, 0.96, "Setting 2 better\n(matched labels help)",
            fontsize=8, color="#d6604d", va="top")
    ax.text(0.90, 0.42, "Setting 1 better\n(SCID more reliable)",
            fontsize=8, color="#2166ac", ha="right")

    # Chance line
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, zorder=0)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, zorder=0)

    ax.set_xlabel("AUC — Setting 1 (SCID labels, label mismatch)", fontsize=11)
    ax.set_ylabel("AUC — Setting 2 (PHQ-8≥10, matched labels)", fontsize=11)
    ax.set_title("Does Label Matching Help?\nSetting 1 vs Setting 2 AUC",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig, "figC_setting1_vs_setting2")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_heatmap()
    plot_slope()
    plot_setting_scatter()
    print("\nDone. Figures saved to:", OUTPUT_DIR)

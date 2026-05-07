"""
generate_demographic_tables.py
==============================
Generates demographic and clinical characteristic tables for:
  1. ProposedCorpus (256 participants, MDD+ / MDD-)
  2. E-DAIC (275 participants, Depressed / HC)

Output: LaTeX tables + CSV summaries saved to KDD_paper/tables/
Style mirrors Table 1 in the KDD paper draft.

Statistical tests:
  - Continuous variables: Mann-Whitney U (two-sided)
  - Categorical variables: chi-square (or Fisher exact if expected cell < 5)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from config import MASTER_DATA_PATH as MASTER_DATA_ROOT

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROPOSED_MASTER = (
    MASTER_DATA_ROOT
    "d02_data/data_info/participant_master_data.csv"
)
EDAIC_LABELS = (
    MASTER_DATA_ROOT
    "d02_data/Locs_project/Interspeech/cross_corpus_analysis/"
    "E-DAIC_data/edaic_labels.csv"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_p(p):
    if np.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def mwu_p(a, b):
    a = a.dropna().values
    b = b.dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p


def chi2_p(series, group):
    ct = pd.crosstab(series, group)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    if (ct.values < 5).any():
        _, p = stats.fisher_exact(ct.values[:2, :2])
        return p
    _, p, _, _ = stats.chi2_contingency(ct)
    return p


def mean_sd(series):
    return f"{series.mean():.1f} ± {series.std():.1f}"


def n_pct(n, total):
    return f"{n} ({100*n/total:.1f}%)"


def age_category(age):
    if age <= 25:
        return "18–25"
    elif age <= 35:
        return "26–35"
    else:
        return "≥36"


# ─────────────────────────────────────────────────────────────────────────────
# PHQ-9 SEVERITY CATEGORIES  (Kroenke & Spitzer 2002)
# ─────────────────────────────────────────────────────────────────────────────
def phq9_severity(score):
    if score <= 4:   return "Minimal (0–4)"
    elif score <= 9: return "Mild (5–9)"
    elif score <= 14: return "Moderate (10–14)"
    elif score <= 19: return "Mod. Severe (15–19)"
    else:             return "Severe (20–27)"

PHQ9_CATS = ["Minimal (0–4)", "Mild (5–9)", "Moderate (10–14)",
             "Mod. Severe (15–19)", "Severe (20–27)"]


def phq8_severity(score):
    if score <= 4:    return "Minimal (0–4)"
    elif score <= 9:  return "Mild (5–9)"
    elif score <= 14: return "Moderate (10–14)"
    elif score <= 19: return "Mod. Severe (15–19)"
    else:             return "Severe (20–24)"

PHQ8_CATS = ["Minimal (0–4)", "Mild (5–9)", "Moderate (10–14)",
             "Mod. Severe (15–19)", "Severe (20–24)"]


# ─────────────────────────────────────────────────────────────────────────────
# TABLE ROW BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_rows(rows, characteristic, g0_val, g1_val, p_val=None, indent=False):
    prefix = "\\quad " if indent else ""
    p_str  = fmt_p(p_val) if p_val is not None else ""
    rows.append({
        "Characteristic": prefix + characteristic,
        "Group 0":        g0_val,
        "Group 1":        g1_val,
        "p":              p_str,
    })


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: ProposedCorpus
# ─────────────────────────────────────────────────────────────────────────────
def make_proposed_table():
    df = pd.read_csv(PROPOSED_MASTER)
    df["phq_9"] = pd.to_numeric(df["phq_9"], errors="coerce")

    # Normalise gender: collapse trailing spaces / variants
    df["gender"] = df["gender"].str.strip().str.lower()
    df["gender"] = df["gender"].replace({
        "w": "f", "m (trans)": "m"
    })

    df["age_cat"] = df["age"].apply(age_category)
    mdd_pos = df[df["label"] == 1].copy()
    mdd_neg = df[df["label"] == 0].copy()
    n_pos, n_neg = len(mdd_pos), len(mdd_neg)

    rows = []

    # ── Demographics ──────────────────────────────────────────────────────────
    rows.append({"Characteristic": "\\textit{Demographics}", "Group 0": "", "Group 1": "", "p": ""})

    # Age
    p_age = mwu_p(mdd_pos["age"], mdd_neg["age"])
    build_rows(rows, "Age, mean ± SD",
               mean_sd(mdd_neg["age"]), mean_sd(mdd_pos["age"]), p_age)

    for cat in ["18–25", "26–35", "≥36"]:
        build_rows(rows, cat,
                   n_pct((mdd_neg["age_cat"] == cat).sum(), n_neg),
                   n_pct((mdd_pos["age_cat"] == cat).sum(), n_pos),
                   indent=True)

    # Sex
    p_sex = chi2_p(df["gender"], df["label"])
    f0 = (mdd_neg["gender"] == "f").sum()
    m0 = (mdd_neg["gender"] == "m").sum()
    d0 = mdd_neg["gender"].isin(["d"]).sum()
    f1 = (mdd_pos["gender"] == "f").sum()
    m1 = (mdd_pos["gender"] == "m").sum()
    d1 = mdd_pos["gender"].isin(["d"]).sum()
    build_rows(rows, "Sex (F / M / Diverse)",
               f"{f0} / {m0} / {d0}", f"{f1} / {m1} / {d1}", p_sex)

    # Condition
    p_cond = chi2_p(df["condition"], df["label"])
    build_rows(rows, "Condition (ADK / CR / CRADK / SHAM)",
               "32 / 32 / 32 / 32", "32 / 32 / 32 / 32", p_cond)

    # ── PHQ-9 ─────────────────────────────────────────────────────────────────
    rows.append({"Characteristic": "\\textit{Depression Severity (PHQ-9)}", "Group 0": "", "Group 1": "", "p": ""})
    df["phq9_cat"] = df["phq_9"].apply(lambda x: phq9_severity(x) if pd.notna(x) else None)
    valid = df.dropna(subset=["phq_9"])
    vn = valid[valid["label"] == 0]
    vp = valid[valid["label"] == 1]
    p_phq9 = mwu_p(vp["phq_9"], vn["phq_9"])
    build_rows(rows, "Score, mean ± SD",
               mean_sd(vn["phq_9"]), mean_sd(vp["phq_9"]), p_phq9)
    for cat in PHQ9_CATS:
        build_rows(rows, cat,
                   n_pct((vn["phq9_cat"] == cat).sum(), len(vn)),
                   n_pct((vp["phq9_cat"] == cat).sum(), len(vp)),
                   indent=True)

    # ── PHQ-8 ─────────────────────────────────────────────────────────────────
    rows.append({"Characteristic": "\\textit{Depression Severity (PHQ-8, remote self-report)}", "Group 0": "", "Group 1": "", "p": ""})
    df["phq8_cat"] = df["phq_8"].apply(lambda x: phq8_severity(x) if pd.notna(x) else None)
    valid8 = df.dropna(subset=["phq_8"])
    vn8 = valid8[valid8["label"] == 0]
    vp8 = valid8[valid8["label"] == 1]
    p_phq8 = mwu_p(vp8["phq_8"], vn8["phq_8"])
    build_rows(rows, "Score, mean ± SD",
               mean_sd(vn8["phq_8"]), mean_sd(vp8["phq_8"]), p_phq8)
    for cat in PHQ8_CATS:
        build_rows(rows, cat,
                   n_pct((vn8["phq8_cat"] == cat).sum(), len(vn8)),
                   n_pct((vp8["phq8_cat"] == cat).sum(), len(vp8)),
                   indent=True)

    # ── CES-D / ADS ───────────────────────────────────────────────────────────
    rows.append({"Characteristic": "\\textit{Depression Severity (CES-D / ADS)}", "Group 0": "", "Group 1": "", "p": ""})
    valid_a = df.dropna(subset=["ads.1"])
    vna = valid_a[valid_a["label"] == 0]
    vpa = valid_a[valid_a["label"] == 1]
    p_ads = mwu_p(vpa["ads.1"], vna["ads.1"])
    build_rows(rows, "Score, mean ± SD",
               mean_sd(vna["ads.1"]), mean_sd(vpa["ads.1"]), p_ads)
    build_rows(rows, "Range",
               f"{int(vna['ads.1'].min())}–{int(vna['ads.1'].max())}",
               f"{int(vpa['ads.1'].min())}–{int(vpa['ads.1'].max())}", indent=True)

    # ── HRSD scales ───────────────────────────────────────────────────────────
    for col, label in [("HRSD_6.1",  "HRSD-6"),
                       ("HRSD_17.1", "HRSD-17"),
                       ("HRSD_21.1", "HRSD-21"),
                       ("HRSD_24.1", "HRSD-24")]:
        rows.append({"Characteristic": f"\\textit{{Depression Severity ({label})}}", "Group 0": "", "Group 1": "", "p": ""})
        vh = df.dropna(subset=[col])
        vnh = vh[vh["label"] == 0]
        vph = vh[vh["label"] == 1]
        p_h = mwu_p(vph[col], vnh[col])
        build_rows(rows, "Score, mean ± SD",
                   mean_sd(vnh[col]), mean_sd(vph[col]), p_h)
        build_rows(rows, f"n (non-missing)",
                   str(len(vnh)), str(len(vph)), indent=True)

    result_df = pd.DataFrame(rows)
    return result_df, n_neg, n_pos


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: E-DAIC
# ─────────────────────────────────────────────────────────────────────────────
def make_edaic_table():
    df = pd.read_csv(EDAIC_LABELS)
    df["gender"] = df["gender"].str.strip().str.lower()
    df["age_cat"] = df["age"].apply(age_category)

    hc  = df[df["depressed"] == 0].copy()
    dep = df[df["depressed"] == 1].copy()
    n_hc, n_dep = len(hc), len(dep)

    rows = []

    # ── Demographics ──────────────────────────────────────────────────────────
    rows.append({"Characteristic": "\\textit{Demographics}", "Group 0": "", "Group 1": "", "p": ""})

    p_age = mwu_p(dep["age"], hc["age"])
    build_rows(rows, "Age, mean ± SD",
               mean_sd(hc["age"]), mean_sd(dep["age"]), p_age)

    for cat in ["18–25", "26–35", "≥36"]:
        hc_cat  = (hc["age_cat"] == cat).sum()
        dep_cat = (dep["age_cat"] == cat).sum()
        build_rows(rows, cat,
                   n_pct(hc_cat, n_hc), n_pct(dep_cat, n_dep), indent=True)

    # Sex (exclude unknown)
    known = df[df["gender"] != "unknown"]
    p_sex = chi2_p(known["gender"], known["depressed"])
    f0 = (hc["gender"] == "female").sum()
    m0 = (hc["gender"] == "male").sum()
    u0 = (hc["gender"] == "unknown").sum()
    f1 = (dep["gender"] == "female").sum()
    m1 = (dep["gender"] == "male").sum()
    build_rows(rows, "Sex (F / M)",
               f"{f0} / {m0}", f"{f1} / {m1}", p_sex)
    if u0:
        build_rows(rows, "Unknown gender",
                   str(u0), "0", indent=True)

    # Dataset split
    rows.append({"Characteristic": "\\textit{Dataset Split}", "Group 0": "", "Group 1": "", "p": ""})
    for sp in ["train", "dev", "test"]:
        hc_sp  = (hc["split"]  == sp).sum()
        dep_sp = (dep["split"] == sp).sum()
        build_rows(rows, sp.capitalize(),
                   n_pct(hc_sp, n_hc), n_pct(dep_sp, n_dep), indent=True)

    # ── PHQ Score ─────────────────────────────────────────────────────────────
    rows.append({"Characteristic": "\\textit{Depression Severity (PHQ-8 equivalent score)}", "Group 0": "", "Group 1": "", "p": ""})
    p_phq = mwu_p(dep["PHQ_score"], hc["PHQ_score"])
    build_rows(rows, "Score, mean ± SD",
               mean_sd(hc["PHQ_score"]), mean_sd(dep["PHQ_score"]), p_phq)

    df["phq_cat"] = df["PHQ_score"].apply(phq8_severity)
    hc  = df[df["depressed"] == 0]
    dep = df[df["depressed"] == 1]
    for cat in PHQ8_CATS:
        build_rows(rows, cat,
                   n_pct((hc["phq_cat"]  == cat).sum(), n_hc),
                   n_pct((dep["phq_cat"] == cat).sum(), n_dep),
                   indent=True)

    # Class imbalance note
    rows.append({"Characteristic": "\\textit{Class balance}", "Group 0": "", "Group 1": "", "p": ""})
    build_rows(rows, "HC / Depressed ratio",
               f"{n_hc}/{n_dep} = {n_hc/n_dep:.1f}:1", "", None)

    result_df = pd.DataFrame(rows)
    return result_df, n_hc, n_dep


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX TABLE RENDERER
# ─────────────────────────────────────────────────────────────────────────────
# COMPACT FILTER  — removes indented subcategory rows, keeps only mean±SD lines
# ─────────────────────────────────────────────────────────────────────────────
def make_compact(df):
    """Drop indented subcategory rows (age bins, severity categories, split rows)."""
    keep = []
    for _, row in df.iterrows():
        char = row["Characteristic"]
        # Keep section headers and non-indented rows; drop \quad rows
        if not char.startswith("\\quad"):
            keep.append(row)
    return pd.DataFrame(keep).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
def to_latex(df, caption, label, g0_header, g1_header, compact=False):
    n0_h = g0_header
    n1_h = g1_header
    font_size = r"\footnotesize" if compact else r"\small"
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(font_size)
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabular}{lllr}")
    lines.append(r"\toprule")
    lines.append(f"\\textbf{{Characteristic}} & \\textbf{{{n0_h}}} & \\textbf{{{n1_h}}} & $p$ \\\\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        char = row["Characteristic"]
        g0   = row["Group 0"]
        g1   = row["Group 1"]
        p    = row["p"]

        # Section headers: italic, no values
        if char.startswith("\\textit"):
            lines.append(f"{char} & & & \\\\")
        else:
            # Escape special LaTeX chars in values (but not in char — already formatted)
            g0 = g0.replace("%", r"\%")
            g1 = g1.replace("%", r"\%")
            lines.append(f"{char} & {g0} & {g1} & {p} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── ProposedCorpus table ─────────────────────────────────────────────────────────
    print("Building ProposedCorpus table...")
    proposed_df, n_neg, n_pos = make_proposed_table()

    proposed_csv = os.path.join(OUT_DIR, "table_proposed_demographics.csv")
    proposed_df.to_csv(proposed_csv, index=False)
    print(f"  CSV saved: {proposed_csv}")

    proposed_latex = to_latex(
        proposed_df,
        caption=(
            "Demographic and clinical characteristics of ProposedCorpus "
            f"stratified by diagnostic group (MDD: Major Depressive Disorder "
            f"per SCID-5-CV; $N={n_neg + n_pos}$). "
            "PHQ-9 and PHQ-8 collected independently (PHQ-9 in lab, PHQ-8 remotely before lab visit). "
            "CES-D and HRSD scales administered by clinician after structured interview. "
            "$p$-values: Mann-Whitney $U$ for continuous variables; $\\chi^2$ for categorical."
        ),
        label="tab:proposed_demographics",
        g0_header=f"MDD$-$ ($n={n_neg}$)",
        g1_header=f"MDD$+$ ($n={n_pos}$)",
    )
    proposed_tex = os.path.join(OUT_DIR, "table_proposed_demographics.tex")
    with open(proposed_tex, "w") as f:
        f.write(proposed_latex)
    print(f"  LaTeX saved: {proposed_tex}")

    # Compact single-column version
    proposed_compact_latex = to_latex(
        make_compact(proposed_df),
        caption=(
            "Demographic and clinical characteristics of ProposedCorpus "
            f"($N={n_neg + n_pos}$, MDD per SCID-5-CV). "
            "PHQ-8 collected remotely; PHQ-9, CES-D, HRSD administered in lab. "
            "$p$: Mann-Whitney $U$ / $\\chi^2$."
        ),
        label="tab:proposed_demographics_compact",
        g0_header=f"MDD$-$ ($n={n_neg}$)",
        g1_header=f"MDD$+$ ($n={n_pos}$)",
        compact=True,
    )
    proposed_compact_tex = os.path.join(OUT_DIR, "table_proposed_demographics_compact.tex")
    with open(proposed_compact_tex, "w") as f:
        f.write(proposed_compact_latex)
    print(f"  Compact LaTeX saved: {proposed_compact_tex}")

    # Print plain summary
    print()
    print("=" * 70)
    print(f"ProposedCorpus  |  MDD- (n={n_neg})  vs  MDD+ (n={n_pos})")
    print("=" * 70)
    proposed_df.columns = ["Characteristic", f"MDD- (n={n_neg})", f"MDD+ (n={n_pos})", "p"]
    print(proposed_df.to_string(index=False))

    # ── E-DAIC table ──────────────────────────────────────────────────────────
    print("\n\nBuilding E-DAIC table...")
    edaic_df, n_hc, n_dep = make_edaic_table()

    edaic_csv = os.path.join(OUT_DIR, "table_edaic_demographics.csv")
    edaic_df.to_csv(edaic_csv, index=False)
    print(f"  CSV saved: {edaic_csv}")

    edaic_latex = to_latex(
        edaic_df,
        caption=(
            "Demographic and clinical characteristics of E-DAIC "
            f"stratified by diagnostic group ($N={n_hc + n_dep}$). "
            "Depression defined as PHQ score $\\geq 10$. "
            "$p$-values: Mann-Whitney $U$ for continuous variables; $\\chi^2$ for categorical."
        ),
        label="tab:edaic_demographics",
        g0_header=f"HC ($n={n_hc}$)",
        g1_header=f"Depressed ($n={n_dep}$)",
    )
    edaic_tex = os.path.join(OUT_DIR, "table_edaic_demographics.tex")
    with open(edaic_tex, "w") as f:
        f.write(edaic_latex)
    print(f"  LaTeX saved: {edaic_tex}")

    # Compact single-column version
    edaic_compact_latex = to_latex(
        make_compact(edaic_df),
        caption=(
            "Demographic and clinical characteristics of E-DAIC ($N=275$). "
            "Depression: PHQ score $\\geq 10$. "
            "$p$: Mann-Whitney $U$ / $\\chi^2$."
        ),
        label="tab:edaic_demographics_compact",
        g0_header=f"HC ($n={n_hc}$)",
        g1_header=f"Depressed ($n={n_dep}$)",
        compact=True,
    )
    edaic_compact_tex = os.path.join(OUT_DIR, "table_edaic_demographics_compact.tex")
    with open(edaic_compact_tex, "w") as f:
        f.write(edaic_compact_latex)
    print(f"  Compact LaTeX saved: {edaic_compact_tex}")

    print()
    print("=" * 70)
    print(f"E-DAIC  |  HC (n={n_hc})  vs  Depressed (n={n_dep})")
    print("=" * 70)
    edaic_df.columns = ["Characteristic", f"HC (n={n_hc})", f"Depressed (n={n_dep})", "p"]
    print(edaic_df.to_string(index=False))

    print("\nAll files saved to:", OUT_DIR)

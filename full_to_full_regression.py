"""
08_full_to_full_regression.py
=====================================
Two additional cross-corpus experiments where one corpus is used in full
(no holdout) for training and the other corpus is used in full for testing.

E1: All Proposed (train+test, all participants) → All E-DAIC (all 275)
E2: All E-DAIC  (all 275)                       → All Proposed (all participants)

These are supplementary to the main experiments (A/B/C/D) in
06_cross_corpus_regression_v2.py, which respect official splits.

Note: E1/E2 are not directly comparable to published E-DAIC benchmarks
because E-DAIC test-split participants are included in training (E2) or
evaluated as a larger undivided set (E1).

Usage:
  python 08_full_to_full_regression.py
  python 08_full_to_full_regression.py --configs NEG_CR LAT_ADK
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from shared_regression_pipeline import (
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    align_features,
    get_param_grids,
    get_regressors,
    get_scalers,
    load_edaic_data,
)

# reuse the helpers from the v2 script
sys.path.insert(0, SCRIPT_DIR)
# import the proposed loaders directly from the v2 module
import importlib.util as _ilu
from config import RESULTS_ROOT
_spec = _ilu.spec_from_file_location(
    "cc_reg_v2",
    os.path.join(SCRIPT_DIR, "06_cross_corpus_regression_v2.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_proposed_reg          = _mod.load_proposed_reg
load_proposed_all_phases_reg = _mod.load_proposed_all_phases_reg
fit_and_eval_reg          = _mod.fit_and_eval_reg
SOURCE_CONFIGS            = _mod.SOURCE_CONFIGS

if CATBOOST_AVAILABLE:
    from catboost import CatBoostRegressor
if LIGHTGBM_AVAILABLE:
    from lightgbm import LGBMRegressor

OUTPUT_BASE = (
    RESULTS_ROOT
    "KDD_paper/cross_corpus_regression_full2full"
)

ALL_PHASES = ["latency", "emotion_induction_1", "negative_training", "positive_training"]


def run_config_full2full(config_name, config, timestamp, scalers, models, param_grids):
    save_dir = OUTPUT_BASE
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  FULL→FULL CROSS-CORPUS | Config: {}".format(config_name))
    print("  Phases: {}  Condition: {}".format(config["phases"], config["condition"]))
    print("=" * 80)

    # ── Load ALL Proposed ─────────────────────────────────────────────────────
    print("\n[1] Loading ALL Proposed (full corpus, no split) ...")
    if config_name == "ALL":
        X_emp, y_emp = load_proposed_all_phases_reg()
    else:
        phases = config["phases"]
        condition = config["condition"]
        if len(phases) == 1:
            X_emp, y_emp = load_proposed_reg(phases[0], condition)
        else:
            all_X, all_y = [], []
            for ph in phases:
                Xp, yp = load_proposed_reg(ph, condition)
                all_X.append(Xp)
                all_y.append(yp)
            common_cols = sorted(set.intersection(*[set(Xp.columns) for Xp in all_X]))
            X_emp = pd.concat([Xp[common_cols] for Xp in all_X])
            y_emp = pd.concat(all_y)

    print("  Proposed ALL: n={}, PHQ-8 mean={:.1f}".format(len(y_emp), y_emp.mean()))

    # ── Load ALL E-DAIC ───────────────────────────────────────────────────────
    print("\n[2] Loading ALL E-DAIC (full corpus, no split) ...")
    try:
        X_daic_all, y_daic_all, _, split_col = load_edaic_data(split_filter=None)
    except FileNotFoundError:
        print("  ERROR: E-DAIC aggregated file not found.")
        return {}

    print("  E-DAIC ALL: n={}, PHQ-8 mean={:.1f}".format(
        len(y_daic_all), y_daic_all.mean()))

    summary_rows = []

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT E1: All Proposed → All E-DAIC
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  EXPERIMENT E1: All Proposed → All E-DAIC (n_train={}, n_test={})".format(
        len(y_emp), len(y_daic_all)))
    print("─" * 80)

    X_emp_al, X_daic_al = align_features(X_emp, X_daic_all)

    imp_E1 = SimpleImputer(strategy="median")
    X_emp_imp = pd.DataFrame(
        imp_E1.fit_transform(X_emp_al),
        columns=X_emp_al.columns, index=X_emp_al.index
    ).astype(np.float32)
    X_daic_imp = pd.DataFrame(
        imp_E1.transform(X_daic_al),
        columns=X_daic_al.columns, index=X_daic_al.index
    ).astype(np.float32)

    _, best_E1 = fit_and_eval_reg(
        models, param_grids, scalers,
        X_emp_imp, y_emp,
        X_daic_imp, y_daic_all,
        experiment_tag="E1_empALL2daicALL_{}".format(config_name),
        save_dir=save_dir, timestamp=timestamp
    )
    if best_E1:
        summary_rows.append({
            "Experiment":   "E1: Proposed-ALL→E-DAIC-ALL",
            "Train_source": "Proposed ALL ({} participants)".format(len(set(y_emp.index))),
            "Test_source":  "E-DAIC ALL ({} participants)".format(len(y_daic_all)),
            "n_train":      len(y_emp),
            "n_test":       len(y_daic_all),
            "Best_model":   "{}/{}".format(best_E1["model_name"], best_E1["scaler"]),
            **best_E1["metrics"],
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT E2: All E-DAIC → All Proposed
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  EXPERIMENT E2: All E-DAIC → All Proposed (n_train={}, n_test={})".format(
        len(y_daic_all), len(y_emp)))
    print("─" * 80)

    X_daic_al2, X_emp_al2 = align_features(X_daic_all, X_emp)

    imp_E2 = SimpleImputer(strategy="median")
    X_daic_imp2 = pd.DataFrame(
        imp_E2.fit_transform(X_daic_al2),
        columns=X_daic_al2.columns, index=X_daic_al2.index
    ).astype(np.float32)
    X_emp_imp2 = pd.DataFrame(
        imp_E2.transform(X_emp_al2),
        columns=X_emp_al2.columns, index=X_emp_al2.index
    ).astype(np.float32)

    _, best_E2 = fit_and_eval_reg(
        models, param_grids, scalers,
        X_daic_imp2, y_daic_all,
        X_emp_imp2, y_emp,
        experiment_tag="E2_daicALL2empALL_{}".format(config_name),
        save_dir=save_dir, timestamp=timestamp
    )
    if best_E2:
        summary_rows.append({
            "Experiment":   "E2: E-DAIC-ALL→Proposed-ALL",
            "Train_source": "E-DAIC ALL ({} participants)".format(len(y_daic_all)),
            "Test_source":  "Proposed ALL ({} participants)".format(len(set(y_emp.index))),
            "n_train":      len(y_daic_all),
            "n_test":       len(y_emp),
            "Best_model":   "{}/{}".format(best_E2["model_name"], best_E2["scaler"]),
            **best_E2["metrics"],
        })

    # ── Config-level summary ──────────────────────────────────────────────────
    if summary_rows:
        summary_df   = pd.DataFrame(summary_rows)
        summary_path = os.path.join(
            save_dir,
            "SUMMARY_full2full_{}_{}.csv".format(config_name, timestamp)
        )
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 80)
        print("  FULL→FULL SUMMARY | Config: {}".format(config_name))
        print("=" * 80)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(summary_df[[
            "Experiment", "n_train", "Train_source", "n_test", "Test_source",
            "Best_model", "CCC", "MAE", "RMSE", "Pearson_r"
        ]].to_string(index=False))
        print("\n  Summary saved: {}".format(summary_path))

    return {r["Experiment"]: r for r in summary_rows}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-to-full cross-corpus PHQ-8 regression: all Proposed ↔ all E-DAIC"
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(SOURCE_CONFIGS.keys()),
        choices=list(SOURCE_CONFIGS.keys()),
        help="Source configurations (default: all)"
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
        result = run_config_full2full(
            cfg_name, SOURCE_CONFIGS[cfg_name],
            ts, scalers, models, grids
        )
        for exp_label, row in result.items():
            all_summaries.append({"config": cfg_name, **row})

    if all_summaries:
        final_df   = pd.DataFrame(all_summaries)
        final_path = os.path.join(
            OUTPUT_BASE,
            "FINAL_COMPARISON_full2full_{}.csv".format(ts)
        )
        final_df.to_csv(final_path, index=False)
        print("\n" + "=" * 80)
        print("  FINAL FULL→FULL COMPARISON — PHQ-8 REGRESSION")
        print("=" * 80)
        print(final_df[[
            "config", "Experiment",
            "n_train", "n_test",
            "Best_model", "CCC", "MAE", "RMSE", "Pearson_r"
        ]].to_string(index=False))
        print("\n  Final comparison saved: {}".format(final_path))

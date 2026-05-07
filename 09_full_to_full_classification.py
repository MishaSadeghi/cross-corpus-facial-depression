"""
09_full_to_full_classification.py
=====================================
Full-corpus-to-full-corpus cross-corpus classification:
  E1: All Proposed (all participants) → All E-DAIC (all 275)
  E2: All E-DAIC  (all 275)           → All Proposed (all participants)

Run for both label settings:
  Setting 1: Proposed uses SCID-5-CV labels; E-DAIC uses PHQ-8>=10 (mismatch)
  Setting 2: Both use PHQ-8>=10 threshold (matched)

Usage:
  python 09_full_to_full_classification.py
  python 09_full_to_full_classification.py --configs NEG_CR LAT_ADK
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

import importlib.util as _ilu

def _load_module(name, filename):
    spec = _ilu.spec_from_file_location(name, os.path.join(SCRIPT_DIR, filename))
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_scid_mod = _load_module("cls_scid", "05_cross_corpus_classification.py")
_phq8_mod = _load_module("cls_phq8", "07_cross_corpus_classification_phq8.py")

# SCID label loaders (Setting 1 Proposed side)
load_empkins_cls           = _scid_mod.load_empkins_cls
load_empkins_all_phases_cls = _scid_mod.load_empkins_all_phases_cls
# PHQ-8 label loaders (Setting 2 Proposed side)
load_empkins_cls_phq8           = _phq8_mod.load_empkins_cls_phq8
load_empkins_all_phases_cls_phq8 = _phq8_mod.load_empkins_all_phases_cls_phq8
# E-DAIC loader (same for both settings — always PHQ-8>=10)
load_edaic_cls  = _scid_mod.load_edaic_cls
# Shared classification pipeline
fit_and_eval_cls = _scid_mod.fit_and_eval_cls
get_classifiers  = _scid_mod.get_classifiers
get_cls_param_grids = _scid_mod.get_cls_param_grids
get_scalers      = _scid_mod.get_scalers
align_features   = _scid_mod.align_features
SOURCE_CONFIGS   = _scid_mod.SOURCE_CONFIGS

OUTPUT_BASE = (
    "/home/woody/empk/empk004h/D02_dataset/D02_final_results_and_models/"
    "KDD_paper/cross_corpus_classification_full2full"
)

ALL_PHASES = ["latency", "emotion_induction_1", "negative_training", "positive_training"]


def load_empkins_full(config_name, config, label_setting):
    """Load all Proposed participants for a given config and label setting."""
    phases    = config["phases"]
    condition = config["condition"]
    if label_setting == "scid":
        loader     = load_empkins_cls
        loader_all = load_empkins_all_phases_cls
    else:
        loader     = load_empkins_cls_phq8
        loader_all = load_empkins_all_phases_cls_phq8

    if config_name == "ALL":
        X_emp, y_emp = loader_all()
    elif len(phases) == 1:
        X_emp, y_emp = loader(phases[0], condition)
    else:
        all_X, all_y = [], []
        for ph in phases:
            Xp, yp = loader(ph, condition)
            all_X.append(Xp)
            all_y.append(yp)
        common_cols = sorted(set.intersection(*[set(Xp.columns) for Xp in all_X]))
        X_emp = pd.concat([Xp[common_cols] for Xp in all_X])
        y_emp = pd.concat(all_y)

    # one label per participant
    y_emp = y_emp.groupby(y_emp.index).first()
    X_emp = X_emp.loc[~X_emp.index.duplicated(keep="first")]
    return X_emp, y_emp


def run_setting(config_name, config, label_setting,
                X_daic_all, y_daic_all,
                timestamp, scalers, models, param_grids):

    tag = label_setting.upper()
    print("\n" + "-" * 80)
    print("  Config: {}  |  Label setting: {}".format(config_name, tag))
    print("-" * 80)

    X_emp, y_emp = load_empkins_full(config_name, config, label_setting)
    print("  Proposed ALL ({}): n={}, dep={}/{}".format(
        tag, len(y_emp), int(y_emp.sum()), int((y_emp == 0).sum())))

    save_dir = OUTPUT_BASE
    summary_rows = []

    # ── E1: All Proposed → All E-DAIC ────────────────────────────────────────
    print("\n  E1: All Proposed → All E-DAIC (n_train={}, n_test={})".format(
        len(y_emp), len(y_daic_all)))

    X_emp_al, X_daic_al = align_features(X_emp, X_daic_all)
    imp = SimpleImputer(strategy="median")
    X_emp_imp  = pd.DataFrame(
        imp.fit_transform(X_emp_al),
        columns=X_emp_al.columns, index=X_emp_al.index).astype(np.float32)
    X_daic_imp = pd.DataFrame(
        imp.transform(X_daic_al),
        columns=X_daic_al.columns, index=X_daic_al.index).astype(np.float32)

    _, best_E1 = fit_and_eval_cls(
        models, param_grids, scalers,
        X_emp_imp, y_emp,
        X_daic_imp, y_daic_all,
        experiment_tag="E1_{}_empALL2daicALL_{}".format(tag, config_name),
        save_dir=save_dir, timestamp=timestamp
    )
    if best_E1:
        summary_rows.append({
            "label_setting": tag,
            "Experiment":    "E1: Proposed-ALL->E-DAIC-ALL",
            "n_train":       len(y_emp),
            "n_test":        len(y_daic_all),
            "Best_model":    "{}/{}".format(best_E1["model_name"], best_E1["scaler"]),
            **best_E1["metrics"],
        })

    # ── E2: All E-DAIC → All Proposed ────────────────────────────────────────
    print("\n  E2: All E-DAIC → All Proposed (n_train={}, n_test={})".format(
        len(y_daic_all), len(y_emp)))

    X_daic_al2, X_emp_al2 = align_features(X_daic_all, X_emp)
    imp2 = SimpleImputer(strategy="median")
    X_daic_imp2 = pd.DataFrame(
        imp2.fit_transform(X_daic_al2),
        columns=X_daic_al2.columns, index=X_daic_al2.index).astype(np.float32)
    X_emp_imp2  = pd.DataFrame(
        imp2.transform(X_emp_al2),
        columns=X_emp_al2.columns, index=X_emp_al2.index).astype(np.float32)

    _, best_E2 = fit_and_eval_cls(
        models, param_grids, scalers,
        X_daic_imp2, y_daic_all,
        X_emp_imp2, y_emp,
        experiment_tag="E2_{}_daicALL2empALL_{}".format(tag, config_name),
        save_dir=save_dir, timestamp=timestamp
    )
    if best_E2:
        summary_rows.append({
            "label_setting": tag,
            "Experiment":    "E2: E-DAIC-ALL->Proposed-ALL",
            "n_train":       len(y_daic_all),
            "n_test":        len(y_emp),
            "Best_model":    "{}/{}".format(best_E2["model_name"], best_E2["scaler"]),
            **best_E2["metrics"],
        })

    return summary_rows


def run_config_full2full(config_name, config, timestamp, scalers, models, param_grids):
    save_dir = OUTPUT_BASE
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  FULL->FULL CLASSIFICATION | Config: {}".format(config_name))
    print("=" * 80)

    # Load E-DAIC once (same for both settings — always PHQ-8>=10)
    print("\n[E-DAIC] Loading all 275 participants ...")
    try:
        X_daic_all, y_daic_all, _ = load_edaic_cls(split_filter=None)
    except FileNotFoundError:
        print("  ERROR: E-DAIC aggregated file not found.")
        return []

    print("  E-DAIC ALL: n={}, dep={}/{}".format(
        len(y_daic_all), int(y_daic_all.sum()), int((y_daic_all == 0).sum())))

    all_rows = []
    for setting in ["scid", "phq8"]:
        rows = run_setting(
            config_name, config, setting,
            X_daic_all, y_daic_all,
            timestamp, scalers, models, param_grids
        )
        all_rows.extend(rows)

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        path = os.path.join(
            save_dir,
            "SUMMARY_full2full_cls_{}_{}.csv".format(config_name, timestamp)
        )
        summary_df.to_csv(path, index=False)
        print("\n  Summary saved: {}".format(path))

    return all_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-to-full cross-corpus classification (SCID + PHQ-8>=10)"
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(SOURCE_CONFIGS.keys()),
        choices=list(SOURCE_CONFIGS.keys()),
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
        rows = run_config_full2full(
            cfg_name, SOURCE_CONFIGS[cfg_name],
            ts, scalers, models, grids
        )
        for row in rows:
            all_summaries.append({"config": cfg_name, **row})

    if all_summaries:
        final_df   = pd.DataFrame(all_summaries)
        final_path = os.path.join(
            OUTPUT_BASE,
            "FINAL_COMPARISON_full2full_cls_{}.csv".format(ts)
        )
        final_df.to_csv(final_path, index=False)
        print("\n" + "=" * 80)
        print("  FINAL FULL->FULL CLASSIFICATION COMPARISON")
        print("=" * 80)
        print(final_df[[
            "config", "label_setting", "Experiment",
            "n_train", "n_test", "Best_model", "AUC", "F1"
        ]].to_string(index=False))
        print("\n  Final comparison saved: {}".format(final_path))

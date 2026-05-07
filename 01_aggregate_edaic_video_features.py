"""
01_aggregate_edaic_video_features.py
=====================================
Aggregates per-participant frame-level E-DAIC OpenFace 2.1.0 CSVs into a
single-row-per-participant feature matrix using the same 18 statistical
functionals as the ProposedCorpus pipeline — producing directly comparable columns.

Run this ONCE before any cross-corpus script.

Input  : <EDAIC_OPENFACE_ROOT from config.py>
           extracted/{ID}_P/features/{ID}_OpenFace2.1.0_Pose_gaze_AUs.csv
Output : <EDAIC_AGGREGATED_PATH from config.py>
           processed_data_stats/edaic_openface_aggregated.csv

Usage:
  python 01_aggregate_edaic_video_features.py
  python 01_aggregate_edaic_video_features.py --min_confidence 0.5 --min_frames 100
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── shared helpers ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from shared_regression_pipeline import (
    EDAIC_LABELS_PATH,
    EDAIC_OPENFACE_ROOT,
    EDAIC_AGGREGATED_PATH,
    META_FRAME_COLS,
    aggregate_openface_frames,
)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(min_confidence: float = 0.5, min_frames: int = 50):
    print("=" * 70)
    print("  E-DAIC OpenFace Aggregation")
    print(f"  min_confidence={min_confidence}  min_frames={min_frames}")
    print("=" * 70)

    # ── Load labels ─────────────────────────────────────────────────────────
    labels_df = pd.read_csv(EDAIC_LABELS_PATH)
    labels_df["participant_id"] = labels_df["participant_id"].astype(str)
    label_map = {
        str(row["participant_id"]): {
            "depressed": int(row["depressed"]),
            "PHQ_score": float(row["PHQ_score"]),
            "split":     str(row["split"]),
            "gender":    str(row.get("gender", "")),
            "age":       row.get("age", np.nan),
        }
        for _, row in labels_df.iterrows()
    }
    print(f"  Labels loaded: {len(label_map)} participants "
          f"({int((labels_df['depressed']==1).sum())} dep / "
          f"{int((labels_df['depressed']==0).sum())} HC)")

    # ── Create output directory ──────────────────────────────────────────────
    out_path = Path(EDAIC_AGGREGATED_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Iterate participants ─────────────────────────────────────────────────
    edaic_root   = Path(EDAIC_OPENFACE_ROOT)
    part_dirs    = sorted(edaic_root.glob("*_P"))
    print(f"  Found {len(part_dirs)} participant directories in:\n  {edaic_root}\n")

    all_rows     = []
    skipped_miss = []
    skipped_conf = []

    for pdir in part_dirs:
        pid = pdir.name.replace("_P", "")
        feat_csv = pdir / "features" / f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"

        if not feat_csv.exists():
            skipped_miss.append(pid)
            continue

        try:
            df = pd.read_csv(feat_csv)
        except Exception as e:
            print(f"  [WARN] {pid}: read error ({e}); skipping.")
            skipped_miss.append(pid)
            continue

        df.columns = df.columns.str.strip()

        # Confidence filter
        if "confidence" in df.columns:
            df_filt = df[df["confidence"] >= min_confidence].copy()
        else:
            df_filt = df.copy()

        if len(df_filt) < min_frames:
            print(f"  [WARN] {pid}: {len(df_filt)} confident frames "
                  f"(< {min_frames}); skipping.")
            skipped_conf.append(pid)
            continue

        # Aggregate
        signal_cols = [c for c in df_filt.columns if c.strip() not in META_FRAME_COLS]
        agg_dict    = aggregate_openface_frames(df_filt[signal_cols])

        # Build row
        row = {"participant_id": pid}
        row.update(agg_dict)

        if pid in label_map:
            row.update(label_map[pid])
        else:
            row.update({"depressed": np.nan, "PHQ_score": np.nan,
                        "split": "unknown", "gender": "", "age": np.nan})

        all_rows.append(row)

    if not all_rows:
        print("\nERROR: no participants processed. Check EDAIC_OPENFACE_ROOT.")
        sys.exit(1)

    # ── Save ─────────────────────────────────────────────────────────────────
    result_df = pd.DataFrame(all_rows)

    # Put metadata columns first for readability
    meta_first = ["participant_id", "depressed", "PHQ_score", "split",
                  "gender", "age"]
    feat_cols  = [c for c in result_df.columns if c not in meta_first]
    result_df  = result_df[meta_first + feat_cols]
    result_df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"  Aggregation complete.")
    print(f"  Output : {out_path}")
    print(f"  Shape  : {result_df.shape}  (participants × features+metadata)")
    print(f"  Processed : {len(all_rows)} participants")
    if skipped_miss:
        print(f"  Skipped (missing/unreadable): {len(skipped_miss)}")
    if skipped_conf:
        print(f"  Skipped (< {min_frames} confident frames): {len(skipped_conf)}")
    dep_vals = result_df["depressed"].dropna()
    print(f"  Label dist: HC={int((dep_vals==0).sum())}, "
          f"dep={int((dep_vals==1).sum())}, "
          f"unlabelled={int(result_df['depressed'].isna().sum())}")
    split_dist = result_df["split"].value_counts().to_dict()
    print(f"  Split dist: {split_dist}")
    phq = result_df["PHQ_score"].dropna()
    print(f"  PHQ-8: mean={phq.mean():.1f}, std={phq.std():.1f}, "
          f"range=[{phq.min():.0f},{phq.max():.0f}]")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate E-DAIC OpenFace frame-level CSVs → per-participant feature matrix"
    )
    parser.add_argument("--min_confidence", type=float, default=0.5,
                        help="Minimum OpenFace confidence to keep a frame (default: 0.5)")
    parser.add_argument("--min_frames", type=int, default=50,
                        help="Minimum high-confidence frames required (default: 50)")
    args = parser.parse_args()
    main(min_confidence=args.min_confidence, min_frames=args.min_frames)

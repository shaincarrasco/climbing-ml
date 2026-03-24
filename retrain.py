"""
retrain.py
----------
Rebuilds all ML models from current pose_frames data in PostgreSQL.

Run this after any significant scraping session to update predictions.

Steps
-----
  1. pose_feature_extraction.py  — aggregate pose_frames → data/pose_features.csv
  2. ml/pose_difficulty_model.py — retrain pose-only grade model
  3. ml/difficulty_model.py      — retrain main hold model with pose features fused in

Usage
-----
    python3 retrain.py              # full retrain
    python3 retrain.py --pose-only  # skip main model (faster, ~30s)
    python3 retrain.py --quick      # pose features + pose model only, skip 57K route retrain
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent


def run(label: str, cmd: list[str]) -> bool:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_BASE_DIR))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit {result.returncode}) — {label}")
        return False
    print(f"\n  ✓ Done in {elapsed:.0f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description="Retrain all ML models from latest pose data")
    ap.add_argument("--pose-only", action="store_true",
                    help="Only retrain the pose model, skip the 57K-route main model")
    ap.add_argument("--quick", action="store_true",
                    help="Pose features + correlations + pose model only (~30s). Skips imputer + main model.")
    args = ap.parse_args()

    print("\n══ Climbing ML — Retrain Pipeline ══════════════════════════")

    # Step 0: Purge low-quality pose frames before aggregation
    # Videos scoring < 35 on data_quality have more noise than signal.
    # Removing their frames prevents them from corrupting pose_features.csv.
    run(
        "Step 0 — Purge low-quality pose frames (score < 35) from DB",
        [sys.executable, "pipeline/data_quality.py", "--purge-bad", "--threshold", "35"],
    )

    # Step 1: always rebuild pose features CSV from DB
    ok = run(
        "Step 1/4 — Aggregate pose_frames → pose_features.csv",
        [sys.executable, "pipeline/pose_feature_extraction.py"],
    )
    if not ok:
        sys.exit(1)

    # Step 1b: save Pearson correlations (non-fatal — DB optional)
    run(
        "Step 1b — Compute pose-metric correlations → data/pose_correlations.json",
        [sys.executable, "pipeline/save_correlations.py"],
    )

    # Step 2: retrain pose-specific model
    ok = run(
        "Step 2/4 — Retrain pose difficulty model",
        [sys.executable, "ml/pose_difficulty_model.py"],
    )
    if not ok:
        sys.exit(1)

    if args.quick:
        print("\n══ Done (--quick: skipped imputer + main model) ════════════")
        return

    # Step 3: train pose imputer (hold geometry → pose for ALL 57K routes)
    ok = run(
        "Step 3/4 — Train pose imputer → data/pose_imputed.csv (full coverage)",
        [sys.executable, "ml/pose_imputer.py"],
    )
    if not ok:
        print("  WARNING: pose imputer failed — main model will use sparse pose data")

    # Step 3b: retrain hold type classifier (enriches geometry features)
    run(
        "Step 3b — Hold type classifier → data/hold_types_predicted.csv",
        [sys.executable, "ml/hold_type_classifier.py"],
    )

    if args.pose_only:
        print("\n══ Done (--pose-only: skipped main model retrain) ══════════")
        return

    # Step 4: retrain main hold model with imputed pose features for all routes
    ok = run(
        "Step 4/4 — Retrain main difficulty model (with full-coverage pose fusion)",
        [sys.executable, "ml/difficulty_model.py"],
    )
    if not ok:
        sys.exit(1)

    print("\n══ All models updated ══════════════════════════════════════")
    print("  Restart the API to load the new models:")
    print("  python3 api/app.py")


if __name__ == "__main__":
    main()

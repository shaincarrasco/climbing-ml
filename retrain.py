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
                    help="Rebuild features + pose model only (fastest, ~30s total)")
    args = ap.parse_args()

    print("\n══ Climbing ML — Retrain Pipeline ══════════════════════════")

    # Step 1: always rebuild pose features CSV from DB
    ok = run(
        "Step 1/4 — Aggregate pose_frames → pose_features.csv",
        [sys.executable, "pipeline/pose_feature_extraction.py"],
    )
    if not ok:
        sys.exit(1)

    # Step 1b: save Pearson correlations (non-fatal — DB optional)
    run(
        "Step 1b/4 — Compute pose-metric correlations → data/pose_correlations.json",
        [sys.executable, "pipeline/save_correlations.py"],
    )

    # Step 2: retrain pose-specific model
    ok = run(
        "Step 2/4 — Retrain pose difficulty model",
        [sys.executable, "ml/pose_difficulty_model.py"],
    )
    if not ok:
        sys.exit(1)

    if args.pose_only or args.quick:
        print("\n══ Done (skipped main model retrain) ══════════════════════")
        return

    # Step 3: retrain main hold model with pose features fused
    ok = run(
        "Step 3/4 — Retrain main difficulty model (with pose fusion)",
        [sys.executable, "ml/difficulty_model.py"],
    )
    if not ok:
        sys.exit(1)

    print("\n══ All models updated ══════════════════════════════════════")
    print("  Restart the API to load the new models:")
    print("  python3 api/app.py")


if __name__ == "__main__":
    main()

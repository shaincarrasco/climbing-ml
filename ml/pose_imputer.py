"""
ml/pose_imputer.py
------------------
Stage 1 of the two-stage model:

  hold geometry (50+ features) → predicted pose metrics (35 pose columns)

Trained on the ~932 routes with real video pose data.
Predicts pose metrics for all 57K routes without video.
Real pose data is kept as-is; predictions fill the gaps.

Output
------
  data/pose_imputed.csv   — all routes_features rows + filled pose columns
  ml/pose_imputer/        — one XGBoost model per pose metric

Why this matters
----------------
With only 1.6% pose coverage, XGBoost ignores pose features (98.4% NaN).
After imputation every route has a pose profile, so the main model can
learn that "a route with crimps, 40cm average reach, 45° angle probably
requires X hip angle and Y tension." Every new scraped video improves
the imputer's predictions for *all* 57K routes — not just the one route.

Usage
-----
    python3 ml/pose_imputer.py              # train + predict all routes
    python3 ml/pose_imputer.py --eval-only  # cross-val metrics only, no output
    python3 ml/pose_imputer.py --predict-only  # skip training, use saved model
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

_BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR      = _BASE_DIR / "data"
MODEL_DIR     = _BASE_DIR / "ml" / "pose_imputer"
ROUTES_CSV    = DATA_DIR  / "routes_features.csv"
POSE_CSV      = DATA_DIR  / "pose_features.csv"
OUTPUT_CSV    = DATA_DIR  / "pose_imputed.csv"
EVAL_JSON     = _BASE_DIR / "ml" / "pose_imputer_eval.json"
MIN_FRAMES    = 40   # routes with fewer frames had poor pose detection — exclude from training

# ── Hold geometry columns used as predictors ──────────────────────────────────
# These come from routes_features.csv and describe the route's physical layout.
GEOMETRY_FEATURES = [
    "board_angle_deg",
    "hand_hold_count", "foot_hold_count", "total_hold_count",
    "foot_ratio", "hand_foot_ratio",
    "avg_reach_cm", "max_reach_cm", "min_reach_cm",
    "reach_std_cm", "reach_range_cm", "reach_cv",
    "avg_lateral_cm", "avg_vertical_cm",
    "max_lateral_cm", "max_vertical_cm",
    "vertical_ratio", "lateral_ratio",
    "avg_foot_spread_cm", "avg_hand_to_foot_cm",
    "avg_foot_hand_x_offset_cm", "max_foot_hand_x_offset_cm",
    "height_span_cm", "lateral_span_cm", "width_height_ratio", "hold_density",
    "direction_changes", "zigzag_ratio",
    "dyno_score", "dyno_flag", "max_reach_z",
    "start_height_pct", "finish_height_pct",
    "hold_spread_x", "hold_spread_y",
    "crimp_ratio", "sloper_ratio", "jug_ratio", "pinch_ratio",
    "difficulty_score",   # target-adjacent signal — helps a lot
]

# Pose columns to impute — all columns in pose_features.csv starting with "pose_"
# plus climb_uuid excluded (it's the join key)
POSE_EXCLUDE = {
    "climb_uuid", "video_count", "frame_count",
    "route_name", "setter", "difficulty_average",
    "quality_average", "ascensionist_count",
}

XGB_PARAMS = {
    "n_estimators":     400,
    "learning_rate":    0.05,
    "max_depth":        5,
    "min_child_weight": 3,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "objective":        "reg:squarederror",
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_training_data():
    """
    Join routes_features.csv with pose_features.csv via PostgreSQL route_id mapping.
    Returns (X, Y, pose_cols, routes_df) where X is geometry features for
    pose-covered routes, Y is their real pose metrics.
    """
    if not ROUTES_CSV.exists():
        sys.exit(f"  ERROR: {ROUTES_CSV} not found — run pipeline/feature_extraction.py first")
    if not POSE_CSV.exists():
        sys.exit(f"  ERROR: {POSE_CSV} not found — run pipeline/pose_feature_extraction.py first")

    routes = pd.read_csv(ROUTES_CSV)
    pose   = pd.read_csv(POSE_CSV)

    # Quality filter: routes need enough frames to have reliable pose aggregates
    if "frame_count" in pose.columns:
        before_q = len(pose)
        pose = pose[pose["frame_count"] >= MIN_FRAMES].copy()
        print(f"  Quality filter (frame_count >= {MIN_FRAMES}): {len(pose):,} / {before_q:,} routes kept")

    pose_cols = [c for c in pose.columns if c not in POSE_EXCLUDE]
    print(f"  Pose targets: {len(pose_cols)} columns")

    # Map climb_uuid → route_id via PostgreSQL
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv(_BASE_DIR / ".env")
        conn = psycopg2.connect(os.getenv("DATABASE_URL",
                                           "postgresql://localhost/climbing_platform"))
        mapping = pd.read_sql(
            "SELECT id AS route_id, external_id FROM board_routes WHERE external_id IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception as e:
        sys.exit(f"  ERROR: DB connection failed ({e})")

    mapping["ext_upper"] = mapping["external_id"].str.upper()
    pose["ext_upper"]    = pose["climb_uuid"].str.upper()

    # pose → route_id
    pose_mapped = pose.merge(mapping[["route_id", "ext_upper"]], on="ext_upper", how="inner")
    pose_mapped = pose_mapped[["route_id"] + pose_cols].drop_duplicates("route_id")

    # routes + pose (inner join for training set)
    merged = routes.merge(pose_mapped, on="route_id", how="inner")
    print(f"  Training rows (routes with real pose data): {len(merged):,}")

    # Geometry features — only use ones that exist in the CSV
    geo_cols = [c for c in GEOMETRY_FEATURES if c in merged.columns]
    X = merged[geo_cols].copy()
    Y = merged[pose_cols].copy()

    return X, Y, pose_cols, geo_cols, routes, pose_mapped


# ── Training ──────────────────────────────────────────────────────────────────

def train(eval_only: bool = False):
    print("\n  Loading data …")
    X, Y, pose_cols, geo_cols, routes, pose_mapped = _load_training_data()

    # Drop pose columns that are entirely NaN in training set
    valid_pose_cols = [c for c in pose_cols if Y[c].notna().sum() >= 20]
    dropped = len(pose_cols) - len(valid_pose_cols)
    if dropped:
        print(f"  Dropped {dropped} pose cols with <20 real values")
    pose_cols = valid_pose_cols
    Y = Y[pose_cols]

    # Fill geometry NaN with column medians (rare edge cases)
    X = X.fillna(X.median(numeric_only=True))

    print(f"  Geometry features: {len(geo_cols)}")
    print(f"  Pose targets:      {len(pose_cols)}")

    # ── 5-fold cross-validation ───────────────────────────────────────────────
    print("\n  5-fold cross-validation …")
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = {col: [] for col in pose_cols}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        for col in pose_cols:
            y_tr  = Y[col].iloc[tr_idx]
            y_val = Y[col].iloc[val_idx]
            mask_tr  = y_tr.notna()
            mask_val = y_val.notna()
            if mask_tr.sum() < 10 or mask_val.sum() < 3:
                continue
            m = xgb.XGBRegressor(**XGB_PARAMS, verbosity=0)
            m.fit(X_tr[mask_tr], y_tr[mask_tr])
            preds = m.predict(X_val[mask_val])
            cv_mae[col].append(mean_absolute_error(y_val[mask_val], preds))

    # Summary
    eval_summary = []
    for col in pose_cols:
        maes = cv_mae[col]
        if maes:
            eval_summary.append({
                "metric": col,
                "cv_mae": round(float(np.mean(maes)), 4),
                "cv_mae_std": round(float(np.std(maes)), 4),
            })
    eval_summary.sort(key=lambda x: x["cv_mae"])

    print(f"\n  {'Pose metric':<35}  {'CV MAE':>8}  {'±':>6}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*6}")
    for e in eval_summary[:15]:
        print(f"  {e['metric']:<35}  {e['cv_mae']:>8.4f}  ±{e['cv_mae_std']:.4f}")
    if len(eval_summary) > 15:
        print(f"  … and {len(eval_summary)-15} more")

    with open(EVAL_JSON, "w") as f:
        json.dump({"cv_results": eval_summary, "n_train": len(X),
                   "n_pose_cols": len(pose_cols), "geo_cols": geo_cols}, f, indent=2)
    print(f"\n  Eval → {EVAL_JSON.relative_to(_BASE_DIR)}")

    if eval_only:
        return

    # ── Train final models on all data ───────────────────────────────────────
    print("\n  Training final models on full dataset …")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_full = X.fillna(X.median(numeric_only=True))
    models = {}
    for col in pose_cols:
        mask = Y[col].notna()
        if mask.sum() < 10:
            continue
        m = xgb.XGBRegressor(**XGB_PARAMS, verbosity=0)
        m.fit(X_full[mask], Y[col][mask])
        m.save_model(str(MODEL_DIR / f"{col}.ubj"))
        models[col] = m

    print(f"  Saved {len(models)} pose imputer models → {MODEL_DIR.relative_to(_BASE_DIR)}/")

    # Save metadata (geo_cols list needed at predict time)
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump({"geo_cols": geo_cols, "pose_cols": list(models.keys())}, f, indent=2)

    # ── Predict for all routes ────────────────────────────────────────────────
    _predict_all(routes, pose_mapped, models, geo_cols)


def _predict_all(routes: pd.DataFrame, pose_mapped: pd.DataFrame,
                 models: dict, geo_cols: list):
    """
    Run imputer models on all routes_features rows.
    Routes with real pose data keep real values.
    Routes without get model predictions.
    Saves data/pose_imputed.csv.
    """
    print("\n  Predicting pose metrics for all routes …")

    X_all = routes[geo_cols].copy().fillna(routes[geo_cols].median(numeric_only=True))
    X_all = X_all.fillna(0)

    predictions = {"route_id": routes["route_id"].values}
    for col, model in models.items():
        predictions[col] = model.predict(X_all.values)

    pred_df = pd.DataFrame(predictions)

    # Merge real pose data — real values take priority over predictions
    real_cols = list(models.keys())
    merged = routes[["route_id"]].merge(
        pose_mapped[["route_id"] + [c for c in real_cols if c in pose_mapped.columns]],
        on="route_id", how="left",
    )

    # For each pose column: use real value if available, else use prediction
    result = routes.copy()
    for col in real_cols:
        pred_col = pred_df.set_index("route_id")[col]
        if col in merged.columns:
            real_col = merged.set_index("route_id")[col]
            # Align on route_id
            result[col] = result["route_id"].map(real_col)
            # Fill NaN with predictions
            missing_mask = result[col].isna()
            result.loc[missing_mask, col] = result.loc[missing_mask, "route_id"].map(pred_col)
        else:
            result[col] = result["route_id"].map(pred_col)

    n_real = pose_mapped["route_id"].isin(routes["route_id"]).sum()
    n_pred = len(routes) - n_real
    print(f"  Real pose data:   {n_real:,} routes")
    print(f"  Imputed (model):  {n_pred:,} routes")
    print(f"  Total:            {len(result):,} routes")

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved → {OUTPUT_CSV.relative_to(_BASE_DIR)}")


def predict_only():
    """Load saved models and re-run imputation without retraining."""
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        sys.exit("  ERROR: No saved imputer models — run without --predict-only first")

    with open(meta_path) as f:
        meta = json.load(f)

    geo_cols  = meta["geo_cols"]
    pose_cols = meta["pose_cols"]

    models = {}
    for col in pose_cols:
        p = MODEL_DIR / f"{col}.ubj"
        if p.exists():
            m = xgb.XGBRegressor()
            m.load_model(str(p))
            models[col] = m

    print(f"  Loaded {len(models)} imputer models")

    routes, pose_mapped = _load_routes_and_pose(geo_cols, pose_cols)
    _predict_all(routes, pose_mapped, models, geo_cols)


def _load_routes_and_pose(geo_cols, pose_cols):
    routes = pd.read_csv(ROUTES_CSV)
    pose   = pd.read_csv(POSE_CSV)

    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv(_BASE_DIR / ".env")
        conn = psycopg2.connect(os.getenv("DATABASE_URL",
                                           "postgresql://localhost/climbing_platform"))
        mapping = pd.read_sql(
            "SELECT id AS route_id, external_id FROM board_routes WHERE external_id IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception as e:
        sys.exit(f"  ERROR: DB connection failed ({e})")

    mapping["ext_upper"] = mapping["external_id"].str.upper()
    pose["ext_upper"]    = pose["climb_uuid"].str.upper()
    valid_pose_cols = [c for c in pose_cols if c in pose.columns]
    pose_mapped = pose.merge(mapping[["route_id", "ext_upper"]], on="ext_upper", how="inner")
    pose_mapped = pose_mapped[["route_id"] + valid_pose_cols].drop_duplicates("route_id")
    return routes, pose_mapped


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train pose imputer: hold geometry → pose metrics")
    ap.add_argument("--eval-only",     action="store_true",
                    help="Cross-val only — no model saved, no output CSV")
    ap.add_argument("--predict-only",  action="store_true",
                    help="Skip training, use saved models to regenerate pose_imputed.csv")
    args = ap.parse_args()

    print("\n══ Pose Imputer ════════════════════════════════════════════")
    if args.predict_only:
        predict_only()
    else:
        train(eval_only=args.eval_only)
    print("══ Done ════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()

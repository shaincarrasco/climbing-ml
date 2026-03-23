"""
ml/pose_difficulty_model.py
----------------------------
Trains an XGBoost difficulty model using pose features extracted from
beta videos. Designed to work with small N (13+ routes) via leave-one-out
cross-validation, and scale up automatically as more videos are scraped.

Usage:
    python ml/pose_difficulty_model.py
    python ml/pose_difficulty_model.py --input data/pose_features.csv
    python ml/pose_difficulty_model.py --stats
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

_BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR     = _BASE_DIR / "data"
MODEL_DIR    = Path(__file__).resolve().parent
INPUT_CSV    = DATA_DIR / "pose_features.csv"
OUTPUT_JSON  = MODEL_DIR / "pose_model_eval.json"
MODEL_PKL    = MODEL_DIR / "pose_difficulty_model.pkl"

# Kilter raw difficulty → approximate V-grade
# difficulty_average from climb_stats is on a ~0–33 scale
KILTER_TO_VGRADE = {
    (0,  14): "V0",
    (14, 17): "V1",
    (17, 19): "V2",
    (19, 20): "V3",
    (20, 21): "V4",
    (21, 22): "V5",
    (22, 24): "V6",
    (24, 26): "V7",
    (26, 27): "V8",
    (27, 28): "V9",
    (28, 29): "V10",
    (29, 30): "V11",
    (30, 31): "V12",
    (31, 33): "V13+",
}

def kilter_to_vgrade(score):
    if pd.isna(score):
        return None
    for (lo, hi), grade in KILTER_TO_VGRADE.items():
        if lo <= score < hi:
            return grade
    return "V13+"


# Features that showed meaningful correlation in initial analysis
# Ordered by absolute correlation strength
TOP_FEATURES = [
    "pose_avg_shoulder_l",
    "pose_min_elbow_r",
    "pose_peak_com_vel",
    "pose_max_hip_spread",
    "pose_avg_com_to_hands",
    "pose_avg_tension",
    "pose_p90_tension",
    "pose_pct_straight_l",
    "pose_avg_elbow_l",
    "pose_avg_elbow_r",
    "pose_p10_hip_angle",
    "pose_avg_hip_angle",
    "pose_hip_angle_range",
]


def load_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df.dropna(subset=["difficulty_average"])
    df["vgrade"] = df["difficulty_average"].apply(kilter_to_vgrade)
    print(f"  Loaded {len(df)} rated routes")
    print(f"  Difficulty range: {df['difficulty_average'].min():.1f} – {df['difficulty_average'].max():.1f}")
    if "route_name" in df.columns:
        for _, row in df.sort_values("difficulty_average").iterrows():
            print(f"    {row['difficulty_average']:>5.1f}  {row.get('vgrade','?'):<5}  {row.get('route_name','?')}")
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    """Use TOP_FEATURES that exist and have enough non-null values."""
    available = []
    for f in TOP_FEATURES:
        if f in df.columns and df[f].notna().sum() >= max(5, len(df) * 0.6):
            available.append(f)
    return available


def train(input_path: str = str(INPUT_CSV)):
    print(f"\nLoading pose features from {input_path}...")
    df = load_data(input_path)

    feature_cols = select_features(df)
    print(f"\nFeatures selected: {len(feature_cols)}")
    for f in feature_cols:
        print(f"  {f}")

    # Fill remaining NULLs with column median
    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = X[col].fillna(X[col].median())
    X = X.values
    y = df["difficulty_average"].values

    n = len(df)
    print(f"\nTraining on {n} routes...")

    if n < 20:
        # Leave-One-Out CV — reliable with small N
        print("  Using Leave-One-Out CV (n < 20)")

        # Ridge regression — stable with small N, won't overfit
        ridge_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ])
        loo = LeaveOneOut()
        y_pred_ridge = cross_val_predict(ridge_pipe, X, y, cv=loo)
        mae_ridge = mean_absolute_error(y, y_pred_ridge)

        # XGBoost with heavy regularization
        xgb_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  xgb.XGBRegressor(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                reg_alpha=2.0,
                reg_lambda=5.0,
                subsample=0.8,
                random_state=42,
                verbosity=0,
            )),
        ])
        y_pred_xgb = cross_val_predict(xgb_pipe, X, y, cv=loo)
        mae_xgb = mean_absolute_error(y, y_pred_xgb)

        # Use whichever performs better
        if mae_ridge <= mae_xgb:
            best_name  = "Ridge"
            y_pred     = y_pred_ridge
            mae        = mae_ridge
        else:
            best_name  = "XGBoost"
            y_pred     = y_pred_xgb
            mae        = mae_xgb

        print(f"\n  Ridge  MAE: {mae_ridge:.2f}  ({mae_ridge/33*15:.1f} V-grades avg error)")
        print(f"  XGBoost MAE: {mae_xgb:.2f}  ({mae_xgb/33*15:.1f} V-grades avg error)")
        print(f"  → Best: {best_name}")

    else:
        # Enough data — 5-fold CV for per-route predictions + honest MAE
        from sklearn.model_selection import KFold
        print(f"  Using 5-fold CV (n >= 20)")
        xgb_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=2.0, subsample=0.8,
                random_state=42, verbosity=0,
            )),
        ])
        ridge_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_xgb   = cross_val_predict(xgb_pipe,   X, y, cv=kf)
        y_pred_ridge = cross_val_predict(ridge_pipe,  X, y, cv=kf)
        mae_xgb   = mean_absolute_error(y, y_pred_xgb)
        mae_ridge = mean_absolute_error(y, y_pred_ridge)
        if mae_ridge <= mae_xgb:
            best_name = "Ridge"
            y_pred    = y_pred_ridge
            mae       = mae_ridge
        else:
            best_name = "XGBoost"
            y_pred    = y_pred_xgb
            mae       = mae_xgb
        print(f"\n  Ridge   MAE: {mae_ridge:.2f}  ({mae_ridge/33*15:.1f} V-grades avg error)")
        print(f"  XGBoost MAE: {mae_xgb:.2f}  ({mae_xgb/33*15:.1f} V-grades avg error)")
        print(f"  → Best: {best_name}")
        print(f"  MAE: {mae:.2f}  ({mae/33*15:.1f} V-grades avg error)")

    # Per-route prediction table
    print(f"\n{'Route':<25} {'Actual':>8} {'Predicted':>10} {'Error':>7} {'Actual':>8} {'→ Pred'}")
    print("─" * 72)
    names = df.get("route_name", pd.Series(["?"] * n))
    for i in range(n):
        actual    = y[i]
        predicted = y_pred[i]
        err       = predicted - actual
        vg_actual = kilter_to_vgrade(actual)
        vg_pred   = kilter_to_vgrade(predicted)
        name      = str(names.iloc[i])[:24]
        match     = "✓" if vg_actual == vg_pred else "✗"
        print(f"  {name:<24} {actual:>7.1f}  {predicted:>9.1f}  {err:>+6.1f}  {vg_actual:<8} → {vg_pred}  {match}")

    # Save evaluation
    eval_data = {
        "n_routes":       n,
        "n_features":     len(feature_cols),
        "feature_cols":   feature_cols,
        "best_model":     best_name,
        "mae_difficulty": round(float(mae), 3),
        "mae_vgrades":    round(float(mae / 33 * 15), 2),
        "cv_method":      "leave-one-out" if n < 20 else "5-fold CV",
        "note":           f"Trained on {n} routes. Accuracy will improve with more scraped beta videos.",
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"\nEvaluation → {OUTPUT_JSON}")

    # Train final model on ALL data and save it
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ])
    final_pipe.fit(X, y)
    col_medians = {col: float(df[col].median()) for col in feature_cols}
    saved = {
        "pipeline":     final_pipe,
        "feature_cols": feature_cols,
        "col_medians":  col_medians,
    }
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(saved, f)
    print(f"Model saved → {MODEL_PKL}")

    # Correlation summary for context
    pose_cols = [c for c in df.columns if c.startswith("pose_")]
    corr = df[pose_cols + ["difficulty_average"]].corr()["difficulty_average"].drop("difficulty_average")
    top_corr = corr.abs().sort_values(ascending=False).head(5)
    print(f"\nTop 5 pose signals for difficulty:")
    for feat, val in top_corr.items():
        direction = "harder → more" if corr[feat] > 0 else "harder → less"
        print(f"  {feat:<35} r={corr[feat]:+.3f}  ({direction})")

    print(f"\nNote: {n} routes is a small sample. Run more beta scraping to improve accuracy.")
    print(f"      Target: 200+ routes for a reliable pose-based difficulty model.")

    return eval_data


def load_model():
    """Load the saved pose model. Returns (pipeline, feature_cols, col_medians)."""
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Pose model not found. Run: python ml/pose_difficulty_model.py")
    with open(MODEL_PKL, "rb") as f:
        saved = pickle.load(f)
    return saved["pipeline"], saved["feature_cols"], saved["col_medians"]


def predict_route(pose_features: dict) -> dict:
    """
    Given a dict of pose aggregate features for one route,
    return predicted difficulty_average and V-grade.
    Accepts partial feature dicts — missing values filled with training medians.
    """
    pipe, feature_cols, col_medians = load_model()
    row = [pose_features.get(col, col_medians.get(col, 0.0)) for col in feature_cols]
    score = float(pipe.predict([row])[0])
    score = max(14.0, min(33.0, score))
    return {
        "predicted_difficulty": round(score, 2),
        "predicted_vgrade":     kilter_to_vgrade(score),
        "features_used":        len([v for v in row if v is not None]),
    }


def main():
    parser = argparse.ArgumentParser(description="Train pose-based difficulty model")
    parser.add_argument("--input",  default=str(INPUT_CSV))
    parser.add_argument("--stats",  action="store_true", help="Show data summary only")
    args = parser.parse_args()

    if args.stats:
        df = load_data(args.input)
        pose_cols = [c for c in df.columns if c.startswith("pose_")]
        print(f"\nPose feature coverage ({len(pose_cols)} features):")
        for col in pose_cols:
            pct = df[col].notna().sum() / len(df) * 100
            print(f"  {col:<35} {pct:.0f}% non-null")
        return

    train(args.input)


if __name__ == "__main__":
    main()

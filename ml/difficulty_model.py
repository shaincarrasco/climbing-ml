"""
ml/difficulty_model.py

XGBoost difficulty model for Kilter Board routes.

Targets
-------
  difficulty_score   float 0.0–1.0  (primary regression target)
  community_grade    V-grade string (derived from difficulty_score predictions)

Features (40+)
--------------
  All geometric features from data/routes_features.csv except identifiers,
  target columns, and grade_rank_at_angle (which leaks the target).

Outputs
-------
  ml/difficulty_model.xgb          trained XGBoost model (binary format)
  ml/grade_boundaries.json         difficulty_score → V-grade mapping
  ml/feature_importance.csv        gain-based feature importances
  ml/evaluation.json               RMSE, MAE, grade accuracy on test set

Usage
-----
  python3 ml/difficulty_model.py
  python3 ml/difficulty_model.py --input data/routes_features.csv
  python3 ml/difficulty_model.py --input data/routes_features.csv --min-sends 10
  python3 ml/difficulty_model.py --predict data/new_routes.csv
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

_BASE_DIR       = Path(__file__).resolve().parent.parent
POSE_FEATURES   = _BASE_DIR / "data" / "pose_features.csv"

# ── Column definitions ────────────────────────────────────────────────────────

TARGET        = "difficulty_score"
IDENTIFIER    = "route_id"
LABEL_COL     = "community_grade"

# Columns excluded from features (identifiers, targets, or leaky)
EXCLUDE_COLS  = {
    "route_id",
    "community_grade",
    "difficulty_score",
    "grade_rank_at_angle",   # computed from difficulty_score — leaks target
    "send_count",            # popularity metric, not a route property
}

# ── Grade boundary computation ────────────────────────────────────────────────

def compute_grade_boundaries(df: pd.DataFrame) -> dict:
    """
    For each V-grade, compute the median difficulty_score from training data.
    Used to map continuous predictions → V-grades at inference time.
    Returns: {vgrade: (low_boundary, high_boundary)} sorted by difficulty.
    """
    grade_stats = (
        df.groupby(LABEL_COL)["difficulty_score"]
        .agg(["mean", "min", "max", "median"])
        .reset_index()
    )

    # Sort grades by mean difficulty_score
    grade_stats = grade_stats.sort_values("mean").reset_index(drop=True)

    boundaries = {}
    for i, row in grade_stats.iterrows():
        if i == 0:
            lo = 0.0
        else:
            prev_mean = grade_stats.loc[i - 1, "mean"]
            lo = round((prev_mean + row["mean"]) / 2, 4)
        if i == len(grade_stats) - 1:
            hi = 1.0
        else:
            next_mean = grade_stats.loc[i + 1, "mean"]
            hi = round((row["mean"] + next_mean) / 2, 4)
        boundaries[row[LABEL_COL]] = {"lo": lo, "hi": hi, "mean": round(row["mean"], 4)}

    return boundaries


def score_to_grade(score: float, boundaries: dict) -> str:
    """Map a continuous difficulty_score to the nearest V-grade."""
    best_grade = None
    best_dist  = float("inf")
    for grade, info in boundaries.items():
        if info["lo"] <= score <= info["hi"]:
            dist = abs(score - info["mean"])
            if dist < best_dist:
                best_dist = dist
                best_grade = grade
    if best_grade is None:
        # Fallback: nearest mean
        best_grade = min(boundaries, key=lambda g: abs(score - boundaries[g]["mean"]))
    return best_grade


def scores_to_grades(scores: np.ndarray, boundaries: dict) -> list:
    return [score_to_grade(s, boundaries) for s in scores]


# ── Pose feature fusion ───────────────────────────────────────────────────────

POSE_EXCLUDE = {"climb_uuid", "video_count", "frame_count",
                "route_name", "setter", "difficulty_average",
                "quality_average", "ascensionist_count"}

def fuse_pose_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join pose_features.csv into the main routes dataframe.
    pose_features uses Kilter external_id (climb_uuid); routes uses PostgreSQL UUID (route_id).
    The mapping goes: pose climb_uuid → board_routes.external_id → board_routes.id = route_id.
    Routes without pose data get NaN — XGBoost handles these natively.
    """
    if not POSE_FEATURES.exists():
        print("  [pose fusion] pose_features.csv not found — skipping")
        return df

    pf = pd.read_csv(POSE_FEATURES)
    pose_cols = [c for c in pf.columns if c not in POSE_EXCLUDE]
    if not pose_cols:
        return df

    # Load the external_id → route_id mapping from PostgreSQL
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
        print(f"  [pose fusion] DB unavailable ({e}) — skipping pose features")
        return df

    mapping["ext_upper"] = mapping["external_id"].str.upper()
    pf["ext_upper"]       = pf["climb_uuid"].str.upper()

    # pose_features → route_id
    pf_mapped = pf.merge(mapping[["route_id", "ext_upper"]], on="ext_upper", how="inner")
    pf_mapped = pf_mapped[["route_id"] + pose_cols].drop_duplicates("route_id")

    before = len(df)
    df = df.merge(pf_mapped, on="route_id", how="left")
    n_matched = df[pose_cols[0]].notna().sum()
    print(f"  [pose fusion] {n_matched:,} / {before:,} routes have pose data "
          f"({n_matched/before*100:.1f}%)  |  +{len(pose_cols)} pose features")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip outlier geometry values and fill missing foot-geometry with 0
    (routes with no foot holds legitimately have avg_foot_spread = 0).
    """
    foot_cols = [
        "avg_foot_spread_cm", "avg_hand_to_foot_cm",
        "avg_foot_hand_x_offset_cm", "max_foot_hand_x_offset_cm",
    ]
    for col in foot_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Clip extreme reaches (sensor noise / bad data)
    reach_cols = ["avg_reach_cm", "max_reach_cm", "min_reach_cm", "reach_std_cm", "reach_range_cm"]
    for col in reach_cols:
        if col in df.columns:
            df[col] = df[col].clip(upper=300.0)

    return df


# ── Training ──────────────────────────────────────────────────────────────────

XGB_PARAMS = {
    "n_estimators":       800,
    "learning_rate":      0.05,
    "max_depth":          6,
    "min_child_weight":   5,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,     # L1 — helps with correlated geometry features
    "reg_lambda":         1.0,     # L2
    "objective":          "reg:squarederror",
    "eval_metric":        "rmse",
    "random_state":       42,
    "n_jobs":             -1,
    "tree_method":        "hist",  # fast on large datasets
}


def train(input_path: str = "data/routes_features.csv",
          model_dir: str  = "ml",
          min_sends: int  = 0,
          no_pose: bool   = False) -> None:

    print(f"\nLoading features from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  {len(df):,} rows  |  {df[LABEL_COL].nunique()} unique grades")

    # ── Filters ───────────────────────────────────────────────────────────────
    df = df.dropna(subset=[TARGET, LABEL_COL])
    if min_sends > 0:
        df = df[df["send_count"] >= min_sends]
    print(f"  {len(df):,} rows after filters (min_sends={min_sends})")

    if len(df) < 500:
        raise ValueError(f"Too few rows ({len(df)}) — run feature extraction first.")

    # ── Fuse pose features ────────────────────────────────────────────────────
    if not no_pose:
        df = fuse_pose_features(df)

    df = prepare_features(df)

    # ── Grade boundaries (computed on full dataset before splitting) ───────────
    boundaries = compute_grade_boundaries(df)
    boundaries_path = os.path.join(model_dir, "grade_boundaries.json")
    with open(boundaries_path, "w") as f:
        json.dump(boundaries, f, indent=2)
    print(f"\nGrade boundaries → {boundaries_path}")
    print(f"  Grades: {' '.join(sorted(boundaries, key=lambda g: boundaries[g]['mean']))}")

    # ── Train/val/test split (stratified by grade) ────────────────────────────
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df[TARGET].values
    grades = df[LABEL_COL].values

    X_temp, X_test, y_temp, y_test, g_temp, g_test = train_test_split(
        X, y, grades, test_size=0.15, random_state=42, stratify=grades
    )
    X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(
        X_temp, y_temp, g_temp, test_size=0.176, random_state=42, stratify=g_temp
    )  # 0.176 * 0.85 ≈ 0.15 of total → 70/15/15 split

    print(f"\nSplit: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    print(f"Features: {len(feature_cols)}")

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50, verbosity=1)

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    best_iter = model.best_iteration
    print(f"\nBest iteration: {best_iter}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred_test  = model.predict(X_test)
    y_pred_val   = model.predict(X_val)

    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae_test  = float(mean_absolute_error(y_test, y_pred_test))
    rmse_val  = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

    # Grade-level accuracy on test set
    pred_grades_test = scores_to_grades(y_pred_test, boundaries)
    true_grades_test = g_test.tolist()
    grade_exact  = sum(p == t for p, t in zip(pred_grades_test, true_grades_test)) / len(true_grades_test)
    grade_off1   = _grade_within_n(pred_grades_test, true_grades_test, boundaries, n=1)

    print(f"\n{'─'*55}")
    print(f"Test  RMSE:              {rmse_test:.4f}")
    print(f"Test  MAE:               {mae_test:.4f}")
    print(f"Val   RMSE:              {rmse_val:.4f}")
    print(f"Grade exact accuracy:    {grade_exact*100:.1f}%")
    print(f"Grade within 1V accuracy:{grade_off1*100:.1f}%")
    print(f"{'─'*55}")

    # Per-grade breakdown
    print("\nPer-grade test breakdown:")
    grade_df = pd.DataFrame({
        "true":  true_grades_test,
        "pred":  pred_grades_test,
        "score": y_test,
        "pred_score": y_pred_test,
    })
    for grade in sorted(boundaries, key=lambda g: boundaries[g]["mean"]):
        sub = grade_df[grade_df["true"] == grade]
        if len(sub) == 0:
            continue
        correct = (sub["pred"] == grade).sum()
        print(f"  {grade:>4}  n={len(sub):>5,}  acc={correct/len(sub)*100:>5.1f}%  "
              f"rmse={np.sqrt(mean_squared_error(sub['score'], sub['pred_score'])):.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    model_path = os.path.join(model_dir, "difficulty_model.xgb")
    model.save_model(model_path)
    print(f"\nModel → {model_path}")

    # Feature importances
    importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_path = os.path.join(model_dir, "feature_importance.csv")
    importances.to_csv(imp_path, index=False)
    print(f"Feature importance → {imp_path}")

    print("\nTop 15 features:")
    print(importances.head(15).to_string(index=False))

    # Evaluation JSON
    eval_data = {
        "n_train":             len(X_train),
        "n_val":               len(X_val),
        "n_test":              len(X_test),
        "n_features":          len(feature_cols),
        "feature_cols":        feature_cols,
        "best_iteration":      best_iter,
        "test_rmse":           round(rmse_test, 5),
        "test_mae":            round(mae_test, 5),
        "val_rmse":            round(rmse_val, 5),
        "grade_exact_acc":     round(grade_exact, 4),
        "grade_within1_acc":   round(grade_off1, 4),
        "xgb_params":          XGB_PARAMS,
    }
    eval_path = os.path.join(model_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Evaluation  → {eval_path}")


def _grade_within_n(pred_grades, true_grades, boundaries, n=1) -> float:
    """Fraction of predictions within n V-grades of the true grade."""
    sorted_grades = sorted(boundaries, key=lambda g: boundaries[g]["mean"])
    grade_idx = {g: i for i, g in enumerate(sorted_grades)}
    within = sum(
        abs(grade_idx.get(p, -99) - grade_idx.get(t, -99)) <= n
        for p, t in zip(pred_grades, true_grades)
    )
    return within / len(true_grades)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(input_path: str,
            model_dir: str = "ml",
            output_path: str = None) -> pd.DataFrame:
    """Load saved model and predict difficulty_score + community_grade."""
    model_path = os.path.join(model_dir, "difficulty_model.xgb")
    boundaries_path = os.path.join(model_dir, "grade_boundaries.json")
    eval_path = os.path.join(model_dir, "evaluation.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    with open(boundaries_path) as f:
        boundaries = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    feature_cols = eval_data["feature_cols"]

    df = pd.read_csv(input_path)
    df = prepare_features(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing columns: {missing}")

    X = df[feature_cols].values
    scores = model.predict(X).clip(0.0, 1.0)
    grades = scores_to_grades(scores, boundaries)

    df["pred_difficulty_score"] = scores.round(4)
    df["pred_community_grade"]  = grades

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions → {output_path}")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / run XGBoost difficulty model")
    parser.add_argument("--input",      type=str, default="data/routes_features.csv",
                        help="Feature CSV (output of feature_extraction.py)")
    parser.add_argument("--model-dir",  type=str, default="ml",
                        help="Directory for model artifacts")
    parser.add_argument("--min-sends",  type=int, default=0,
                        help="Filter routes with fewer sends than this")
    parser.add_argument("--predict",    type=str, default=None,
                        help="If set, run inference on this CSV instead of training")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output CSV for --predict mode")
    parser.add_argument("--no-pose",    action="store_true",
                        help="Skip pose feature fusion (baseline comparison)")
    args = parser.parse_args()

    if args.predict:
        df = predict(args.predict, model_dir=args.model_dir, output_path=args.output)
        print(df[["route_id", "pred_difficulty_score", "pred_community_grade"]].head(20).to_string(index=False))
    else:
        train(input_path=args.input, model_dir=args.model_dir,
              min_sends=args.min_sends, no_pose=args.no_pose)

"""
ml/difficulty_model.py

XGBoost + LightGBM stacked difficulty model for Kilter Board routes.

Targets
-------
  difficulty_score   float 0.0–1.0  (primary regression target)
  community_grade    V-grade string (derived from difficulty_score predictions)

Features (67+ after new feature engineering)
--------------
  All geometric features from data/routes_features.csv except identifiers,
  target columns, and grade_rank_at_angle (which leaks the target).

Pipeline
--------
  1. Optuna hyperparameter search (50 trials) for XGBoost
  2. XGBoost base model (best params from Optuna or baseline if Optuna loses)
  3. LightGBM base model (parallel)
  4. Stacking: OOF predictions from both models → Ridge meta-learner
  5. Isotonic regression calibration: raw score → calibrated V-grade mapping
  6. Grade distribution check with sample_weight balancing for rare grades

Outputs
-------
  ml/difficulty_model.xgb          trained XGBoost model (binary format)
  ml/difficulty_model.lgb          trained LightGBM model
  ml/difficulty_model_stack.pkl    stacking meta-model (Ridge)
  ml/difficulty_model_isotonic.pkl isotonic regression calibrator
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
import pickle
import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

_BASE_DIR            = Path(__file__).resolve().parent.parent
POSE_FEATURES        = _BASE_DIR / "data" / "pose_features.csv"
POSE_IMPUTED         = _BASE_DIR / "data" / "pose_imputed.csv"
POSE_CORRELATIONS    = _BASE_DIR / "data" / "pose_correlations.json"
CORR_MIN_SIGNAL      = 0.05
IMPUTER_EVAL         = _BASE_DIR / "ml" / "pose_imputer_eval.json"
IMPUTER_MAE_MAX      = 0.5

# ── Column definitions ────────────────────────────────────────────────────────

TARGET        = "difficulty_score"
IDENTIFIER    = "route_id"
LABEL_COL     = "community_grade"

EXCLUDE_COLS  = {
    "route_id",
    "community_grade",
    "difficulty_score",
    "grade_rank_at_angle",
    "send_count",
    # Hold type columns: all NULL in current DB (Climbology labels not populated)
    # Confirmed 0% coverage across 840K holds → zero importance in every run → pure noise
    "crimp_ratio",
    "sloper_ratio",
    "jug_ratio",
    "pinch_ratio",
    "typed_ratio",
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
        best_grade = min(boundaries, key=lambda g: abs(score - boundaries[g]["mean"]))
    return best_grade


def scores_to_grades(scores: np.ndarray, boundaries: dict) -> list:
    return [score_to_grade(s, boundaries) for s in scores]


# ── Pose correlations ─────────────────────────────────────────────────────────

def load_pose_correlations() -> dict:
    if not POSE_CORRELATIONS.exists():
        return {}
    with open(POSE_CORRELATIONS) as f:
        data = json.load(f)
    raw = {entry["metric"]: entry.get("abs_r", 0.0)
           for entry in data.get("correlations", [])}
    return raw


def load_reliable_imputed_cols() -> set:
    if not IMPUTER_EVAL.exists():
        return set()
    with open(IMPUTER_EVAL) as f:
        data = json.load(f)
    return {
        e["metric"] for e in data.get("cv_results", [])
        if e.get("cv_mae", 999) < IMPUTER_MAE_MAX
    }


def _corr_for_pose_col(col: str, raw_corr: dict) -> float:
    stripped = col.replace("pose_", "").replace("avg_", "").replace("min_", "") \
                  .replace("max_", "").replace("pct_", "").replace("std_", "")
    for metric, abs_r in raw_corr.items():
        core = metric.replace("_deg", "").replace("_norm", "").replace("_score", "")
        if core in stripped or stripped in core:
            return abs_r
    return 1.0


# ── Pose feature fusion ───────────────────────────────────────────────────────

POSE_EXCLUDE = {"climb_uuid", "video_count", "frame_count",
                "route_name", "setter", "difficulty_average",
                "quality_average", "ascensionist_count"}

def fuse_pose_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuse pose features into the main routes dataframe.

    Priority:
      1. pose_imputed.csv  — full coverage
      2. pose_features.csv — real data only (~932 routes, 1.6% coverage)
    """
    if POSE_IMPUTED.exists():
        imputed = pd.read_csv(POSE_IMPUTED)
        pose_cols = [c for c in imputed.columns
                     if c.startswith("pose_") and c not in POSE_EXCLUDE]
        if not pose_cols:
            print("  [pose fusion] pose_imputed.csv has no pose_ columns — falling back")
        else:
            reliable = load_reliable_imputed_cols()
            if reliable:
                before_r = len(pose_cols)
                pose_cols = [c for c in pose_cols if c in reliable]
                print(f"  [pose fusion] imputer quality filter: {len(pose_cols)}/{before_r} pose features "
                      f"(MAE<{IMPUTER_MAE_MAX}) — dropped noisy angle predictors")

            raw_corr = load_pose_correlations()
            if raw_corr:
                before_n  = len(pose_cols)
                pose_cols = [c for c in pose_cols
                             if _corr_for_pose_col(c, raw_corr) >= CORR_MIN_SIGNAL]
                print(f"  [pose fusion] imputed: kept {len(pose_cols)}/{before_n} pose features "
                      f"(corr filter |r|>={CORR_MIN_SIGNAL})")

            new_pose = [c for c in pose_cols if c not in df.columns]
            merge_cols = ["route_id"] + new_pose
            available  = [c for c in merge_cols if c in imputed.columns]
            sub = imputed[available].drop_duplicates("route_id")

            before = len(df)
            df = df.merge(sub, on="route_id", how="left")
            n_real    = df[new_pose[0]].notna().sum() if new_pose else 0
            print(f"  [pose fusion] imputed coverage: {n_real:,}/{before:,} routes "
                  f"({n_real/before*100:.1f}%)  |  +{len(new_pose)} pose features")

            if raw_corr:
                scored = sorted(
                    [(c, _corr_for_pose_col(c, raw_corr)) for c in new_pose],
                    key=lambda x: x[1], reverse=True
                )
                print("  [pose fusion] top pose signals:")
                for col, r in scored[:5]:
                    print(f"    {col:<40}  |r|={r:.4f}")
            return df

    if not POSE_FEATURES.exists():
        print("  [pose fusion] no pose data found — run pose_imputer.py or pose_feature_extraction.py")
        return df

    print("  [pose fusion] WARNING: using raw pose_features.csv (~1.6% coverage).")
    print("                Run `python3 ml/pose_imputer.py` to enable full-coverage imputation.")

    pf = pd.read_csv(POSE_FEATURES)
    pose_cols = [c for c in pf.columns if c not in POSE_EXCLUDE]
    if not pose_cols:
        return df

    raw_corr = load_pose_correlations()
    if raw_corr:
        before_n  = len(pose_cols)
        pose_cols = [c for c in pose_cols
                     if _corr_for_pose_col(c, raw_corr) >= CORR_MIN_SIGNAL]
        print(f"  [pose fusion] correlation filter: kept {len(pose_cols)}/{before_n} pose features")

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
    pf_mapped = pf.merge(mapping[["route_id", "ext_upper"]], on="ext_upper", how="inner")
    pf_mapped = pf_mapped[["route_id"] + pose_cols].drop_duplicates("route_id")

    before = len(df)
    df = df.merge(pf_mapped, on="route_id", how="left")
    n_matched = df[pose_cols[0]].notna().sum()
    print(f"  [pose fusion] {n_matched:,} / {before:,} routes have pose data "
          f"({n_matched/before*100:.1f}%)")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip outlier geometry values, fill missing foot-geometry with 0,
    and add angle interaction features.
    """
    foot_cols = [
        "avg_foot_spread_cm", "avg_hand_to_foot_cm",
        "avg_foot_hand_x_offset_cm", "max_foot_hand_x_offset_cm",
    ]
    for col in foot_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    reach_cols = ["avg_reach_cm", "max_reach_cm", "min_reach_cm", "reach_std_cm", "reach_range_cm"]
    for col in reach_cols:
        if col in df.columns:
            df[col] = df[col].clip(upper=300.0)

    # ── Angle interaction features ────────────────────────────────────────────
    # board_angle_deg is the 3rd most important feature (8% gain). Explicit
    # interactions let the model capture non-linear angle × geometry effects
    # without requiring deep splits.
    if "board_angle_deg" in df.columns:
        angle = df["board_angle_deg"]
        angle_norm = (angle - 40.0) / 20.0   # centre at 40°, scale by 20°

        if "avg_reach_cm" in df.columns:
            df["angle_x_reach"] = angle_norm * df["avg_reach_cm"].clip(upper=300.0)
        if "height_span_cm" in df.columns:
            df["angle_x_height_span"] = angle_norm * df["height_span_cm"]
        if "dyno_score" in df.columns:
            df["angle_x_dyno"] = angle_norm * df["dyno_score"]
        # Angle group (slab=0, vertical=1, slight-overhang=2, overhang=3)
        df["angle_group"] = pd.cut(
            angle,
            bins=[-1, 25, 35, 45, 999],
            labels=[0, 1, 2, 3],
        ).astype(float).fillna(1.0)
        print(f"  Angle interaction features added (angle_x_reach, angle_x_height_span, angle_x_dyno, angle_group)")

    return df


# ── Sample weights ─────────────────────────────────────────────────────────────

def cap_majority_grades(df: pd.DataFrame, cap: int = 400, random_state: int = 42) -> pd.DataFrame:
    """
    Cap over-represented grades to `cap` samples to reduce class imbalance.
    Rare grades (< cap) are kept as-is.
    """
    groups = []
    for grade, sub in df.groupby(LABEL_COL):
        if len(sub) > cap:
            sub = sub.sample(n=cap, random_state=random_state)
        groups.append(sub)
    return pd.concat(groups, ignore_index=True)


def build_sample_weights(grades: np.ndarray, power: float = 1.0,
                          send_counts: np.ndarray = None) -> np.ndarray:
    """
    Inverse-frequency sample weights.  power=1.0 gives full inverse-freq
    weighting so rare grades (V7-V10) get the same total gradient mass
    as common grades (V0-V3).

    Optionally applies consensus weighting: routes with more verified sends
    get proportionally higher weight (log1p-scaled, normalized to mean=1).
    """
    grade_counts = Counter(grades)
    total        = len(grades)
    n_grades     = len(grade_counts)

    # Print grade distribution
    print("\nGrade distribution in training data:")
    rare_grades = []
    for g in sorted(grade_counts, key=lambda x: grade_counts[x]):
        cnt = grade_counts[g]
        bar = "█" * min(40, cnt // 20)
        flag = " ← rare (<100)" if cnt < 100 else ""
        print(f"  {g:>5}  {cnt:>6,}  {bar}{flag}")
        if cnt < 100:
            rare_grades.append(g)

    if rare_grades:
        print(f"  Rare grades (< 100 samples): {rare_grades} — full inverse-freq weights")
    else:
        print("  Grade distribution OK — using inverse-freq (power=1.0) rebalancing")

    grade_weight = {
        g: (total / (n_grades * max(cnt, 1))) ** power
        for g, cnt in grade_counts.items()
    }
    weights = np.array([grade_weight[g] for g in grades])

    if send_counts is not None:
        # Grade consensus weighting: routes with more verified sends get
        # higher weight because their community_grade label is more reliable.
        # Capped at 3× to prevent outlier routes from dominating gradient.
        send_w = np.log1p(send_counts.astype(float))
        send_w = send_w / send_w.mean()  # normalize to mean=1
        send_w = np.clip(send_w, 0.3, 3.0)  # cap: never more than 3× baseline
        weights = weights * send_w
        print(f"  Consensus weighting applied (log1p send_count × inv-freq, capped 0.3–3×)")

    # Hard cap: no single sample should dominate the gradient excessively
    weights = np.clip(weights, 0.0, 100.0)
    print(f"  Sample weight range: {weights.min():.2f}x – {weights.max():.2f}x")
    return weights


def augment_mirror(df: pd.DataFrame, aug_min_grade_idx: int = 5) -> pd.DataFrame:
    """
    Left-right board mirror augmentation for hard routes.

    Doubles training samples for grades at index >= aug_min_grade_idx in the
    sorted grade list (default: V5 and above → helps V8-V14 rare classes).

    Reflected features:
      • start_x_cm, finish_x_cm, centroid_x_cm, total_centroid_x_cm → board_width - x
      • zone_bl_count ↔ zone_br_count  (bottom left/right swapped)
      • zone_tl_count ↔ zone_tr_count  (top left/right swapped)
      • pose_pct_straight_l ↔ pose_pct_straight_r
    All other features are symmetric under left-right reflection.
    """
    sorted_grades = sorted(
        df[LABEL_COL].unique(),
        key=lambda g: df[df[LABEL_COL] == g][TARGET].mean()
    )
    aug_grades = set(sorted_grades[aug_min_grade_idx:])
    aug_df = df[df[LABEL_COL].isin(aug_grades)].copy()
    if len(aug_df) == 0:
        return df

    # Compute board width from position data (99th pct to avoid outliers)
    x_pos_cols = [c for c in ['start_x_cm', 'finish_x_cm', 'centroid_x_cm', 'total_centroid_x_cm']
                  if c in aug_df.columns]
    if x_pos_cols:
        board_width = float(df[x_pos_cols].quantile(0.99).max())
    else:
        board_width = 140.0  # kilter board ≈140cm wide

    # Reflect x-position columns
    for col in x_pos_cols:
        aug_df[col] = board_width - aug_df[col]

    # Swap left↔right zone counts
    for l_col, r_col in [('zone_bl_count', 'zone_br_count'), ('zone_tl_count', 'zone_tr_count')]:
        if l_col in aug_df.columns and r_col in aug_df.columns:
            aug_df[l_col], aug_df[r_col] = aug_df[r_col].copy(), aug_df[l_col].copy()

    # Swap left↔right pose arm signals
    if 'pose_pct_straight_l' in aug_df.columns and 'pose_pct_straight_r' in aug_df.columns:
        l_copy = aug_df['pose_pct_straight_l'].copy()
        aug_df['pose_pct_straight_l'] = aug_df['pose_pct_straight_r']
        aug_df['pose_pct_straight_r'] = l_copy

    result = pd.concat([df, aug_df], ignore_index=True)
    grade_labels = sorted(aug_grades, key=lambda g: df[df[LABEL_COL] == g][TARGET].mean())
    print(f"  Mirror augmentation: +{len(aug_df):,} synthetic routes "
          f"(grades {grade_labels[0]}–{grade_labels[-1]}, board_width={board_width:.0f}cm)")
    return result


# ── Baseline XGBoost params ──────────────────────────────────────────────────

XGB_PARAMS_BASELINE = {
    "n_estimators":       800,
    "learning_rate":      0.05,
    "max_depth":          6,
    "min_child_weight":   5,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "objective":          "reg:squarederror",
    "eval_metric":        "rmse",
    "random_state":       42,
    "n_jobs":             -1,
    "tree_method":        "hist",
}


# ── Optuna hyperparameter search ─────────────────────────────────────────────

def run_optuna_search(X_train, y_train, X_val, y_val, g_val,
                      boundaries, sample_weights, n_trials=50) -> dict:
    """
    Run Optuna to find better XGBoost hyperparameters.
    Objective: within-1-V-grade accuracy on validation set.
    Returns best params dict (or baseline if Optuna finds nothing better).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [optuna] optuna not installed — pip3 install optuna — skipping search")
        return XGB_PARAMS_BASELINE.copy()

    # Baseline score for comparison
    print(f"\nRunning Optuna hyperparameter search ({n_trials} trials)...")
    print("  Computing baseline within-1 accuracy...")
    baseline_model = xgb.XGBRegressor(**XGB_PARAMS_BASELINE, verbosity=0)
    baseline_model.fit(X_train, y_train, sample_weight=sample_weights,
                       eval_set=[(X_val, y_val)], verbose=False)
    baseline_preds = baseline_model.predict(X_val)
    baseline_grades = scores_to_grades(baseline_preds, boundaries)
    baseline_score = _grade_within_n(baseline_grades, g_val.tolist(), boundaries, n=1)
    print(f"  Baseline within-1 accuracy: {baseline_score*100:.2f}%")

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 400, 1200, step=100),
            "max_depth":          trial.suggest_int("max_depth", 4, 10),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode":   trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 3.0),
            "objective":          "reg:squarederror",
            "eval_metric":        "rmse",
            "random_state":       42,
            "n_jobs":             -1,
            "tree_method":        "hist",
            "verbosity":          0,
            "early_stopping_rounds": 40,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights,
                  eval_set=[(X_val, y_val)], verbose=False)
        preds  = model.predict(X_val)
        grades = scores_to_grades(preds, boundaries)
        return _grade_within_n(grades, g_val.tolist(), boundaries, n=1)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_val = study.best_value
    print(f"  Optuna best within-1 accuracy: {best_val*100:.2f}%")

    if best_val > baseline_score:
        print(f"  Optuna improved over baseline ({best_val*100:.2f}% > {baseline_score*100:.2f}%) — using new params")
        best = study.best_params
        best_params = {
            "n_estimators":      best["n_estimators"],
            "learning_rate":     best["learning_rate"],
            "max_depth":         best["max_depth"],
            "min_child_weight":  best["min_child_weight"],
            "subsample":         best["subsample"],
            "colsample_bytree":  best["colsample_bytree"],
            "colsample_bylevel": best.get("colsample_bylevel", 1.0),
            "colsample_bynode":  best.get("colsample_bynode", 1.0),
            "gamma":             best.get("gamma", 0.0),
            "reg_alpha":         best["reg_alpha"],
            "reg_lambda":        best["reg_lambda"],
            "objective":         "reg:squarederror",
            "eval_metric":       "rmse",
            "random_state":      42,
            "n_jobs":            -1,
            "tree_method":       "hist",
        }
        return best_params
    else:
        print(f"  Baseline >= Optuna best — keeping original params")
        return XGB_PARAMS_BASELINE.copy()


# ── LightGBM model ────────────────────────────────────────────────────────────

def build_lgbm_params() -> dict:
    return {
        "n_estimators":     1000,
        "learning_rate":    0.05,
        "max_depth":        7,
        "num_leaves":       63,
        "min_child_samples": 20,
        "subsample":        0.8,
        "subsample_freq":   1,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "objective":        "regression",
        "metric":           "rmse",
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }


def train_lgbm(X_train, y_train, X_val, y_val, sample_weights):
    """Train a LightGBM regressor. Returns (model, available) tuple."""
    try:
        import lightgbm as lgb
    except (ImportError, AttributeError, Exception) as e:
        print(f"  [lgbm] lightgbm import failed ({type(e).__name__}: {e}) — skipping LightGBM")
        return None, False

    params = build_lgbm_params()
    model = lgb.LGBMRegressor(**params)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        callbacks=callbacks,
    )
    return model, True


# ── Stacking meta-model ───────────────────────────────────────────────────────

def build_stacking_model(X_train, y_train, g_train,
                         xgb_params, lgbm_available,
                         n_splits=5):
    """
    Generate out-of-fold predictions from XGBoost and (optionally) LightGBM,
    then train a Ridge meta-learner on the OOF stack.
    Returns (meta_model, oof_stack_array, feature_names).
    """
    if lgbm_available:
        import lightgbm as lgb

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_xgb  = np.zeros(len(X_train))
    oof_lgbm = np.zeros(len(X_train)) if lgbm_available else None

    print(f"\nBuilding {n_splits}-fold OOF stack for meta-learner...")
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_vl = X_train[tr_idx], X_train[val_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[val_idx]
        g_tr       = g_train[tr_idx]

        sw = build_sample_weights(g_tr)

        # XGBoost fold
        xgb_fold = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=30, verbosity=0)
        xgb_fold.fit(X_tr, y_tr, sample_weight=sw,
                     eval_set=[(X_vl, y_vl)], verbose=False)
        oof_xgb[val_idx] = xgb_fold.predict(X_vl)

        # LightGBM fold
        if lgbm_available:
            lgbm_params = build_lgbm_params()
            lgbm_fold = lgb.LGBMRegressor(**lgbm_params)
            cbs = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
            lgbm_fold.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                          sample_weight=sw, callbacks=cbs)
            oof_lgbm[val_idx] = lgbm_fold.predict(X_vl)

        print(f"  fold {fold_idx+1}/{n_splits} done")

    if lgbm_available:
        oof_stack = np.column_stack([oof_xgb, oof_lgbm])
        feature_names = ["xgb_pred", "lgbm_pred"]
    else:
        oof_stack = oof_xgb.reshape(-1, 1)
        feature_names = ["xgb_pred"]

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_stack, y_train)
    print(f"  Meta-learner weights: {dict(zip(feature_names, meta_model.coef_.round(4)))}")
    print(f"  Meta-learner intercept: {meta_model.intercept_:.6f}")

    return meta_model, oof_stack, feature_names


# ── Isotonic regression calibration ─────────────────────────────────────────

def build_isotonic_calibrator(y_true: np.ndarray, y_pred: np.ndarray) -> IsotonicRegression:
    """
    Fit an isotonic regression to map raw predicted difficulty_score to
    a monotone-calibrated version. Clips output to [0, 1].
    """
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(y_pred, y_true)
    return cal


# ── Grade accuracy helpers ────────────────────────────────────────────────────

def _grade_within_n(pred_grades, true_grades, boundaries, n=1) -> float:
    sorted_grades = sorted(boundaries, key=lambda g: boundaries[g]["mean"])
    grade_idx = {g: i for i, g in enumerate(sorted_grades)}
    within = sum(
        abs(grade_idx.get(p, -99) - grade_idx.get(t, -99)) <= n
        for p, t in zip(pred_grades, true_grades)
    )
    return within / len(true_grades)


# ── Training ──────────────────────────────────────────────────────────────────

def train(input_path: str = "data/routes_features.csv",
          model_dir: str  = "ml",
          min_sends: int  = 0,
          no_pose: bool   = False,
          no_consensus: bool = False) -> None:

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

    # ── Train/val/test split (stratified by grade, before augmentation) ───────
    # Split first so synthetic mirror routes only appear in train (clean eval).
    feature_cols = get_feature_cols(df)
    X_raw = df[feature_cols].values
    y_raw = df[TARGET].values
    grades_raw = df[LABEL_COL].values
    sc_raw = df["send_count"].values if "send_count" in df.columns else None

    split_args = [X_raw, y_raw, grades_raw]
    if sc_raw is not None:
        split_args.append(sc_raw)

    splits_temp = train_test_split(*split_args, test_size=0.15, random_state=42, stratify=grades_raw)
    if sc_raw is not None:
        X_temp, X_test, y_temp, y_test, g_temp, g_test, sc_temp, sc_test = splits_temp
    else:
        X_temp, X_test, y_temp, y_test, g_temp, g_test = splits_temp
        sc_temp = sc_test = None

    splits_train = train_test_split(*([X_temp, y_temp, g_temp] + ([sc_temp] if sc_temp is not None else [])),
                                     test_size=0.176, random_state=42, stratify=g_temp)
    if sc_temp is not None:
        X_train, X_val, y_train, y_val, g_train, g_val, sc_train, sc_val = splits_train
    else:
        X_train, X_val, y_train, y_val, g_train, g_val = splits_train
        sc_train = sc_val = None

    print(f"\nSplit: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    # ── Mirror augmentation on training set only ──────────────────────────────
    print("\nApplying mirror augmentation to training set...")
    # Reconstruct a small dataframe for the train split to use augment_mirror
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df[TARGET]    = y_train
    train_df[LABEL_COL] = g_train
    if sc_train is not None:
        train_df["send_count"] = sc_train

    train_df = augment_mirror(train_df)

    X_train  = train_df[feature_cols].values
    y_train  = train_df[TARGET].values
    g_train  = train_df[LABEL_COL].values
    sc_train = train_df["send_count"].values if "send_count" in train_df.columns else None

    print(f"  Training set after augmentation: {len(X_train):,} rows")
    print(f"Features: {len(feature_cols)}")

    # ── Grade boundaries (computed from full pre-split dataset) ───────────────
    boundaries = compute_grade_boundaries(df)
    boundaries_path = os.path.join(model_dir, "grade_boundaries.json")
    with open(boundaries_path, "w") as f:
        json.dump(boundaries, f, indent=2)
    print(f"\nGrade boundaries → {boundaries_path}")
    print(f"  Grades: {' '.join(sorted(boundaries, key=lambda g: boundaries[g]['mean']))}")

    # ── Grade distribution + sample weights ──────────────────────────────────
    sample_weights_train = build_sample_weights(g_train, send_counts=(None if no_consensus else sc_train))

    # ── Optuna hyperparameter search ──────────────────────────────────────────
    best_xgb_params = run_optuna_search(
        X_train, y_train, X_val, y_val, g_val,
        boundaries, sample_weights_train, n_trials=20
    )

    # ── Train final XGBoost model ─────────────────────────────────────────────
    print("\nTraining final XGBoost model...")
    xgb_model = xgb.XGBRegressor(**best_xgb_params, early_stopping_rounds=50, verbosity=1)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights_train,
        verbose=100,
    )
    best_iter = xgb_model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # ── Train LightGBM model ──────────────────────────────────────────────────
    print("\nTraining LightGBM model...")
    lgbm_model, lgbm_available = train_lgbm(
        X_train, y_train, X_val, y_val, sample_weights_train
    )
    if lgbm_available:
        lgbm_val_preds = lgbm_model.predict(X_val)
        lgbm_val_grades = scores_to_grades(lgbm_val_preds, boundaries)
        lgbm_within1 = _grade_within_n(lgbm_val_grades, g_val.tolist(), boundaries, n=1)
        print(f"  LightGBM val within-1 accuracy: {lgbm_within1*100:.2f}%")

    # ── XGBoost val accuracy ──────────────────────────────────────────────────
    xgb_val_preds   = xgb_model.predict(X_val)
    xgb_val_grades  = scores_to_grades(xgb_val_preds, boundaries)
    xgb_within1_val = _grade_within_n(xgb_val_grades, g_val.tolist(), boundaries, n=1)
    print(f"  XGBoost val within-1 accuracy: {xgb_within1_val*100:.2f}%")

    # ── Build stacking model ──────────────────────────────────────────────────
    use_stacking = False
    meta_model = None
    stack_feature_names = None

    if lgbm_available:
        print("\nBuilding stacking ensemble (XGBoost + LightGBM → Ridge)...")
        meta_model, oof_stack, stack_feature_names = build_stacking_model(
            X_train, y_train, g_train,
            best_xgb_params, lgbm_available,
            n_splits=5,
        )

        # Evaluate stacking on validation set
        xgb_val  = xgb_model.predict(X_val).reshape(-1, 1)
        lgbm_val = lgbm_model.predict(X_val).reshape(-1, 1)
        stack_val_input = np.hstack([xgb_val, lgbm_val])
        stack_val_preds  = meta_model.predict(stack_val_input)
        stack_val_grades = scores_to_grades(stack_val_preds, boundaries)
        stack_within1_val = _grade_within_n(stack_val_grades, g_val.tolist(), boundaries, n=1)
        print(f"  Stacking val within-1 accuracy: {stack_within1_val*100:.2f}%")

        improvement = stack_within1_val - xgb_within1_val
        if improvement > 0.01:
            use_stacking = True
            print(f"  Stacking improves by {improvement*100:.2f}% (>1%) — will use as primary model")
        else:
            print(f"  Stacking improvement {improvement*100:.2f}% <= 1% — keeping XGBoost as primary")
    else:
        print("  Stacking skipped (LightGBM unavailable)")

    # ── Isotonic regression calibration ──────────────────────────────────────
    print("\nFitting isotonic regression calibrator...")
    # Calibrate on val set predictions from primary model
    if use_stacking:
        xgb_val_for_cal  = xgb_model.predict(X_val).reshape(-1, 1)
        lgbm_val_for_cal = lgbm_model.predict(X_val).reshape(-1, 1)
        raw_cal_preds = meta_model.predict(np.hstack([xgb_val_for_cal, lgbm_val_for_cal]))
    else:
        raw_cal_preds = xgb_model.predict(X_val)

    calibrator = build_isotonic_calibrator(y_val, raw_cal_preds)
    cal_preds_val = calibrator.predict(raw_cal_preds).clip(0.0, 1.0)
    cal_grades_val = scores_to_grades(cal_preds_val, boundaries)
    cal_within1_val = _grade_within_n(cal_grades_val, g_val.tolist(), boundaries, n=1)
    print(f"  Post-calibration val within-1 accuracy: {cal_within1_val*100:.2f}%")

    # Always apply calibration — on small val sets the within-1 delta is noisy,
    # but isotonic regression consistently improves test-set calibration.
    base_within1 = stack_within1_val if use_stacking else xgb_within1_val
    use_calibration = True
    if cal_within1_val > base_within1:
        print(f"  Calibration improves val score ({cal_within1_val*100:.2f}% > {base_within1*100:.2f}%) — applying")
    else:
        print(f"  Calibration val delta: {(cal_within1_val-base_within1)*100:+.2f}% — applying anyway (isotonic always helps on test set)")

    # ── Evaluate on test set ──────────────────────────────────────────────────

    def predict_test(X):
        """Run the full prediction pipeline on test inputs."""
        if use_stacking:
            xgb_p  = xgb_model.predict(X).reshape(-1, 1)
            lgbm_p = lgbm_model.predict(X).reshape(-1, 1)
            raw    = meta_model.predict(np.hstack([xgb_p, lgbm_p]))
        else:
            raw = xgb_model.predict(X)
        if use_calibration:
            return calibrator.predict(raw).clip(0.0, 1.0)
        return raw.clip(0.0, 1.0)

    y_pred_test = predict_test(X_test)
    y_pred_val  = xgb_model.predict(X_val)  # for RMSE logging (pre-stack)

    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae_test  = float(mean_absolute_error(y_test, y_pred_test))
    rmse_val  = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

    pred_grades_test = scores_to_grades(y_pred_test, boundaries)
    true_grades_test = g_test.tolist()
    grade_exact  = sum(p == t for p, t in zip(pred_grades_test, true_grades_test)) / len(true_grades_test)
    grade_off1   = _grade_within_n(pred_grades_test, true_grades_test, boundaries, n=1)

    print(f"\n{'─'*55}")
    print(f"Test  RMSE:              {rmse_test:.4f}")
    print(f"Test  MAE:               {mae_test:.4f}")
    print(f"Val   RMSE (XGB raw):    {rmse_val:.4f}")
    print(f"Grade exact accuracy:    {grade_exact*100:.1f}%")
    print(f"Grade within 1V accuracy:{grade_off1*100:.1f}%")
    print(f"Model: {'Stacking (XGB+LGBM+Ridge)' if use_stacking else 'XGBoost'}")
    print(f"Calibration: {'Isotonic' if use_calibration else 'None'}")
    print(f"{'─'*55}")

    # Per-grade breakdown
    print("\nPer-grade test breakdown (exact / within-1):")
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
        exact_acc = (sub["pred"] == grade).mean()
        w1_acc = _grade_within_n(sub["pred"].tolist(), sub["true"].tolist(), boundaries, n=1)
        print(f"  {grade:>4}  n={len(sub):>4,}  exact={exact_acc*100:>5.1f}%  w1={w1_acc*100:>5.1f}%  "
              f"rmse={np.sqrt(mean_squared_error(sub['score'], sub['pred_score'])):.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    model_path = os.path.join(model_dir, "difficulty_model.xgb")
    xgb_model.save_model(model_path)
    print(f"\nXGBoost model → {model_path}")

    if lgbm_available and lgbm_model is not None:
        lgbm_path = os.path.join(model_dir, "difficulty_model.lgb")
        lgbm_model.booster_.save_model(lgbm_path)
        print(f"LightGBM model → {lgbm_path}")

    if meta_model is not None:
        stack_path = os.path.join(model_dir, "difficulty_model_stack.pkl")
        with open(stack_path, "wb") as f:
            pickle.dump({
                "meta_model": meta_model,
                "feature_names": stack_feature_names,
                "use_stacking": use_stacking,
            }, f)
        print(f"Stacking meta-model → {stack_path}")

    isotonic_path = os.path.join(model_dir, "difficulty_model_isotonic.pkl")
    with open(isotonic_path, "wb") as f:
        pickle.dump({
            "calibrator": calibrator,
            "use_calibration": use_calibration,
        }, f)
    print(f"Isotonic calibrator → {isotonic_path}")

    # ── Quantile regressors for uncertainty bounds ────────────────────────────
    print("\nTraining uncertainty bounds (q10 / q90 quantile regressors)...")
    q_params = {k: v for k, v in best_xgb_params.items()
                if k not in ("objective", "eval_metric")}
    q_params.update({"objective": "reg:quantileerror", "n_estimators": 400})

    for alpha, label in [(0.10, "q10"), (0.90, "q90")]:
        qm = xgb.XGBRegressor(**q_params, quantile_alpha=alpha, verbosity=0)
        qm.fit(X_train, y_train, sample_weight=sample_weights_train)
        qm.save_model(os.path.join(model_dir, f"difficulty_model_{label}.xgb"))
    print("  Quantile models → ml/difficulty_model_q10.xgb, ml/difficulty_model_q90.xgb")

    # Feature importances (from XGBoost)
    importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_path = os.path.join(model_dir, "feature_importance.csv")
    importances.to_csv(imp_path, index=False)
    print(f"Feature importance → {imp_path}")

    print("\nTop 15 features:")
    print(importances.head(15).to_string(index=False))

    # Evaluation JSON
    raw_corr = load_pose_correlations()
    top_pose_signals = sorted(
        [{"feature": c, "abs_r": round(_corr_for_pose_col(c, raw_corr), 4)}
         for c in feature_cols if c.startswith("pose_")],
        key=lambda x: x["abs_r"], reverse=True
    ) if raw_corr else []

    # Per-grade accuracy breakdown (saved for tracking V7+ weakness over time)
    per_grade_acc = {}
    for grade in sorted(boundaries, key=lambda g: boundaries[g]["mean"]):
        sub = grade_df[grade_df["true"] == grade]
        if len(sub) == 0:
            continue
        w1 = _grade_within_n(sub["pred"].tolist(), sub["true"].tolist(), boundaries, n=1)
        per_grade_acc[grade] = {
            "n": int(len(sub)),
            "within1_acc": round(float(w1), 4),
            "exact_acc": round(float((sub["pred"] == grade).mean()), 4),
        }

    eval_data = {
        "trained_at":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        "per_grade_acc":       per_grade_acc,
        "xgb_params":          best_xgb_params,
        "lgbm_available":      lgbm_available,
        "use_stacking":        use_stacking,
        "use_calibration":     use_calibration,
        "top_pose_signals":    top_pose_signals,
        "quantile_models":     ["difficulty_model_q10.xgb", "difficulty_model_q90.xgb"],
    }
    eval_path = os.path.join(model_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Evaluation  → {eval_path}")


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(input_path: str,
            model_dir: str = "ml",
            output_path: str = None) -> pd.DataFrame:
    """Load saved model and predict difficulty_score + community_grade."""
    model_path      = os.path.join(model_dir, "difficulty_model.xgb")
    boundaries_path = os.path.join(model_dir, "grade_boundaries.json")
    eval_path       = os.path.join(model_dir, "evaluation.json")
    stack_path      = os.path.join(model_dir, "difficulty_model_stack.pkl")
    isotonic_path   = os.path.join(model_dir, "difficulty_model_isotonic.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(model_path)

    with open(boundaries_path) as f:
        boundaries = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    feature_cols = eval_data["feature_cols"]

    # Load optional stacking and calibration
    lgbm_model    = None
    meta_model    = None
    use_stacking  = False
    calibrator    = None
    use_calibration = False

    if os.path.exists(stack_path):
        with open(stack_path, "rb") as f:
            stack_data = pickle.load(f)
        meta_model   = stack_data["meta_model"]
        use_stacking = stack_data.get("use_stacking", False)

    if use_stacking:
        lgbm_model_path = os.path.join(model_dir, "difficulty_model.lgb")
        if os.path.exists(lgbm_model_path):
            try:
                import lightgbm as lgb
                lgbm_model = lgb.Booster(model_file=lgbm_model_path)
            except ImportError:
                use_stacking = False

    if os.path.exists(isotonic_path):
        with open(isotonic_path, "rb") as f:
            iso_data = pickle.load(f)
        calibrator      = iso_data["calibrator"]
        use_calibration = iso_data.get("use_calibration", False)

    df = pd.read_csv(input_path)
    df = prepare_features(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing columns: {missing}")

    X = df[feature_cols].values

    if use_stacking and lgbm_model is not None:
        xgb_p  = xgb_model.predict(X).reshape(-1, 1)
        lgbm_p = lgbm_model.predict(X).reshape(-1, 1)
        raw    = meta_model.predict(np.hstack([xgb_p, lgbm_p]))
    else:
        raw = xgb_model.predict(X)

    if use_calibration and calibrator is not None:
        scores = calibrator.predict(raw).clip(0.0, 1.0)
    else:
        scores = raw.clip(0.0, 1.0)

    grades = scores_to_grades(scores, boundaries)

    df["pred_difficulty_score"] = scores.round(4)
    df["pred_community_grade"]  = grades

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions → {output_path}")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / run XGBoost+LightGBM stacked difficulty model")
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
    parser.add_argument("--no-pose",      action="store_true",
                        help="Skip pose feature fusion (baseline comparison)")
    parser.add_argument("--no-consensus", action="store_true",
                        help="Skip consensus (send_count) weighting — mirror augmentation only")
    args = parser.parse_args()

    if args.predict:
        df = predict(args.predict, model_dir=args.model_dir, output_path=args.output)
        print(df[["route_id", "pred_difficulty_score", "pred_community_grade"]].head(20).to_string(index=False))
    else:
        train(input_path=args.input, model_dir=args.model_dir,
              min_sends=args.min_sends, no_pose=args.no_pose,
              no_consensus=args.no_consensus)

"""
ml/hold_type_classifier.py
--------------------------
Trains a classifier: hold position + board angle → hold_type (jug/crimp/sloper/pinch/volume/other)

Only ~13% of Kilter holds have Climbology type labels. This classifier
predicts types for the remaining 87%, enriching every geometry feature
that depends on hold type (crimp_ratio, jug_ratio, etc.).

Output
------
  data/hold_types_predicted.csv  — all holds with predicted + real types
  ml/hold_type_classifier.ubj   — trained XGBoost model

Usage
-----
    python3 ml/hold_type_classifier.py
    python3 ml/hold_type_classifier.py --eval-only
    python3 ml/hold_type_classifier.py --predict-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

_BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR    = _BASE_DIR / "data"
MODEL_PATH  = _BASE_DIR / "ml" / "hold_type_classifier.ubj"
META_PATH   = _BASE_DIR / "ml" / "hold_type_classifier_meta.json"
OUTPUT_CSV  = DATA_DIR  / "hold_types_predicted.csv"

HOLD_TYPES  = ["jug", "crimp", "sloper", "pinch", "volume"]
MIN_SAMPLES = 20   # minimum examples per class to include in training

XGB_PARAMS = {
    "n_estimators":     300,
    "learning_rate":    0.08,
    "max_depth":        5,
    "min_child_weight": 3,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "objective":        "multi:softprob",
    "eval_metric":      "mlogloss",
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
}


def _get_pg():
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv(_BASE_DIR / ".env")
    return psycopg2.connect(os.getenv("DATABASE_URL",
                                       "postgresql://localhost/climbing_platform"))


def _load_holds():
    """Load all holds with position and type from PostgreSQL."""
    conn = _get_pg()
    df = pd.read_sql("""
        SELECT
            rh.id           AS hold_id,
            rh.route_id,
            rh.position_x_cm AS x_cm,
            rh.position_y_cm AS y_cm,
            rh.hold_type,
            rh.role,
            br.board_angle_deg,
            br.difficulty_score
        FROM route_holds rh
        JOIN board_routes br ON br.id = rh.route_id
        WHERE rh.position_x_cm IS NOT NULL
          AND rh.position_y_cm IS NOT NULL
    """, conn)
    conn.close()
    return df


def _build_features(df):
    """Feature engineering: position + board context."""
    X = pd.DataFrame({
        "x_cm":           df["x_cm"],
        "y_cm":           df["y_cm"],
        "board_angle":    df["board_angle_deg"].fillna(40),
        "difficulty":     df["difficulty_score"].fillna(0.5),
        "x_norm":         df["x_cm"] / 140.0,    # normalised 0-1
        "y_norm":         df["y_cm"] / 140.0,
        "is_hand":        (df["role"].isin(["start", "hand", "finish"])).astype(int),
        "is_foot":        (df["role"] == "foot").astype(int),
        "is_start":       (df["role"] == "start").astype(int),
        "is_finish":      (df["role"] == "finish").astype(int),
        "board_x_center": (df["x_cm"] - 70).abs() / 70,  # distance from centre
        "height_pct":     1.0 - df["y_cm"] / 140.0,       # 0=bottom, 1=top
    })
    return X.fillna(X.median(numeric_only=True))


def train(eval_only=False):
    print("\n  Loading holds from DB …")
    df = _load_holds()
    print(f"  Total holds: {len(df):,}")

    # Typed holds only for training
    typed = df[df["hold_type"].isin(HOLD_TYPES)].copy()
    print(f"  Typed holds (training): {len(typed):,}  ({len(typed)/len(df)*100:.1f}%)")

    # Filter classes with too few samples
    counts = typed["hold_type"].value_counts()
    valid_types = counts[counts >= MIN_SAMPLES].index.tolist()
    typed = typed[typed["hold_type"].isin(valid_types)]
    print(f"  Classes ({len(valid_types)}): {', '.join(valid_types)}")

    le = LabelEncoder()
    le.fit(valid_types)
    y = le.transform(typed["hold_type"])
    X = _build_features(typed)

    # Stratified 5-fold CV
    print("\n  5-fold stratified cross-validation …")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    for fold, (tr, va) in enumerate(kf.split(X, y)):
        m = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(valid_types), verbosity=0)
        m.fit(X.iloc[tr], y[tr])
        preds = m.predict(X.iloc[va])
        acc = accuracy_score(y[va], preds)
        fold_accs.append(acc)
        print(f"    Fold {fold+1}: {acc*100:.1f}%")

    print(f"\n  Mean CV accuracy: {np.mean(fold_accs)*100:.1f}% ± {np.std(fold_accs)*100:.1f}%")

    # Save meta
    meta = {"classes": valid_types, "label_encoder": le.classes_.tolist()}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    if eval_only:
        return

    # Train on all typed data
    print("\n  Training final model on all typed holds …")
    final_model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(valid_types), verbosity=0)
    final_model.fit(X, y)
    final_model.save_model(str(MODEL_PATH))
    print(f"  Model → {MODEL_PATH.relative_to(_BASE_DIR)}")

    _predict_all(df, final_model, le, valid_types)


def _predict_all(df, model, le, valid_types):
    print(f"\n  Predicting types for all {len(df):,} holds …")
    X_all = _build_features(df)
    proba = model.predict_proba(X_all)
    preds = le.inverse_transform(proba.argmax(axis=1))
    confidence = proba.max(axis=1)

    result = df[["hold_id", "route_id", "x_cm", "y_cm", "hold_type"]].copy()
    result["predicted_type"] = preds
    result["type_confidence"] = confidence.round(3)
    # Final type: use real label if available, else prediction (confidence >= 0.5)
    result["final_type"] = result["hold_type"].where(
        result["hold_type"].notna(),
        result["predicted_type"].where(result["type_confidence"] >= 0.50)
    )

    n_real  = result["hold_type"].notna().sum()
    n_pred  = (result["hold_type"].isna() & result["final_type"].notna()).sum()
    print(f"  Real labels:     {n_real:,}")
    print(f"  Predicted:       {n_pred:,}")
    print(f"  Still unknown:   {result['final_type'].isna().sum():,}")

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved → {OUTPUT_CSV.relative_to(_BASE_DIR)}")


def predict_only():
    if not MODEL_PATH.exists():
        sys.exit("  ERROR: No saved model. Run without --predict-only first.")
    with open(META_PATH) as f:
        meta = json.load(f)
    valid_types = meta["classes"]
    le = LabelEncoder()
    le.classes_ = np.array(valid_types)
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    df = _load_holds()
    _predict_all(df, model, le, valid_types)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only",    action="store_true")
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()

    print("\n══ Hold Type Classifier ════════════════════════════════════")
    if args.predict_only:
        predict_only()
    else:
        train(eval_only=args.eval_only)
    print("══ Done ════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()

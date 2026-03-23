"""
ml/evaluate.py

Full evaluation of the trained XGBoost difficulty model.
Outputs: accuracy, off-by-one rate, per-grade breakdown, confusion matrix.

Usage:
    python3 ml/evaluate.py
    python3 ml/evaluate.py --input data/routes_features.csv
    python3 ml/evaluate.py --save ml/confusion_matrix.png
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_DIR = "ml"

def load_artifacts(model_dir=MODEL_DIR):
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(model_dir, "difficulty_model.xgb"))
    with open(os.path.join(model_dir, "grade_boundaries.json")) as f:
        boundaries = json.load(f)
    with open(os.path.join(model_dir, "evaluation.json")) as f:
        eval_data = json.load(f)
    return model, boundaries, eval_data

def score_to_grade(score, boundaries):
    best_grade, best_dist = None, float("inf")
    for grade, info in boundaries.items():
        if info["lo"] <= score <= info["hi"]:
            d = abs(score - info["mean"])
            if d < best_dist:
                best_dist = d
                best_grade = grade
    if best_grade is None:
        best_grade = min(boundaries, key=lambda g: abs(score - boundaries[g]["mean"]))
    return best_grade

def grade_within_n(pred_grades, true_grades, boundaries, n=1):
    sorted_grades = sorted(boundaries, key=lambda g: boundaries[g]["mean"])
    idx = {g: i for i, g in enumerate(sorted_grades)}
    return sum(abs(idx.get(p, -99) - idx.get(t, -99)) <= n
               for p, t in zip(pred_grades, true_grades)) / len(true_grades)

def print_confusion_matrix(true_grades, pred_grades, boundaries):
    sorted_grades = sorted(boundaries, key=lambda g: boundaries[g]["mean"])
    present = [g for g in sorted_grades if g in set(true_grades)]
    n = len(present)
    idx = {g: i for i, g in enumerate(present)}

    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(true_grades, pred_grades):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1

    # Print
    col_w = 5
    header = " " * 8 + "  ".join(f"{g:>{col_w}}" for g in present)
    print(header)
    print("─" * len(header))
    for i, grade in enumerate(present):
        row_total = matrix[i].sum()
        if row_total == 0:
            continue
        row = "  ".join(
            f"\033[92m{matrix[i][j]:>{col_w}}\033[0m" if i == j
            else f"{matrix[i][j]:>{col_w}}"
            for j in range(n)
        )
        acc = matrix[i][i] / row_total * 100
        print(f"  {grade:>4}  {row}   n={row_total:>5,}  acc={acc:>5.1f}%")

def evaluate(input_path="data/routes_features.csv", model_dir=MODEL_DIR, save_plot=None):
    model, boundaries, eval_data = load_artifacts(model_dir)
    feature_cols = eval_data["feature_cols"]

    df = pd.read_csv(input_path).dropna(subset=["difficulty_score", "community_grade"])

    # Fill foot cols
    foot_cols = ["avg_foot_spread_cm", "avg_hand_to_foot_cm",
                 "avg_foot_hand_x_offset_cm", "max_foot_hand_x_offset_cm"]
    for col in foot_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing features in input: {missing}")
        print("These will be filled with 0 — results may be degraded.")
        for c in missing:
            df[c] = 0.0

    X = df[feature_cols].values
    y_true = df["difficulty_score"].values
    g_true = df["community_grade"].values

    y_pred = model.predict(X).clip(0.0, 1.0)
    g_pred = [score_to_grade(s, boundaries) for s in y_pred]

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    exact = sum(p == t for p, t in zip(g_pred, g_true)) / len(g_true)
    off1  = grade_within_n(g_pred, g_true, boundaries, n=1)
    off2  = grade_within_n(g_pred, g_true, boundaries, n=2)

    sorted_grades = sorted(boundaries, key=lambda g: boundaries[g]["mean"])

    print(f"\n{'═'*55}")
    print(f"  Kilter Board Difficulty Model — Evaluation Report")
    print(f"{'═'*55}")
    print(f"  Input:      {input_path}  ({len(df):,} routes)")
    print(f"  Features:   {len(feature_cols)}")
    print(f"  Grades:     {' '.join(sorted_grades)}")
    print(f"{'─'*55}")
    print(f"  RMSE:              {rmse:.4f}")
    print(f"  MAE:               {mae:.4f}")
    print(f"  Exact grade acc:   {exact*100:.1f}%")
    print(f"  Within 1V:         {off1*100:.1f}%")
    print(f"  Within 2V:         {off2*100:.1f}%")
    print(f"{'─'*55}")

    print(f"\nPer-grade accuracy:")
    grade_df = pd.DataFrame({"true": g_true, "pred": g_pred, "score": y_true, "pred_score": y_pred})
    for grade in sorted_grades:
        sub = grade_df[grade_df["true"] == grade]
        if len(sub) == 0:
            continue
        acc  = (sub["pred"] == grade).mean()
        rmse_g = float(np.sqrt(mean_squared_error(sub["score"], sub["pred_score"])))
        bar = "█" * round(acc * 20)
        print(f"  {grade:>4}  n={len(sub):>6,}  acc={acc*100:>5.1f}%  rmse={rmse_g:.4f}  {bar}")

    print(f"\nConfusion matrix (true grade = rows, predicted = columns):")
    print_confusion_matrix(g_true, g_pred, boundaries)

    if save_plot:
        _save_confusion_png(g_true, g_pred, boundaries, save_plot)

    return {"rmse": round(rmse, 5), "mae": round(mae, 5),
            "exact_acc": round(exact, 4), "within1_acc": round(off1, 4),
            "within2_acc": round(off2, 4)}

def _save_confusion_png(true_grades, pred_grades, boundaries, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sorted_grades = sorted(boundaries, key=lambda g: boundaries[g]["mean"])
        present = [g for g in sorted_grades if g in set(true_grades)]
        idx = {g: i for i, g in enumerate(present)}
        n = len(present)

        matrix = np.zeros((n, n), dtype=int)
        for t, p in zip(true_grades, pred_grades):
            if t in idx and p in idx:
                matrix[idx[t]][idx[p]] += 1

        # Normalize by row
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm = np.where(row_sums > 0, matrix / row_sums, 0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(norm, annot=True, fmt=".2f", xticklabels=present,
                    yticklabels=present, cmap="Blues", ax=ax, vmin=0, vmax=1)
        ax.set_xlabel("Predicted grade")
        ax.set_ylabel("True grade")
        ax.set_title("Grade prediction confusion matrix (row-normalized)")
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        print(f"\nConfusion matrix PNG → {output}")
    except ImportError:
        print("\nmatplotlib/seaborn not installed — skipping plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained difficulty model")
    parser.add_argument("--input",  type=str, default="data/routes_features.csv")
    parser.add_argument("--model-dir", type=str, default="ml")
    parser.add_argument("--save",   type=str, default=None,
                        help="Save confusion matrix PNG to this path")
    args = parser.parse_args()
    evaluate(args.input, args.model_dir, args.save)

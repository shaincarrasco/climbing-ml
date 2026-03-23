"""
ml/feature_importance.py

Visualizes feature importances from the trained XGBoost difficulty model.
Reads ml/feature_importance.csv and prints a ranked bar chart to terminal.
Optionally saves a matplotlib PNG.

Usage:
    python3 ml/feature_importance.py
    python3 ml/feature_importance.py --top 20
    python3 ml/feature_importance.py --save ml/feature_importance.png
"""

import argparse
import pandas as pd
import os

def bar(value, max_value, width=40):
    filled = round((value / max_value) * width) if max_value > 0 else 0
    return "█" * filled

def print_importances(imp_path="ml/feature_importance.csv", top=20):
    df = pd.read_csv(imp_path).sort_values("importance", ascending=False).head(top)
    max_imp = df["importance"].max()

    print(f"\nTop {len(df)} features by XGBoost gain importance")
    print(f"  Model: ml/difficulty_model.xgb")
    print(f"  Source: {imp_path}")
    print("─" * 65)
    for _, row in df.iterrows():
        b = bar(row["importance"], max_imp)
        print(f"  {row['feature']:35s}  {row['importance']:.4f}  {b}")
    print("─" * 65)

    # Group by category
    categories = {
        "Board/angle":    ["board_angle_deg"],
        "Foot geometry":  ["foot_hold_count", "avg_foot_spread_cm", "foot_ratio",
                           "avg_hand_to_foot_cm", "avg_foot_hand_x_offset_cm",
                           "max_foot_hand_x_offset_cm", "hand_foot_ratio"],
        "Hold counts":    ["total_hold_count", "hand_hold_count", "moves_count"],
        "Reach":          ["avg_reach_cm", "max_reach_cm", "min_reach_cm",
                           "reach_std_cm", "reach_range_cm", "reach_cv"],
        "Span/position":  ["height_span_cm", "lateral_span_cm", "width_height_ratio",
                           "hold_density", "start_x_cm", "start_y_cm",
                           "finish_x_cm", "finish_y_cm", "start_height_pct",
                           "finish_height_pct", "centroid_x_cm", "centroid_y_cm",
                           "total_centroid_x_cm", "total_centroid_y_cm",
                           "hold_spread_x", "hold_spread_y"],
        "Direction":      ["avg_lateral_cm", "avg_vertical_cm", "max_lateral_cm",
                           "max_vertical_cm", "vertical_ratio", "lateral_ratio",
                           "direction_changes", "zigzag_ratio"],
        "Dynamic":        ["dyno_score", "dyno_flag", "max_reach_z"],
        "Hold types":     ["crimp_ratio", "sloper_ratio", "jug_ratio",
                           "pinch_ratio", "typed_ratio"],
    }

    all_imp = pd.read_csv(imp_path)
    print("\nImportance by category:")
    cat_totals = []
    for cat, cols in categories.items():
        total = all_imp[all_imp["feature"].isin(cols)]["importance"].sum()
        cat_totals.append((cat, total))
    cat_totals.sort(key=lambda x: x[1], reverse=True)
    max_cat = cat_totals[0][1]
    for cat, total in cat_totals:
        b = bar(total, max_cat, width=30)
        print(f"  {cat:18s}  {total:.4f}  {b}")

def save_plot(imp_path="ml/feature_importance.csv", output="ml/feature_importance.png", top=20):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_csv(imp_path).sort_values("importance").tail(top)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(df["feature"], df["importance"], color="#4C9BE8")
        ax.set_xlabel("XGBoost Gain Importance")
        ax.set_title(f"Top {top} Features — Kilter Board Difficulty Model")
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        print(f"\nPlot saved → {output}")
    except ImportError:
        print("\nmatplotlib not installed — skipping plot. Run: pip install matplotlib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize feature importances")
    parser.add_argument("--input", type=str, default="ml/feature_importance.csv")
    parser.add_argument("--top",   type=int, default=20)
    parser.add_argument("--save",  type=str, default=None,
                        help="Save bar chart PNG to this path")
    args = parser.parse_args()

    print_importances(args.input, args.top)
    if args.save:
        save_plot(args.input, args.save, args.top)

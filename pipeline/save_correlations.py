"""
pipeline/save_correlations.py
------------------------------
Computes Pearson correlations between every pose metric and difficulty_score,
then writes data/pose_correlations.json.

Run automatically by retrain.py after pose_feature_extraction.py.
Also usable standalone:
    python3 pipeline/save_correlations.py
    python3 pipeline/save_correlations.py --print
"""

import json
import os
import sys
import argparse
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent

# ── Metrics to correlate ──────────────────────────────────────────────────────

METRICS = [
    "hip_angle_deg",
    "tension_score",
    "com_height_norm",
    "left_arm_reach_norm",
    "right_arm_reach_norm",
    "hip_spread_deg",
    "elbow_l_deg",
    "elbow_r_deg",
    "knee_l_deg",
    "knee_r_deg",
    "com_velocity",
    "shoulder_rot_deg",
]

# ── DB connection ──────────────────────────────────────────────────────────────

def _get_conn():
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv(_BASE_DIR / ".env")
    return psycopg2.connect(os.getenv("DATABASE_URL",
                                       "postgresql://localhost/climbing_platform"))


# ── Correlation query ──────────────────────────────────────────────────────────

def compute_correlations(conn) -> dict:
    """Return {metric: pearson_r} for all METRICS vs difficulty_score."""
    cur = conn.cursor()
    result = {}
    for metric in METRICS:
        cur.execute(f"""
            SELECT CORR(pf.{metric}, br.difficulty_score)
            FROM pose_frames pf
            JOIN board_routes br ON UPPER(br.external_id) = UPPER(pf.climb_uuid)
            WHERE pf.{metric} IS NOT NULL AND br.difficulty_score IS NOT NULL
        """)
        val = cur.fetchone()[0]
        result[metric] = round(float(val), 4) if val is not None else None
    cur.close()
    return result


def compute_grade_benchmarks(conn) -> list:
    """Per-grade median/mean pose stats for tracking progression."""
    cur = conn.cursor()
    cur.execute("""
        SELECT
            br.community_grade,
            MIN(br.difficulty_score)                                               AS diff_score,
            ROUND(AVG(pf.hip_angle_deg)::numeric, 1)                              AS avg_hip_angle,
            ROUND(AVG(pf.tension_score)::numeric, 3)                              AS avg_tension,
            ROUND(AVG(pf.com_height_norm)::numeric, 3)                            AS avg_com_height,
            ROUND(AVG((pf.left_arm_reach_norm+pf.right_arm_reach_norm)/2)::numeric,3) AS avg_reach,
            ROUND(AVG(pf.hip_spread_deg)::numeric, 1)                             AS avg_hip_spread,
            ROUND(AVG(pf.elbow_l_deg)::numeric, 1)                                AS avg_elbow_l,
            ROUND(AVG(pf.elbow_r_deg)::numeric, 1)                                AS avg_elbow_r,
            ROUND(AVG(pf.knee_l_deg)::numeric, 1)                                 AS avg_knee,
            ROUND(AVG(pf.com_velocity)::numeric, 4)                               AS avg_com_vel,
            COUNT(DISTINCT pf.attempt_id)                                          AS videos,
            COUNT(*)                                                               AS frames
        FROM pose_frames pf
        JOIN board_routes br ON UPPER(br.external_id) = UPPER(pf.climb_uuid)
        WHERE br.community_grade IS NOT NULL
          AND br.difficulty_score IS NOT NULL
        GROUP BY br.community_grade
        HAVING COUNT(*) >= 20
        ORDER BY MIN(br.difficulty_score)
    """)
    rows = cur.fetchall()
    col_names = [d[0] for d in cur.description]
    benchmarks = []
    for row in rows:
        entry = dict(zip(col_names, row))
        for k, v in entry.items():
            if hasattr(v, '__float__'):
                entry[k] = float(v)
        benchmarks.append(entry)
    cur.close()
    return benchmarks


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compute pose-metric correlations and save to data/pose_correlations.json")
    ap.add_argument("--print", action="store_true", help="Print results to stdout")
    args = ap.parse_args()

    try:
        conn = _get_conn()
    except Exception as e:
        print(f"  [correlations] DB connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("  [correlations] Computing Pearson correlations vs difficulty_score …")
    correlations = compute_correlations(conn)

    print("  [correlations] Computing per-grade benchmarks …")
    benchmarks = compute_grade_benchmarks(conn)
    conn.close()

    # Sort by absolute correlation strength — strongest predictors first
    sorted_corr = sorted(
        [{"metric": k, "r": v, "abs_r": abs(v) if v is not None else 0.0}
         for k, v in correlations.items()],
        key=lambda x: x["abs_r"],
        reverse=True,
    )

    payload = {
        "correlations":     sorted_corr,
        "grade_benchmarks": benchmarks,
        "note": "r = Pearson with difficulty_score. |r|>0.3 = meaningful signal.",
    }

    out_path = _BASE_DIR / "data" / "pose_correlations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  [correlations] Saved → {out_path.relative_to(_BASE_DIR)}")
    print(f"  [correlations] {len(sorted_corr)} metrics  |  {len(benchmarks)} grades with benchmark data")

    if args.print or True:  # always print a quick summary
        print("\n  Top correlations with difficulty_score:")
        for entry in sorted_corr[:6]:
            r = entry["r"]
            bar = "█" * int(abs(r) * 20) if r is not None else ""
            sign = "+" if (r or 0) >= 0 else "-"
            print(f"    {entry['metric']:<25}  {sign}{abs(r or 0):.4f}  {bar}")


if __name__ == "__main__":
    main()

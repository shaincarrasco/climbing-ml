"""
pipeline/pose_feature_extraction.py
------------------------------------
Aggregates raw pose_frames (per-frame) into one row per route with
~35 ML-ready features. Joins with kilter.db for difficulty labels.

Output CSV columns
------------------
  climb_uuid            — Kilter route identifier (join key)
  route_name            — human-readable name
  setter                — setter username
  difficulty_average    — Kilter raw difficulty score (e.g. 23.0 = ~V4)
  quality_average       — community quality rating
  ascensionist_count    — number of sends (data quality proxy)
  video_count           — beta videos contributing to this row
  frame_count           — total pose frames

  Compression / hip:
    pose_avg_hip_angle      avg shoulder-hip-knee angle
    pose_min_hip_angle      peak compression moment
    pose_hip_angle_range    max - min (variability = dynamic hip use)
    pose_avg_hip_spread     avg leg split angle (drop-knees / flags)
    pose_max_hip_spread     peak leg split

  Arm mechanics:
    pose_avg_elbow_l/r      avg elbow bend (lower = more bent)
    pose_min_elbow_l/r      max elbow bend (hardest pull moment)
    pose_pct_straight_l/r   % frames with straight arm (resting on arms)
    pose_elbow_asymmetry    abs(avg_l - avg_r) — dominant arm bias
    pose_avg_shoulder_l/r   avg shoulder flexion vs torso

  Leg / foot mechanics:
    pose_avg_knee_l/r       avg knee angle
    pose_min_knee           peak knee bend (high-step indicator)
    pose_avg_foot_hand_diff avg foot height relative to hands
    pose_pct_high_feet      % frames feet are above hands

  Body position:
    pose_avg_tension        avg body tightness (0–1)
    pose_p90_tension        sustained high tension
    pose_com_travel         max - min COM height (how far up wall)
    pose_avg_com_to_hands   avg normalised COM-to-hands distance

  Movement dynamics (velocities):
    pose_avg_com_vel        avg COM speed (low = static style)
    pose_peak_com_vel       peak COM speed (dynamic moves)
    pose_avg_hand_vel       avg of L+R hand speed
    pose_peak_hand_vel      peak hand speed (lunge / slap detection)
    pose_avg_hip_ang_vel    avg hip angular velocity
    pose_peak_elbow_ang_vel peak elbow angular velocity (explosive pulls)

  Symmetry:
    pose_reach_asymmetry    abs(avg_l_reach - avg_r_reach)

Usage:
    python pipeline/pose_feature_extraction.py
    python pipeline/pose_feature_extraction.py --output data/pose_features.csv
    python pipeline/pose_feature_extraction.py --min-videos 2 --stats
    python pipeline/pose_feature_extraction.py --merge   # merge with routes_features.csv
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

_BASE_DIR   = Path(__file__).resolve().parent.parent
KILTER_DB   = _BASE_DIR / "kilter.db"
DATA_DIR    = _BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "pose_features.csv"
ROUTES_CSV  = DATA_DIR / "routes_features.csv"


# ── DB connections ────────────────────────────────────────────────────────────
def get_pg():
    load_dotenv()
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME",     "climbing_platform"),
        user=os.getenv("DB_USER",       "shaincarrasco"),
        password=os.getenv("DB_PASSWORD", "") or None,
        host=os.getenv("DB_HOST",       "localhost"),
        port=os.getenv("DB_PORT",       "5432"),
    )


def get_sqlite():
    return sqlite3.connect(KILTER_DB)


# ── Pull raw pose_frames and aggregate ────────────────────────────────────────
POSE_AGG_QUERY = """
SELECT
    climb_uuid,
    COUNT(DISTINCT attempt_id)                                      AS video_count,
    COUNT(*)                                                        AS frame_count,

    -- Hip / compression
    AVG(hip_angle_deg)                                              AS pose_avg_hip_angle,
    MIN(hip_angle_deg)                                              AS pose_min_hip_angle,
    MAX(hip_angle_deg) - MIN(hip_angle_deg)                        AS pose_hip_angle_range,
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY hip_angle_deg)     AS pose_p10_hip_angle,

    -- Hip spread (drop-knee / flag)
    AVG(hip_spread_deg)                                             AS pose_avg_hip_spread,
    MAX(hip_spread_deg)                                             AS pose_max_hip_spread,

    -- Elbow angles
    AVG(elbow_l_deg)                                                AS pose_avg_elbow_l,
    AVG(elbow_r_deg)                                                AS pose_avg_elbow_r,
    MIN(elbow_l_deg)                                                AS pose_min_elbow_l,
    MIN(elbow_r_deg)                                                AS pose_min_elbow_r,
    ABS(AVG(elbow_l_deg) - AVG(elbow_r_deg))                       AS pose_elbow_asymmetry,

    -- Straight arm percentages
    AVG(CASE WHEN is_straight_arm_l THEN 1.0 ELSE 0.0 END)        AS pose_pct_straight_l,
    AVG(CASE WHEN is_straight_arm_r THEN 1.0 ELSE 0.0 END)        AS pose_pct_straight_r,

    -- Shoulder flexion
    AVG(shoulder_l_deg)                                             AS pose_avg_shoulder_l,
    AVG(shoulder_r_deg)                                             AS pose_avg_shoulder_r,

    -- Knee angles
    AVG(knee_l_deg)                                                 AS pose_avg_knee_l,
    AVG(knee_r_deg)                                                 AS pose_avg_knee_r,
    MIN(LEAST(knee_l_deg, knee_r_deg))                             AS pose_min_knee,

    -- Foot-hand height
    AVG(foot_hand_height_diff)                                      AS pose_avg_foot_hand_diff,
    AVG(CASE WHEN foot_hand_height_diff > 0 THEN 1.0 ELSE 0.0 END) AS pose_pct_high_feet,

    -- Tension / body control
    AVG(tension_score)                                              AS pose_avg_tension,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY tension_score)     AS pose_p90_tension,

    -- COM position and travel
    MAX(com_height_norm) - MIN(com_height_norm)                    AS pose_com_travel,
    AVG(com_to_hands_dist)                                          AS pose_avg_com_to_hands,

    -- Reach asymmetry
    ABS(AVG(left_arm_reach_norm) - AVG(right_arm_reach_norm))     AS pose_reach_asymmetry,

    -- Velocities (only from rows where velocity was computed)
    AVG(com_velocity)                                               AS pose_avg_com_vel,
    MAX(com_velocity)                                               AS pose_peak_com_vel,
    AVG((hand_l_velocity + hand_r_velocity) / 2.0)                AS pose_avg_hand_vel,
    MAX(GREATEST(hand_l_velocity, hand_r_velocity))                AS pose_peak_hand_vel,
    AVG(hip_ang_vel)                                                AS pose_avg_hip_ang_vel,
    MAX(GREATEST(elbow_l_ang_vel, elbow_r_ang_vel))                AS pose_peak_elbow_ang_vel

FROM pose_frames
WHERE climb_uuid IS NOT NULL
  AND hip_angle_deg IS NOT NULL
  AND elbow_l_deg IS NOT NULL
GROUP BY climb_uuid
"""


def extract_pose_features(min_videos: int = 1) -> pd.DataFrame:
    """Aggregate pose_frames per climb_uuid into one-row-per-route features."""
    conn = get_pg()
    df = pd.read_sql(POSE_AGG_QUERY, conn)
    conn.close()

    df = df[df["video_count"] >= min_videos].copy()
    print(f"  pose_frames aggregated: {len(df)} routes ({df['frame_count'].sum():,} total frames)")
    return df


# ── Join difficulty labels from kilter.db ─────────────────────────────────────
def add_difficulty_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Join kilter.db climb_stats + climbs for difficulty and grade labels."""
    uuids = df["climb_uuid"].tolist()
    placeholders = ",".join("?" * len(uuids))

    con = get_sqlite()
    labels = pd.read_sql(f"""
        SELECT
            cs.climb_uuid,
            c.name              AS route_name,
            c.setter_username   AS setter,
            cs.difficulty_average,
            cs.quality_average,
            cs.ascensionist_count
        FROM climb_stats cs
        JOIN climbs c ON c.uuid = cs.climb_uuid
        WHERE cs.climb_uuid IN ({placeholders})
        GROUP BY cs.climb_uuid
        HAVING difficulty_average = MAX(cs.difficulty_average)
    """, con, params=uuids)
    con.close()

    merged = df.merge(labels, on="climb_uuid", how="left")
    matched = merged["difficulty_average"].notna().sum()
    print(f"  Difficulty labels matched: {matched} / {len(merged)} routes")
    return merged


# ── Optionally merge with existing routes_features.csv ───────────────────────
def merge_with_routes_features(pose_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join pose features with the hold-geometry features from routes_features.csv.
    Uses UPPER(climb_uuid) = UPPER(external_id) via board_routes in PostgreSQL.
    """
    if not ROUTES_CSV.exists():
        print(f"  routes_features.csv not found at {ROUTES_CSV} — skipping merge")
        return pose_df

    routes = pd.read_csv(ROUTES_CSV)
    print(f"  routes_features.csv: {len(routes):,} rows")

    # Get climb_uuid → route_id mapping from PostgreSQL
    conn = get_pg()
    mapping = pd.read_sql("""
        SELECT UPPER(external_id) AS climb_uuid, id::text AS route_id,
               community_grade, difficulty_score
        FROM board_routes
        WHERE source = 'kilter' AND external_id IS NOT NULL
    """, conn)
    conn.close()

    pose_df["climb_uuid_upper"] = pose_df["climb_uuid"].str.upper()
    pose_with_id = pose_df.merge(
        mapping, left_on="climb_uuid_upper", right_on="climb_uuid", how="left"
    ).drop(columns=["climb_uuid_upper", "climb_uuid_y"], errors="ignore")
    pose_with_id = pose_with_id.rename(columns={"climb_uuid_x": "climb_uuid"})

    matched = pose_with_id["route_id"].notna().sum()
    print(f"  Mapped to board_routes route_id: {matched} / {len(pose_with_id)}")

    if matched == 0:
        print("  No route_id matches found — outputting pose features only")
        return pose_df

    merged = routes.merge(pose_with_id, on="route_id", how="inner")
    print(f"  Combined rows: {len(merged)} (routes with both geometry + pose features)")
    return merged


# ── Stats summary ─────────────────────────────────────────────────────────────
def print_stats():
    conn = get_pg()
    cur  = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(DISTINCT climb_uuid) FILTER (WHERE climb_uuid IS NOT NULL) AS routes_with_pose,
            COUNT(DISTINCT attempt_id)                                        AS total_videos,
            COUNT(*)                                                          AS total_frames,
            ROUND(AVG(tension_score)::numeric, 3)                            AS global_avg_tension,
            ROUND(AVG(hip_angle_deg)::numeric, 1)                            AS global_avg_hip,
            ROUND(AVG(elbow_l_deg)::numeric, 1)                              AS global_avg_elbow
        FROM pose_frames
        WHERE hip_angle_deg IS NOT NULL
    """)
    row = cur.fetchone()

    cur.execute("""
        SELECT climb_uuid, COUNT(*) AS frames,
               ROUND(AVG(tension_score)::numeric,3) AS tension,
               ROUND(AVG(hip_angle_deg)::numeric,1) AS hip
        FROM pose_frames
        WHERE climb_uuid IS NOT NULL AND hip_angle_deg IS NOT NULL
        GROUP BY climb_uuid
        ORDER BY frames DESC
    """)
    per_route = cur.fetchall()
    cur.close()
    conn.close()

    print(f"\nPose data summary")
    print(f"  Routes with pose data : {row[0]}")
    print(f"  Total beta videos     : {row[1]}")
    print(f"  Total frames          : {row[2]:,}")
    print(f"  Global avg tension    : {row[3]}")
    print(f"  Global avg hip angle  : {row[4]}°")
    print(f"  Global avg elbow      : {row[5]}°")

    if per_route:
        print(f"\n  {'climb_uuid':<34} {'frames':>6} {'tension':>8} {'hip':>6}")
        print(f"  {'─'*60}")
        for r in per_route:
            print(f"  {r[0]:<34} {r[1]:>6,} {str(r[2]):>8} {str(r[3]):>6}°")

    # Check how many routes could join to difficulty labels
    con = get_sqlite()
    cur2 = con.cursor()
    if per_route:
        uuids = [r[0] for r in per_route]
        placeholders = ",".join("?" * len(uuids))
        cur2.execute(
            f"SELECT COUNT(DISTINCT climb_uuid) FROM climb_stats WHERE climb_uuid IN ({placeholders})",
            uuids
        )
        rated = cur2.fetchone()[0]
        print(f"\n  Routes matchable to difficulty labels: {rated} / {len(uuids)}")
    con.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate pose_frames → one-row-per-route ML features"
    )
    parser.add_argument("--output",     default=str(OUTPUT_PATH),
                        help=f"Output CSV path (default: {OUTPUT_PATH})")
    parser.add_argument("--min-videos", type=int, default=1,
                        help="Minimum beta videos required per route (default: 1)")
    parser.add_argument("--merge",      action="store_true",
                        help="Also merge with routes_features.csv for combined model training")
    parser.add_argument("--stats",      action="store_true",
                        help="Print data coverage stats and exit")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    print(f"\nExtracting pose features...")
    df = extract_pose_features(min_videos=args.min_videos)

    if df.empty:
        print("No pose data found. Run the beta scraper first.")
        return

    print(f"\nJoining difficulty labels...")
    df = add_difficulty_labels(df)

    if args.merge:
        print(f"\nMerging with routes_features.csv...")
        df = merge_with_routes_features(df)

    # Round floats for readability
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].round(4)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nOutput: {args.output}")
    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {len(df.columns)}")
    print(f"\nColumns:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col:<35} {non_null}/{len(df)} non-null")


if __name__ == "__main__":
    main()

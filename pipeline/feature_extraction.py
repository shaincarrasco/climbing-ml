"""
pipeline/feature_extraction.py

Full feature extraction from climbing_platform PostgreSQL database.
Outputs: data/routes_features.csv

FEATURES (50+):
  HOLD COUNTS & RATIOS
    hand_hold_count, foot_hold_count, total_hold_count, moves_count
    foot_ratio, hand_foot_ratio

  REACH GEOMETRY (hand-to-hand)
    avg_reach_cm, max_reach_cm, min_reach_cm
    reach_std_cm, reach_range_cm, reach_cv

  MOVE DIRECTION
    avg_lateral_cm, avg_vertical_cm
    max_lateral_cm, max_vertical_cm
    vertical_ratio, lateral_ratio

  FOOT HOLD GEOMETRY
    avg_foot_spread_cm        mean distance between consecutive foot holds
    avg_hand_to_foot_cm       mean distance from each hand hold to nearest foot hold
    avg_foot_hand_x_offset_cm mean horizontal offset: foot vs nearest hand hold above
                              (large = feet wide of hands = flagging/drop-knee required)
    max_foot_hand_x_offset_cm worst-case horizontal offset

  SPATIAL SPAN
    height_span_cm, lateral_span_cm, width_height_ratio, hold_density

  SEQUENCE COMPLEXITY
    direction_changes, zigzag_ratio
    reach_cv (coefficient of variation of reaches)

  DYNAMIC SCORE
    dyno_score        angle-weighted ratio of max_reach to avg_reach
                      higher = more likely to contain a dynamic move
    dyno_flag         1 if dyno_score suggests a likely dyno, 0 otherwise
    max_reach_z       how many std devs the max reach is above the mean
                      (detects single outlier crux move)

  BOARD POSITION
    start_x_cm, start_y_cm, finish_x_cm, finish_y_cm
    start_height_pct, finish_height_pct
    centroid_x_cm, centroid_y_cm
    total_centroid_x_cm, total_centroid_y_cm
    hold_spread_x, hold_spread_y

  GRADE CONTEXT
    grade_rank_at_angle   percentile rank within same angle (0=easiest 1=hardest)

  HOLD TYPE RATIOS (from Climbology labels)
    crimp_ratio       fraction of hand holds that are crimps
    sloper_ratio      fraction of hand holds that are slopers
    jug_ratio         fraction of hand holds that are jugs
    pinch_ratio       fraction of hand holds that are pinches
    typed_ratio       fraction of all holds with a known hold_type label

  TARGETS & WEIGHTS
    community_grade, difficulty_score, send_count, board_angle_deg

Usage:
    python3 pipeline/feature_extraction.py
    python3 pipeline/feature_extraction.py --angle 40
    python3 pipeline/feature_extraction.py --min-sends 5
    python3 pipeline/feature_extraction.py --limit 10000 --output data/sample.csv
"""

import os
import math
import argparse
import psycopg2
import psycopg2.extras
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BOARD_HEIGHT_CM = 140.0
BOARD_WIDTH_CM  = 140.0

# Angle-based dyno threshold multiplier.
# Steeper boards make the same reach feel more dynamic.
# At 0 degrees (slab) a 2x reach is just a long move.
# At 70 degrees (roof) a 1.6x reach is likely a dyno.
DYNO_ANGLE_WEIGHTS = {
    0:  2.8,
    20: 2.6,
    35: 2.3,
    40: 2.2,
    45: 2.1,
    50: 2.0,
    55: 1.95,
    60: 1.9,
    65: 1.85,
    70: 1.8,
}

def get_dyno_threshold(angle):
    """Return the reach ratio threshold above which a move is likely dynamic."""
    angles = sorted(DYNO_ANGLE_WEIGHTS.keys())
    if angle <= angles[0]:
        return DYNO_ANGLE_WEIGHTS[angles[0]]
    if angle >= angles[-1]:
        return DYNO_ANGLE_WEIGHTS[angles[-1]]
    for i in range(len(angles) - 1):
        if angles[i] <= angle <= angles[i+1]:
            lo, hi = angles[i], angles[i+1]
            t = (angle - lo) / (hi - lo)
            return DYNO_ANGLE_WEIGHTS[lo] + t * (DYNO_ANGLE_WEIGHTS[hi] - DYNO_ANGLE_WEIGHTS[lo])
    return 1.8

def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "climbing_platform"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD", ""),
    )

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def safe_mean(values):
    return sum(values) / len(values) if values else 0.0

def safe_std(values):
    if len(values) < 2:
        return 0.0
    mean = safe_mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

def nearest_hold(hold, candidates, filter_fn=None):
    """Return the nearest hold from candidates to the given hold."""
    pool = [h for h in candidates if filter_fn(h)] if filter_fn else candidates
    if not pool:
        return None
    hx = hold.get("position_x_cm") or 0
    hy = hold.get("position_y_cm") or 0
    return min(pool, key=lambda h: dist(hx, hy, h.get("position_x_cm") or 0, h.get("position_y_cm") or 0))

def compute_features(route_id, angle, grade, difficulty_score, send_count, holds):
    if not holds or len(holds) < 2:
        return None

    hand_holds = [h for h in holds if h["role"] in ("start", "hand", "finish")]
    foot_holds  = [h for h in holds if h["role"] == "foot"]

    if len(hand_holds) < 2:
        return None

    # Sort hand holds by sequence then y position
    sequenced = [h for h in hand_holds if h.get("hand_sequence") is not None]
    if len(sequenced) >= 2:
        hand_seq = sorted(sequenced, key=lambda h: h["hand_sequence"])
    else:
        hand_seq = sorted(hand_holds, key=lambda h: h.get("position_y_cm") or 0)

    # ── Hand-to-hand reach geometry ───────────────────────────────────────────
    reaches, laterals, verticals = [], [], []
    move_angles = []
    direction_changes = 0
    prev_lat_sign = None

    for i in range(1, len(hand_seq)):
        h1, h2 = hand_seq[i-1], hand_seq[i]
        x1 = h1.get("position_x_cm") or 0
        y1 = h1.get("position_y_cm") or 0
        x2 = h2.get("position_x_cm") or 0
        y2 = h2.get("position_y_cm") or 0

        dx = abs(x2 - x1)
        dy = y2 - y1

        reaches.append(dist(x1, y1, x2, y2))
        laterals.append(dx)
        verticals.append(abs(dy))

        import math as _math
        move_angles.append(abs(_math.degrees(_math.atan2(dx, max(dy, 0.001)))))

        lat_sign = 1 if (x2 - x1) > 0 else (-1 if (x2 - x1) < 0 else 0)
        if prev_lat_sign is not None and lat_sign != 0 and lat_sign != prev_lat_sign:
            direction_changes += 1
        if lat_sign != 0:
            prev_lat_sign = lat_sign

    if not reaches:
        return None

    avg_reach = safe_mean(reaches)
    reach_std = safe_std(reaches)
    reach_cv  = reach_std / avg_reach if avg_reach > 0 else 0
    moves     = len(reaches)

    # ── Foot hold geometry ────────────────────────────────────────────────────

    # Mean distance between consecutive foot holds
    foot_spreads = []
    if len(foot_holds) >= 2:
        foot_sorted = sorted(foot_holds, key=lambda h: h.get("position_y_cm") or 0)
        for i in range(1, len(foot_sorted)):
            f1, f2 = foot_sorted[i-1], foot_sorted[i]
            foot_spreads.append(dist(
                f1.get("position_x_cm") or 0, f1.get("position_y_cm") or 0,
                f2.get("position_x_cm") or 0, f2.get("position_y_cm") or 0
            ))
    avg_foot_spread = safe_mean(foot_spreads)

    # Mean distance from each hand hold to nearest foot hold
    hand_to_foot_dists = []
    foot_hand_x_offsets = []

    for hh in hand_seq:
        hx = hh.get("position_x_cm") or 0
        hy = hh.get("position_y_cm") or 0

        if foot_holds:
            nearest_foot = nearest_hold(hh, foot_holds)
            if nearest_foot:
                fx = nearest_foot.get("position_x_cm") or 0
                fy = nearest_foot.get("position_y_cm") or 0
                hand_to_foot_dists.append(dist(hx, hy, fx, fy))

        # Horizontal offset: find nearest foot hold BELOW this hand hold
        feet_below = [f for f in foot_holds
                      if (f.get("position_y_cm") or 0) < hy]
        if feet_below:
            nearest_below = nearest_hold(hh, feet_below)
            if nearest_below:
                fx = nearest_below.get("position_x_cm") or 0
                foot_hand_x_offsets.append(abs(hx - fx))

    avg_hand_to_foot    = safe_mean(hand_to_foot_dists)
    avg_foot_hand_x_off = safe_mean(foot_hand_x_offsets)
    max_foot_hand_x_off = max(foot_hand_x_offsets) if foot_hand_x_offsets else 0

    # ── Spatial span ──────────────────────────────────────────────────────────
    all_xs = [h.get("position_x_cm") or 0 for h in hand_seq]
    all_ys = [h.get("position_y_cm") or 0 for h in hand_seq]
    height_span  = max(all_ys) - min(all_ys)
    lateral_span = max(all_xs) - min(all_xs)

    all_hx = [h.get("position_x_cm") or 0 for h in holds]
    all_hy = [h.get("position_y_cm") or 0 for h in holds]

    # ── Dynamic score ─────────────────────────────────────────────────────────
    # How many std devs above mean is the max reach?
    max_reach_z = (max(reaches) - avg_reach) / reach_std if reach_std > 0 else 0

    # Angle-weighted ratio of max reach to avg reach
    dyno_threshold = get_dyno_threshold(angle)
    dyno_ratio     = max(reaches) / avg_reach if avg_reach > 0 else 0
    dyno_score     = round(dyno_ratio / dyno_threshold, 4)
    dyno_flag      = 1 if (dyno_ratio >= dyno_threshold and max_reach_z >= 2.0 and max(reaches) > 150) else 0

    # ── Hand-foot ratio ───────────────────────────────────────────────────────
    n_hand = len(hand_holds)
    n_foot = len(foot_holds)
    hand_foot_ratio = round(n_hand / n_foot, 4) if n_foot > 0 else float(n_hand)

    # ── Hold type ratios (Climbology labels) ──────────────────────────────────
    hand_typed = [h for h in hand_holds if h.get("hold_type")]
    n_typed    = len(hand_typed)
    def _type_ratio(t):
        return round(sum(1 for h in hand_typed if h["hold_type"] == t) / n_typed, 4) if n_typed > 0 else 0.0
    crimp_ratio  = _type_ratio("crimp")
    sloper_ratio = _type_ratio("sloper")
    jug_ratio    = _type_ratio("jug")
    pinch_ratio  = _type_ratio("pinch")
    all_typed    = [h for h in holds if h.get("hold_type")]
    typed_ratio  = round(len(all_typed) / len(holds), 4) if holds else 0.0

    avg_v = avg_reach if avg_reach > 0 else 1

    return {
        # identifiers & targets
        "route_id":                   str(route_id),
        "board_angle_deg":            angle,
        "community_grade":            grade,
        "difficulty_score":           difficulty_score,
        "send_count":                 send_count,

        # hold counts
        "hand_hold_count":            n_hand,
        "foot_hold_count":            n_foot,
        "total_hold_count":           len(holds),
        "moves_count":                moves,
        "foot_ratio":                 round(n_foot / len(holds), 4) if holds else 0,
        "hand_foot_ratio":            hand_foot_ratio,

        # reach geometry
        "avg_reach_cm":               round(avg_reach, 2),
        "max_reach_cm":               round(max(reaches), 2),
        "min_reach_cm":               round(min(reaches), 2),
        "reach_std_cm":               round(reach_std, 2),
        "reach_range_cm":             round(max(reaches) - min(reaches), 2),
        "reach_cv":                   round(reach_cv, 4),

        # move direction
        "avg_lateral_cm":             round(safe_mean(laterals), 2),
        "avg_vertical_cm":            round(safe_mean(verticals), 2),
        "max_lateral_cm":             round(max(laterals), 2),
        "max_vertical_cm":            round(max(verticals), 2),
        "vertical_ratio":             round(safe_mean(verticals) / avg_v, 4),
        "lateral_ratio":              round(safe_mean(laterals) / avg_v, 4),

        # foot geometry
        "avg_foot_spread_cm":         round(avg_foot_spread, 2),
        "avg_hand_to_foot_cm":        round(avg_hand_to_foot, 2),
        "avg_foot_hand_x_offset_cm":  round(avg_foot_hand_x_off, 2),
        "max_foot_hand_x_offset_cm":  round(max_foot_hand_x_off, 2),

        # spatial span
        "height_span_cm":             round(height_span, 2),
        "lateral_span_cm":            round(lateral_span, 2),
        "width_height_ratio":         round(lateral_span / height_span, 4) if height_span > 0 else 0,
        "hold_density":               round(len(holds) / (height_span * lateral_span) * 100, 6) if height_span > 0 and lateral_span > 0 else 0,

        # sequence complexity
        "direction_changes":          direction_changes,
        "zigzag_ratio":               round(direction_changes / moves, 4) if moves > 0 else 0,
        "avg_move_angle_deg":         round(safe_mean(move_angles), 2),
        "max_move_angle_deg":         round(max(move_angles), 2) if move_angles else 0,
        "move_angle_std":             round(safe_std(move_angles), 2),
        "crux_reach_ratio":           round(max(reaches) / (safe_mean(reaches) or 1), 4),
        "path_linearity":             round(
            dist(hand_seq[0].get("position_x_cm") or 0, hand_seq[0].get("position_y_cm") or 0,
                 hand_seq[-1].get("position_x_cm") or 0, hand_seq[-1].get("position_y_cm") or 0)
            / (sum(reaches) or 1), 4
        ),

        # dynamic score
        "dyno_score":                 dyno_score,
        "dyno_flag":                  dyno_flag,
        "max_reach_z":                round(max_reach_z, 4),

        # board position
        "start_x_cm":                 round((hand_seq[0].get("position_x_cm") or 0), 2),
        "start_y_cm":                 round((hand_seq[0].get("position_y_cm") or 0), 2),
        "finish_x_cm":                round((hand_seq[-1].get("position_x_cm") or 0), 2),
        "finish_y_cm":                round((hand_seq[-1].get("position_y_cm") or 0), 2),
        "start_height_pct":           round((hand_seq[0].get("position_y_cm") or 0) / BOARD_HEIGHT_CM, 4),
        "finish_height_pct":          round((hand_seq[-1].get("position_y_cm") or 0) / BOARD_HEIGHT_CM, 4),
        "centroid_x_cm":              round(safe_mean(all_xs), 2),
        "centroid_y_cm":              round(safe_mean(all_ys), 2),
        "total_centroid_x_cm":        round(safe_mean(all_hx), 2),
        "total_centroid_y_cm":        round(safe_mean(all_hy), 2),
        "hold_spread_x":              round(safe_std(all_xs), 2),
        "hold_spread_y":              round(safe_std(all_ys), 2),

        # hold type ratios
        "crimp_ratio":                crimp_ratio,
        "sloper_ratio":               sloper_ratio,
        "jug_ratio":                  jug_ratio,
        "pinch_ratio":                pinch_ratio,
        "typed_ratio":                typed_ratio,
    }

def extract(angle=None, min_sends=0, limit=None,
            output_path="data/routes_features.csv", batch_size=1000):

    conn = get_conn()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    angle_filter = f"AND board_angle_deg = {angle}" if angle else ""
    send_filter  = f"AND send_count >= {min_sends}" if min_sends > 0 else ""
    limit_clause = f"LIMIT {limit}" if limit else ""

    cur.execute(f"""
        SELECT id, board_angle_deg, community_grade, difficulty_score, send_count
        FROM board_routes
        WHERE source = 'kilter'
          AND community_grade IS NOT NULL
          AND difficulty_score IS NOT NULL
          {angle_filter}
          {send_filter}
        ORDER BY send_count DESC NULLS LAST
        {limit_clause}
    """)
    routes = cur.fetchall()
    print(f"\nFetching routes from PostgreSQL...")
    print(f"  {len(routes):,} routes to process")

    if not routes:
        print("No routes found.")
        return

    route_index = {str(r["id"]): dict(r) for r in routes}
    route_ids   = list(route_index.keys())
    all_features = []
    skipped = 0

    for batch_start in range(0, len(route_ids), batch_size):
        batch = route_ids[batch_start : batch_start + batch_size]

        cur.execute("""
            SELECT rh.route_id, rh.role,
                   rh.position_x_cm, rh.position_y_cm,
                   rh.hand_sequence, rh.hold_type
            FROM route_holds rh
            WHERE rh.route_id = ANY(%s::uuid[])
        """, (batch,))

        holds_by_route = {}
        for h in cur.fetchall():
            rid = str(h["route_id"])
            holds_by_route.setdefault(rid, []).append(dict(h))

        for rid in batch:
            r     = route_index[rid]
            feats = compute_features(
                route_id         = rid,
                angle            = r["board_angle_deg"],
                grade            = r["community_grade"],
                difficulty_score = r["difficulty_score"],
                send_count       = r["send_count"],
                holds            = holds_by_route.get(rid, [])
            )
            if feats:
                all_features.append(feats)
            else:
                skipped += 1

        done = min(batch_start + batch_size, len(route_ids))
        print(f"  {done:,} / {len(route_ids):,} — {len(all_features):,} extracted, {skipped} skipped")

    cur.close()
    conn.close()

    if not all_features:
        print("No features extracted.")
        return

    df = pd.DataFrame(all_features)
    df["grade_rank_at_angle"] = df.groupby("board_angle_deg")["difficulty_score"].rank(pct=True).round(4)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'─'*60}")
    print(f"Complete: {len(df):,} rows  |  {len(df.columns)} features  |  {skipped} skipped")
    print(f"Output:   {output_path}")

    print(f"\nGrade distribution:")
    for grade, count in df["community_grade"].value_counts().sort_index().items():
        bar = "█" * min(50, count // 500)
        print(f"  {grade:>4}  {count:>7,}  {bar}")

    print(f"\nDynamic move stats:")
    print(f"  Routes flagged as likely dyno: {df['dyno_flag'].sum():,} ({df['dyno_flag'].mean()*100:.1f}%)")
    print(f"  Avg dyno_score: {df['dyno_score'].mean():.3f}")
    print(f"  Max dyno_score: {df['dyno_score'].max():.3f}")

    print(f"\nFoot geometry stats:")
    print(f"  avg_foot_spread_cm mean:        {df['avg_foot_spread_cm'].mean():.1f}")
    print(f"  avg_foot_hand_x_offset_cm mean: {df['avg_foot_hand_x_offset_cm'].mean():.1f}")
    print(f"  Avg hand_foot_ratio:            {df['hand_foot_ratio'].mean():.2f}")

    print(f"\nKey feature stats:")
    cols = ["avg_reach_cm","max_reach_cm","height_span_cm","dyno_score","zigzag_ratio","board_angle_deg","difficulty_score"]
    print(df[cols].describe().round(2).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ML features from climbing_platform")
    parser.add_argument("--angle",      type=int, default=None)
    parser.add_argument("--min-sends",  type=int, default=0)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--output",     type=str, default="data/routes_features.csv")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    extract(angle=args.angle, min_sends=args.min_sends, limit=args.limit,
            output_path=args.output, batch_size=args.batch_size)

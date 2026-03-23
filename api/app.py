"""
api/app.py

Flask API for the Climbing Intelligence Platform.

Endpoints:
  GET  /api/board               All 692 Kilter Original holds with Climbology types + positions
  GET  /api/routes              Routes from DB (filterable by grade, angle, min_sends)
  GET  /api/route/<id>          Single route with all holds
  POST /api/predict             Live grade prediction from hold positions + angle (54 features, 79.4% within-1V)
  GET  /api/stats               DB counts + pose coverage (no model internals)

  --- Climber (customer) API ---
  GET  /api/climber/benchmarks              Body mechanics benchmarks by V-grade
  POST /api/climber/recommendations         Personalised route recommendations
  POST /api/climber/weak-points             Technique gaps between current and target grade

  --- Gym (operator) API ---
  GET  /api/gym/dashboard                   Grade distribution, top routes, pose coverage
  GET  /api/gym/setting-recommendations     What grades/styles the board needs more of
  GET  /api/gym/route-performance           Route send counts + quality analytics

Run:
  python3 api/app.py
  python3 api/app.py --port 5001
"""

import os, sys, json, math, sqlite3, random
import numpy as np
import pandas as pd
import psycopg2, psycopg2.extras
import xgboost as xgb
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def _find_project_root() -> str:
    """Resolve project root even if shell cwd is stale after a folder rename."""
    # Try relative to this file first (works when __file__ is absolute or resolvable)
    candidate = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    if os.path.exists(os.path.join(candidate, "kilter.db")):
        return candidate
    # Fallback: walk known Desktop locations
    for name in ("ClimbingML", "Climbing ML", "climbing_ml", "ClimbingML"):
        p = os.path.expanduser(f"~/Desktop/{name}")
        if os.path.exists(os.path.join(p, "kilter.db")):
            return p
    return candidate  # best guess

_PROJECT_ROOT = _find_project_root()
SQLITE_PATH   = os.getenv("KILTER_DB_PATH", os.path.join(_PROJECT_ROOT, "kilter.db"))
MODEL_DIR   = os.path.join(_PROJECT_ROOT, "ml")
DATA_DIR    = os.path.join(_PROJECT_ROOT, "data")

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ──────────────────────────────────────────────────
_model      = None
_boundaries = None
_eval_data  = None

def get_model():
    global _model, _boundaries, _eval_data
    if _model is None:
        _model = xgb.XGBRegressor()
        _model.load_model(os.path.join(MODEL_DIR, "difficulty_model.xgb"))
        with open(os.path.join(MODEL_DIR, "grade_boundaries.json")) as f:
            _boundaries = json.load(f)
        with open(os.path.join(MODEL_DIR, "evaluation.json")) as f:
            _eval_data = json.load(f)
    return _model, _boundaries, _eval_data

# ── Board layout configs ────────────────────────────────────────────────────────
BOARD_CONFIGS = {
    "original": {
        "layout_id":   1,
        "name":        "Kilter Board Original",
        "description": "Commercial/gym layout. Larger ergonomic holds, dynamic gymnastic style.",
        "hold_character": "Bolt Ons (jugs/slopers) + Screw Ons (crimps/pinches)",
        # Climbology CSV coordinate mapping: hole.x = csv_x * 4 + 4, hole.y = csv_y * 4 + 4
        "climbology_x_offset": 4,
        "climbology_y_offset": 4,
        "sets": {
            "all":       None,           # no set filter
            "bolt-ons":  [1],            # Bolt Ons set_id
            "screw-ons": [20],           # Screw Ons set_id
        },
        "default_set": "all",
    },
    "homewall": {
        "layout_id":   8,
        "name":        "Kilter Board Homewall",
        "description": "Home/smaller space layout. More crimps, pinches, and flakes. Technical and skin-friendly.",
        "hold_character": "Mainline + Auxiliary. Higher density (4.34 vs 2.20 holds/sqft). More technical/static.",
        # Homewall Climbology mapping: best match at x_off=-60, y_off=-8
        "climbology_x_offset": -60,
        "climbology_y_offset": -8,
        "sets": {
            "all":           None,          # all sets
            "mainline":      [26, 28],      # Mainline + Mainline Kickboard
            "auxiliary":     [27, 29],      # Auxiliary + Auxiliary Kickboard
            "full-ride":     [26, 27, 28, 29],  # all sets
        },
        "default_set": "mainline",
    },
}

# ── Load board holds once per layout ───────────────────────────────────────────
_board_cache = {}

def get_board_holds(board_type="original", set_filter="all"):
    cache_key = f"{board_type}:{set_filter}"
    if cache_key in _board_cache:
        return _board_cache[cache_key]

    cfg = BOARD_CONFIGS.get(board_type)
    if not cfg:
        cfg = BOARD_CONFIGS["original"]

    climbology = pd.read_csv(os.path.join(DATA_DIR, "kilter_hold_types.csv"))
    climbology["x_mapped"] = climbology["x_coordinate"] * 4 + cfg["climbology_x_offset"]
    climbology["y_mapped"] = climbology["y_coordinate"] * 4 + cfg["climbology_y_offset"]

    set_ids = cfg["sets"].get(set_filter, cfg["sets"].get(cfg["default_set"]))

    set_clause = ""
    if set_ids:
        ids_str = ",".join(str(i) for i in set_ids)
        set_clause = f"AND p.set_id IN ({ids_str})"

    con = sqlite3.connect(SQLITE_PATH)
    holes = pd.read_sql(f"""
        SELECT h.id AS hole_id, h.x, h.y, p.id AS placement_id,
               s.name AS set_name, s.hsm AS set_hsm
        FROM holes h
        JOIN placements p ON p.hole_id = h.id
        JOIN sets s ON p.set_id = s.id
        WHERE p.layout_id = {cfg['layout_id']}
        {set_clause}
        ORDER BY h.y, h.x
    """, con)
    con.close()

    merged = holes.merge(
        climbology[["x_mapped", "y_mapped", "type", "depth", "size"]],
        left_on=["x", "y"], right_on=["x_mapped", "y_mapped"], how="left"
    )
    merged["hold_type"] = merged["type"].fillna("unknown")

    # For Homewall: holds with set_hsm=4 are foot-only kickboard holds
    if board_type == "homewall":
        merged.loc[merged["set_hsm"] == 4, "hold_type"] = "foot"

    x_min, x_max = int(merged["x"].min()), int(merged["x"].max())
    y_min, y_max = int(merged["y"].min()), int(merged["y"].max())
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    holds = []
    for _, row in merged.iterrows():
        x_pct = round((row["x"] - x_min) / x_range * 100, 2)
        y_pct = round(100 - (row["y"] - y_min) / y_range * 100, 2)
        holds.append({
            "id":        int(row["hole_id"]),
            "pid":       int(row["placement_id"]),
            "x":         int(row["x"]),
            "y":         int(row["y"]),
            "x_pct":     x_pct,
            "y_pct":     y_pct,
            "hold_type": row["hold_type"],
            "set_name":  row["set_name"],
        })

    result = {
        "board_type":  board_type,
        "board_name":  cfg["name"],
        "description": cfg["description"],
        "hold_character": cfg["hold_character"],
        "set_filter":  set_filter,
        "sets_available": list(cfg["sets"].keys()),
        "holds":   holds,
        "x_min":   x_min, "x_max": x_max,
        "y_min":   y_min, "y_max": y_max,
        "x_range": x_range, "y_range": y_range,
        "count":   len(holds),
        "type_counts": {
            t: int((merged["hold_type"] == t).sum())
            for t in ["jug", "crimp", "sloper", "pinch", "foot", "unknown"]
        },
    }
    _board_cache[cache_key] = result
    return result

# ── DB connection ───────────────────────────────────────────────────────────────
def get_pg():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "climbing_platform"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        user=os.getenv("DB_USER", os.getenv("USER")),
        password=os.getenv("DB_PASSWORD", ""),
    )

# ── Grade utilities ─────────────────────────────────────────────────────────────
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

# ── Feature extraction (mirrors pipeline/feature_extraction.py) ─────────────────
def _dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def _safe_mean(vals): return sum(vals) / len(vals) if vals else 0.0
def _safe_std(vals):
    if len(vals) < 2: return 0.0
    m = _safe_mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

DYNO_WEIGHTS = {0:2.8,20:2.6,35:2.3,40:2.2,45:2.1,50:2.0,55:1.95,60:1.9,65:1.85,70:1.8}

def _dyno_threshold(angle):
    angles = sorted(DYNO_WEIGHTS.keys())
    if angle <= angles[0]:  return DYNO_WEIGHTS[angles[0]]
    if angle >= angles[-1]: return DYNO_WEIGHTS[angles[-1]]
    for i in range(len(angles)-1):
        if angles[i] <= angle <= angles[i+1]:
            lo, hi = angles[i], angles[i+1]
            t = (angle - lo) / (hi - lo)
            return DYNO_WEIGHTS[lo] + t * (DYNO_WEIGHTS[hi] - DYNO_WEIGHTS[lo])
    return 1.8

def compute_features_from_holds(holds, angle):
    """
    Compute all 54 ML features from a list of hold dicts.
    Each hold: {x_cm, y_cm, role, hold_type (optional)}
    Returns feature dict or None if not enough holds.
    """
    if not holds or len(holds) < 2:
        return None

    hand_holds = [h for h in holds if h.get("role") in ("start", "hand", "finish")]
    foot_holds  = [h for h in holds if h.get("role") == "foot"]

    if len(hand_holds) < 2:
        return None

    # Sort by sequence, then y
    sequenced = [h for h in hand_holds if h.get("hand_sequence") is not None]
    if len(sequenced) >= 2:
        hand_seq = sorted(sequenced, key=lambda h: h["hand_sequence"])
    else:
        hand_seq = sorted(hand_holds, key=lambda h: h.get("y_cm") or 0)

    # Hand-to-hand geometry
    reaches, laterals, verticals, move_angles = [], [], [], []
    direction_changes, prev_lat_sign = 0, None

    for i in range(1, len(hand_seq)):
        h1, h2 = hand_seq[i-1], hand_seq[i]
        x1 = h1.get("x_cm") or 0; y1 = h1.get("y_cm") or 0
        x2 = h2.get("x_cm") or 0; y2 = h2.get("y_cm") or 0
        d  = _dist(x1, y1, x2, y2)
        reaches.append(d)
        laterals.append(abs(x2 - x1))
        verticals.append(abs(y2 - y1))
        # Move angle: 0° = straight up, 90° = horizontal, negative = diagonal
        if d > 0:
            move_angles.append(abs(math.degrees(math.atan2(abs(x2 - x1), max(y2 - y1, 0.001)))))
        lat_sign = 1 if (x2 - x1) > 0 else (-1 if (x2 - x1) < 0 else 0)
        if prev_lat_sign and lat_sign and lat_sign != prev_lat_sign:
            direction_changes += 1
        if lat_sign:
            prev_lat_sign = lat_sign

    if not reaches:
        return None

    avg_reach = _safe_mean(reaches)
    reach_std = _safe_std(reaches)
    reach_cv  = reach_std / avg_reach if avg_reach > 0 else 0
    moves     = len(reaches)

    # Foot geometry
    foot_spreads = []
    if len(foot_holds) >= 2:
        for i in range(1, len(foot_holds)):
            f1, f2 = foot_holds[i-1], foot_holds[i]
            foot_spreads.append(_dist(
                f1.get("x_cm") or 0, f1.get("y_cm") or 0,
                f2.get("x_cm") or 0, f2.get("y_cm") or 0
            ))
    avg_foot_spread = _safe_mean(foot_spreads)

    hand_to_foot_dists, foot_hand_x_offsets = [], []
    for hh in hand_seq:
        hx = hh.get("x_cm") or 0; hy = hh.get("y_cm") or 0
        if foot_holds:
            nf = min(foot_holds, key=lambda f: _dist(hx, hy, f.get("x_cm") or 0, f.get("y_cm") or 0))
            hand_to_foot_dists.append(_dist(hx, hy, nf.get("x_cm") or 0, nf.get("y_cm") or 0))
        feet_below = [f for f in foot_holds if (f.get("y_cm") or 0) < hy]
        if feet_below:
            nb = min(feet_below, key=lambda f: _dist(hx, hy, f.get("x_cm") or 0, f.get("y_cm") or 0))
            foot_hand_x_offsets.append(abs(hx - (nb.get("x_cm") or 0)))

    # Spatial span
    all_xs = [h.get("x_cm") or 0 for h in hand_seq]
    all_ys = [h.get("y_cm") or 0 for h in hand_seq]
    all_hx = [h.get("x_cm") or 0 for h in holds]
    all_hy = [h.get("y_cm") or 0 for h in holds]
    height_span  = max(all_ys) - min(all_ys)
    lateral_span = max(all_xs) - min(all_xs)

    # Dynamic score
    max_reach_z = (max(reaches) - avg_reach) / reach_std if reach_std > 0 else 0
    dyno_threshold = _dyno_threshold(angle)
    dyno_ratio     = max(reaches) / avg_reach if avg_reach > 0 else 0
    dyno_score     = round(dyno_ratio / dyno_threshold, 4)
    dyno_flag      = 1 if (dyno_ratio >= dyno_threshold and max_reach_z >= 2.0 and max(reaches) > 150) else 0

    n_hand = len(hand_holds)
    n_foot = len(foot_holds)
    hand_foot_ratio = round(n_hand / n_foot, 4) if n_foot > 0 else float(n_hand)
    avg_v = avg_reach if avg_reach > 0 else 1

    # Hold type ratios
    hand_typed = [h for h in hand_holds if h.get("hold_type") and h["hold_type"] not in ("unknown", "foot")]
    n_typed = len(hand_typed)
    def _tr(t): return round(sum(1 for h in hand_typed if h.get("hold_type") == t) / n_typed, 4) if n_typed > 0 else 0.0
    all_typed = [h for h in holds if h.get("hold_type") and h["hold_type"] not in ("unknown",)]
    typed_ratio = round(len(all_typed) / len(holds), 4) if holds else 0.0

    return {
        "board_angle_deg":            float(angle),
        "hand_hold_count":            n_hand,
        "foot_hold_count":            n_foot,
        "total_hold_count":           len(holds),
        "moves_count":                moves,
        "foot_ratio":                 round(n_foot / len(holds), 4) if holds else 0,
        "hand_foot_ratio":            hand_foot_ratio,
        "avg_reach_cm":               round(avg_reach, 2),
        "max_reach_cm":               round(max(reaches), 2),
        "min_reach_cm":               round(min(reaches), 2),
        "reach_std_cm":               round(reach_std, 2),
        "reach_range_cm":             round(max(reaches) - min(reaches), 2),
        "reach_cv":                   round(reach_cv, 4),
        "avg_lateral_cm":             round(_safe_mean(laterals), 2),
        "avg_vertical_cm":            round(_safe_mean(verticals), 2),
        "max_lateral_cm":             round(max(laterals), 2),
        "max_vertical_cm":            round(max(verticals), 2),
        "vertical_ratio":             round(_safe_mean(verticals) / avg_v, 4),
        "lateral_ratio":              round(_safe_mean(laterals) / avg_v, 4),
        "avg_foot_spread_cm":         round(avg_foot_spread, 2),
        "avg_hand_to_foot_cm":        round(_safe_mean(hand_to_foot_dists), 2),
        "avg_foot_hand_x_offset_cm":  round(_safe_mean(foot_hand_x_offsets), 2),
        "max_foot_hand_x_offset_cm":  round(max(foot_hand_x_offsets) if foot_hand_x_offsets else 0, 2),
        "height_span_cm":             round(height_span, 2),
        "lateral_span_cm":            round(lateral_span, 2),
        "width_height_ratio":         round(lateral_span / height_span, 4) if height_span > 0 else 0,
        "hold_density":               round(len(holds) / (height_span * lateral_span) * 100, 6) if height_span > 0 and lateral_span > 0 else 0,
        "direction_changes":          direction_changes,
        "zigzag_ratio":               round(direction_changes / moves, 4) if moves > 0 else 0,
        "dyno_score":                 dyno_score,
        "dyno_flag":                  dyno_flag,
        "max_reach_z":                round(max_reach_z, 4),
        "start_x_cm":                 round(hand_seq[0].get("x_cm") or 0, 2),
        "start_y_cm":                 round(hand_seq[0].get("y_cm") or 0, 2),
        "finish_x_cm":                round(hand_seq[-1].get("x_cm") or 0, 2),
        "finish_y_cm":                round(hand_seq[-1].get("y_cm") or 0, 2),
        "start_height_pct":           round((hand_seq[0].get("y_cm") or 0) / 140.0, 4),
        "finish_height_pct":          round((hand_seq[-1].get("y_cm") or 0) / 140.0, 4),
        "centroid_x_cm":              round(_safe_mean(all_xs), 2),
        "centroid_y_cm":              round(_safe_mean(all_ys), 2),
        "total_centroid_x_cm":        round(_safe_mean(all_hx), 2),
        "total_centroid_y_cm":        round(_safe_mean(all_hy), 2),
        "hold_spread_x":              round(_safe_std(all_xs), 2),
        "hold_spread_y":              round(_safe_std(all_ys), 2),
        "crimp_ratio":                _tr("crimp"),
        "sloper_ratio":               _tr("sloper"),
        "jug_ratio":                  _tr("jug"),
        "pinch_ratio":                _tr("pinch"),
        "typed_ratio":                typed_ratio,
        # Move angle features (captures route line / style beyond lateral vs vertical)
        "avg_move_angle_deg":         round(_safe_mean(move_angles), 2),
        "max_move_angle_deg":         round(max(move_angles) if move_angles else 0, 2),
        "move_angle_std":             round(_safe_std(move_angles), 2),
        # Crux detection: consecutive reach ratio spike
        "crux_reach_ratio":           round(max(reaches) / (_safe_mean(reaches) or 1), 4),
        # Route linearity: how straight up vs diagonal the overall path is
        "path_linearity":             round(
            (hand_seq[-1].get("y_cm", 0) - hand_seq[0].get("y_cm", 0)) /
            sum(reaches) if sum(reaches) > 0 else 0, 4),
    }

# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.route("/api/board")
def board():
    """
    Return board holds with Climbology types and display positions.
    Query params:
      board_type: 'original' (default) or 'homewall'
      set_filter: 'all' (default), 'bolt-ons', 'screw-ons', 'mainline', 'auxiliary', 'full-ride'
    """
    board_type = request.args.get("board_type", "original")
    set_filter = request.args.get("set_filter", "all")
    return jsonify(get_board_holds(board_type, set_filter))


@app.route("/api/boards")
def boards():
    """Return metadata for all available board types."""
    return jsonify({
        k: {
            "name":           v["name"],
            "description":    v["description"],
            "hold_character": v["hold_character"],
            "sets":           list(v["sets"].keys()),
            "default_set":    v["default_set"],
        }
        for k, v in BOARD_CONFIGS.items()
    })


@app.route("/api/routes")
def routes():
    """
    Return routes from board_routes.
    Query params: grade, angle, min_sends, limit (default 50), offset (default 0), q (name search)
    """
    grade     = request.args.get("grade")
    angle     = request.args.get("angle", type=int)
    min_sends = request.args.get("min_sends", default=5, type=int)
    limit     = min(request.args.get("limit", default=50, type=int), 200)
    offset    = request.args.get("offset", default=0, type=int)
    q         = request.args.get("q", "").strip()
    has_pose  = request.args.get("has_pose", "").strip() == "1"

    filters = ["source = 'kilter'", "community_grade IS NOT NULL", "difficulty_score IS NOT NULL"]
    params  = []

    if grade:
        filters.append("community_grade = %s"); params.append(grade)
    if angle is not None:
        filters.append("board_angle_deg = %s"); params.append(angle)
    if min_sends > 0:
        filters.append("send_count >= %s"); params.append(min_sends)
    if q:
        filters.append("name ILIKE %s"); params.append(f"%{q}%")
    if has_pose:
        filters.append("EXISTS (SELECT 1 FROM pose_frames pf WHERE UPPER(pf.climb_uuid) = UPPER(br.external_id) LIMIT 1)")

    where = " AND ".join(filters)

    try:
        pg  = get_pg()
        cur = pg.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(f"SELECT COUNT(*) FROM board_routes br WHERE {where}", params)
        total = cur.fetchone()[0]

        cur.execute(f"""
            SELECT br.id, br.name, br.setter_name, br.community_grade, br.difficulty_score,
                   br.board_angle_deg, br.send_count, br.avg_quality_rating,
                   (EXISTS (
                       SELECT 1 FROM pose_frames pf
                       WHERE UPPER(pf.climb_uuid) = UPPER(br.external_id)
                       LIMIT 1
                   )) AS has_pose
            FROM board_routes br
            WHERE {where}
            ORDER BY send_count DESC NULLS LAST
            LIMIT %s OFFSET %s
        """, params + [limit, offset])

        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["id"] = str(r["id"])
            r["board_angle_deg"] = int(r["board_angle_deg"]) if r["board_angle_deg"] else None
            r["difficulty_score"] = round(float(r["difficulty_score"]), 4) if r["difficulty_score"] else None
            r["avg_quality_rating"] = round(float(r["avg_quality_rating"]), 2) if r["avg_quality_rating"] else None
            r["has_pose"] = bool(r.get("has_pose", False))

        cur.close(); pg.close()
        return jsonify({"total": total, "routes": rows, "limit": limit, "offset": offset})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/route/<route_id>")
def route_detail(route_id):
    """Return a single route with all its holds."""
    board_data = get_board_holds()
    hold_lookup = {h["pid"]: h for h in board_data["holds"]}

    try:
        pg  = get_pg()
        cur = pg.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT id, name, setter_name, community_grade, difficulty_score,
                   board_angle_deg, send_count, avg_quality_rating, external_id
            FROM board_routes WHERE id = %s
        """, (route_id,))
        row = cur.fetchone()
        if not row:
            cur.close(); pg.close()
            return jsonify({"error": "Route not found"}), 404

        route = dict(row)
        route["id"] = str(route["id"])
        route["board_angle_deg"] = int(route["board_angle_deg"]) if route["board_angle_deg"] else None

        cur.execute("""
            SELECT grid_col, grid_row, position_x_cm, position_y_cm,
                   role, hand_sequence, hold_type, board_angle_deg
            FROM route_holds WHERE route_id = %s
            ORDER BY hand_sequence NULLS LAST, position_y_cm
        """, (route_id,))

        holds = []
        for h in cur.fetchall():
            hd = dict(h)
            hd["position_x_cm"] = round(float(hd["position_x_cm"]), 2) if hd["position_x_cm"] else None
            hd["position_y_cm"] = round(float(hd["position_y_cm"]), 2) if hd["position_y_cm"] else None
            # Attach display percentages from board hold lookup via position match
            if hd["position_x_cm"] and hd["position_y_cm"]:
                x_pct = round((hd["position_x_cm"] - board_data["x_min"]) / board_data["x_range"] * 100, 2)
                y_pct = round(100 - (hd["position_y_cm"] - board_data["y_min"]) / board_data["y_range"] * 100, 2)
                hd["x_pct"] = x_pct
                hd["y_pct"] = y_pct
            holds.append(hd)

        route["holds"] = holds
        cur.close(); pg.close()
        return jsonify(route)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Live grade prediction from placed holds.

    Body JSON: {
        "holds": [
            {"x_cm": float, "y_cm": float, "role": "start"|"hand"|"foot"|"finish",
             "hold_type": "jug"|"crimp"|... (optional)}
        ],
        "angle": int
    }
    Returns: {grade, score, features, confidence}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    holds = data.get("holds", [])
    angle = int(data.get("angle", 40))

    if len(holds) < 2:
        return jsonify({"error": "Need at least 2 holds"}), 400

    # Validate and normalize roles
    valid_roles = {"start", "hand", "foot", "finish"}
    for h in holds:
        if h.get("role") not in valid_roles:
            h["role"] = "hand"

    features = compute_features_from_holds(holds, angle)
    if features is None:
        return jsonify({"error": "Need at least 2 hand holds (start/hand/finish)"}), 400

    model, boundaries, eval_data = get_model()
    feature_cols = eval_data["feature_cols"]

    # Build feature vector
    feat_vec = np.array([[features.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
    # Clip outlier reaches
    for i, col in enumerate(feature_cols):
        if "reach" in col:
            feat_vec[0][i] = min(feat_vec[0][i], 300.0)

    score = float(model.predict(feat_vec)[0])
    score = max(0.0, min(1.0, score))
    grade = score_to_grade(score, boundaries)

    # Confidence: distance to nearest grade boundary
    info = boundaries.get(grade, {})
    if info:
        lo, hi = info.get("lo", 0), info.get("hi", 1)
        margin = (hi - lo)
        dist_to_edge = min(abs(score - lo), abs(score - hi))
        confidence = round(min(dist_to_edge / (margin / 2), 1.0), 3) if margin > 0 else 0.5
    else:
        confidence = 0.5

    # Route DNA (0-100 percentile-like scores for display)
    dna = {
        "reach":      min(100, round(features["avg_reach_cm"] / 100 * 100)),
        "sustained":  min(100, round(features["moves_count"] / 12 * 100)),
        "dynamic":    min(100, round(features["dyno_score"] * 50)),
        "lateral":    min(100, round(features["lateral_ratio"] * 100)),
        "complexity": min(100, round(features["zigzag_ratio"] * 120 + features.get("move_angle_std", 0) * 0.8)),
        "power":      min(100, round((features["sloper_ratio"] * 60 + features["pinch_ratio"] * 40 + features["crimp_ratio"] * 30))),
    }

    return jsonify({
        "grade":      grade,
        "score":      round(score, 4),
        "confidence": confidence,
        "dna":        dna,
        "features":   {k: round(v, 3) if isinstance(v, float) else v for k, v in features.items()},
    })


@app.route("/api/suggest", methods=["POST"])
def suggest():
    """
    Suggest next holds for the pathfinder.
    Given placed holds and angle, suggest the best next hand hold positions.

    Body: {"holds": [...], "angle": int, "count": int}
    """
    data = request.get_json(silent=True) or {}
    holds       = data.get("holds", [])
    angle       = int(data.get("angle", 40))
    count       = min(int(data.get("count", 6)), 12)
    board_data  = get_board_holds()

    if not holds:
        return jsonify({"suggestions": []})

    # Get last hand hold
    hand_holds = [h for h in holds if h.get("role") in ("start", "hand")]
    if not hand_holds:
        return jsonify({"suggestions": []})

    last = hand_holds[-1]
    lx   = last.get("x_cm", 0)
    ly   = last.get("y_cm", 0)

    # Placed positions (to exclude)
    placed = {(h.get("x_cm"), h.get("y_cm")) for h in holds}

    # Scale reach range by angle — steeper = shorter optimal reach
    min_reach = 30 if angle <= 40 else 25
    max_reach = 90 if angle <= 40 else 75

    center_x = (board_data["x_min"] + board_data["x_max"]) / 2
    candidates = []
    for hold in board_data["holds"]:
        if hold["hold_type"] == "foot":
            continue
        hx = hold["x"]
        hy = hold["y"]
        if (hx, hy) in placed:
            continue
        d = _dist(lx, ly, hx, hy)
        if min_reach <= d <= max_reach and hy >= ly:
            lateral_penalty = abs(hx - center_x) / 100
            hold_bonus = {"jug": 0, "crimp": 0.05, "pinch": 0.1, "sloper": 0.15, "unknown": 0.2}
            score = d + lateral_penalty * 10 + hold_bonus.get(hold["hold_type"], 0.2) * 20

            # Build human-readable impact tags
            dx = abs(hx - lx)
            dy = hy - ly   # positive = upward
            impact = []
            if d > 65:
                impact.append("big reach")
            elif d > 48:
                impact.append("medium reach")
            else:
                impact.append("close move")
            if dx > 35:
                impact.append("lateral +" + str(round(dx)) + "cm")
            if dy > 50:
                impact.append("high step")
            if hold["hold_type"] in ("crimp", "sloper", "pinch"):
                impact.append(hold["hold_type"] + " (harder)")
            elif hold["hold_type"] == "jug":
                impact.append("jug (rest)")
            if d > 60 and dx > 25:
                impact.append("increases difficulty")

            candidates.append({
                **hold,
                "dist":   round(d, 1),
                "score":  round(score, 2),
                "impact": impact,
            })

    candidates.sort(key=lambda h: h["score"])
    suggestions = candidates[:count]

    return jsonify({"suggestions": suggestions})


@app.route("/api/auto_generate", methods=["POST"])
def auto_generate():
    """
    Auto-generate a complete route.
    Body: {"angle": int, "difficulty": float 0-1, "hold_count": int}
    """
    data        = request.get_json(silent=True) or {}
    angle       = int(data.get("angle", 40))
    difficulty  = float(data.get("difficulty", 0.4))   # 0=easy, 1=hard
    hold_count  = min(int(data.get("hold_count", 8)), 16)
    board_data  = get_board_holds()

    all_holds   = [h for h in board_data["holds"] if h["hold_type"] != "foot"]
    y_min, y_max = board_data["y_min"], board_data["y_max"]
    y_range     = max(y_max - y_min, 1)
    center_x    = (board_data["x_min"] + board_data["x_max"]) / 2

    # Start holds: bottom 30% of board, spread 20-50cm apart
    bottom_holds = [h for h in all_holds if (h["y"] - y_min) / y_range < 0.30]
    if len(bottom_holds) < 4:
        bottom_holds = sorted(all_holds, key=lambda h: h["y"])[:max(20, len(all_holds)//4)]

    # Pick start holds: two near-center, spread by difficulty (harder = wider spread)
    target_spread = 20 + difficulty * 35
    best_pair, best_diff = None, float("inf")
    sample = random.sample(bottom_holds, min(40, len(bottom_holds)))
    for i, a in enumerate(sample):
        for b in sample[i+1:]:
            d = _dist(a["x"], a["y"], b["x"], b["y"])
            if abs(d - target_spread) < best_diff and abs(a["x"] - center_x) < 80 and abs(b["x"] - center_x) < 80:
                best_diff = abs(d - target_spread)
                best_pair = (a, b)

    if not best_pair:
        return jsonify({"error": "not enough holds"}), 400

    route_holds = [
        {**best_pair[0], "role": "start"},
        {**best_pair[1], "role": "start"},
    ]
    placed = {(best_pair[0]["x"], best_pair[0]["y"]), (best_pair[1]["x"], best_pair[1]["y"])}
    last_x = (best_pair[0]["x"] + best_pair[1]["x"]) / 2
    last_y = (best_pair[0]["y"] + best_pair[1]["y"]) / 2

    # Scale reach by angle and difficulty
    reach_min = max(20, 30 - difficulty * 10)
    reach_max = min(95, 55 + difficulty * 35)
    hand_slots = hold_count - 3  # minus 2 starts + 1 finish

    # Add hand holds going upward
    hold_bonus_diff = {"jug": -0.1, "crimp": 0.1, "pinch": 0.15, "sloper": 0.2, "unknown": 0.05}
    for _ in range(hand_slots):
        candidates = []
        for h in all_holds:
            if (h["x"], h["y"]) in placed:
                continue
            d = _dist(last_x, last_y, h["x"], h["y"])
            if reach_min <= d <= reach_max and h["y"] >= last_y:
                lateral = abs(h["x"] - center_x) / 100
                diff_match = abs(hold_bonus_diff.get(h["hold_type"], 0.05) - difficulty * 0.15)
                score = d * 0.4 + lateral * 8 + diff_match * 20 + random.uniform(0, 5)
                candidates.append((score, h))
        if not candidates:
            break
        candidates.sort(key=lambda t: t[0])
        # Pick from top 3 for variety
        pick = random.choice(candidates[:min(3, len(candidates))])[1]
        route_holds.append({**pick, "role": "hand"})
        placed.add((pick["x"], pick["y"]))
        last_x, last_y = pick["x"], pick["y"]

    # Finish hold: top 20% of board, near center
    top_holds = [h for h in all_holds
                 if (h["y"] - y_min) / y_range > 0.75
                 and (h["x"], h["y"]) not in placed]
    if top_holds:
        finish = min(top_holds, key=lambda h: abs(h["x"] - center_x) + abs(h["y"] - y_max) * 0.5)
        route_holds.append({**finish, "role": "finish"})

    return jsonify({"holds": route_holds})


@app.route("/api/stats")
def stats():
    """Quick DB stats for the UI header."""
    try:
        pg  = get_pg()
        cur = pg.cursor()
        cur.execute("SELECT COUNT(*) FROM board_routes WHERE source='kilter'")
        n_routes = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM board_routes WHERE source='kilter' AND community_grade IS NOT NULL")
        n_graded = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT climb_uuid), COUNT(*) FROM pose_frames WHERE climb_uuid IS NOT NULL")
        pose_row = cur.fetchone()
        cur.close(); pg.close()
        return jsonify({
            "total_routes":   n_routes,
            "graded_routes":  n_graded,
            "grade_range":    "V0–V14",
            "hold_count":     get_board_holds()["count"],
            "pose_climbs":    pose_row[0],
            "pose_frames":    pose_row[1],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Customer (Climber) API ────────────────────────────────────────────────────
#
# These endpoints are designed for the end-user product: a climber who wants
# to understand their weaknesses and get personalized route recommendations.
#
# All /api/climber/* endpoints accept a climber profile in the request body
# (height_cm, wingspan_cm, experience, on_sight_grade) and return content
# personalised to that climber. In production this would come from auth/session;
# for now it's passed explicitly so the frontend can use localStorage profile.

@app.route("/api/climber/benchmarks")
def climber_benchmarks():
    """
    Body mechanics benchmarks by V-grade.
    Returns median hip angle, tension, arm reach, and straight-arm % for each
    grade band — gives a climber a sense of where their body should be at their
    target grade.
    """
    try:
        pg  = get_pg()
        cur = pg.cursor()
        cur.execute("""
            SELECT
                br.community_grade,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pf.hip_angle_deg)::numeric, 1)        AS p50_hip_angle,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pf.tension_score)::numeric, 3)        AS p50_tension,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                    (pf.left_arm_reach_norm + pf.right_arm_reach_norm) / 2)::numeric, 3)                AS p50_arm_reach,
                ROUND(AVG(CASE WHEN pf.is_straight_arm_l OR pf.is_straight_arm_r THEN 1.0 ELSE 0.0 END)::numeric, 3) AS pct_straight_arm,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pf.com_height_norm)::numeric, 3)      AS p50_com_height,
                ROUND(AVG(pf.hip_spread_deg)::numeric, 1)                                               AS avg_hip_spread,
                COUNT(DISTINCT pf.climb_uuid)                                                            AS route_count,
                COUNT(*)                                                                                  AS frame_count
            FROM pose_frames pf
            JOIN board_routes br ON br.external_id = pf.climb_uuid
            WHERE br.community_grade IS NOT NULL
              AND pf.hip_angle_deg IS NOT NULL
              AND pf.tension_score IS NOT NULL
            GROUP BY br.community_grade
            ORDER BY MIN(br.difficulty_score)
        """)
        rows = cur.fetchall()
        cur.close(); pg.close()

        cols = ["grade", "hip_angle", "tension", "arm_reach", "pct_straight_arm",
                "com_height", "hip_spread", "route_count", "frame_count"]
        return jsonify({
            "benchmarks": [
                dict(zip(cols, (v if not isinstance(v, type(None)) else None
                                for v in row)))
                for row in rows
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/climber/recommendations", methods=["POST"])
def climber_recommendations():
    """
    Personalized route recommendations for a climber.

    Body JSON: {
        "current_grade": "V5",          // climber's current on-sight grade
        "target_grade":  "V6",          // grade they want to reach (optional)
        "angle":         40,            // preferred angle (optional)
        "weakness":      "dynamic",     // DNA dimension to target (optional)
        "limit":         10
    }

    Returns routes that:
      - Are at the target grade (or 1V above current if not specified)
      - Prioritize the targeted DNA dimension
      - Have pose data available (so climber can study movement)
    """
    data          = request.get_json(silent=True) or {}
    current_grade = data.get("current_grade", "V4")
    target_grade  = data.get("target_grade")
    angle         = data.get("angle")
    weakness      = data.get("weakness")           # "dynamic","lateral","reach","power","sustained","complexity"
    limit         = min(int(data.get("limit", 10)), 30)

    # Map V-grade strings to difficulty_score range
    GRADE_SCORES = {
        "V0":0.30,"V1":0.35,"V2":0.40,"V3":0.45,"V4":0.50,"V5":0.55,
        "V6":0.60,"V7":0.65,"V8":0.70,"V9":0.74,"V10":0.78,"V11":0.82,
        "V12":0.85,"V13":0.89,"V14":0.94,
    }
    grades_ordered = list(GRADE_SCORES.keys())

    if not target_grade:
        idx = grades_ordered.index(current_grade) if current_grade in grades_ordered else 4
        target_grade = grades_ordered[min(idx + 1, len(grades_ordered) - 1)]

    target_score = GRADE_SCORES.get(target_grade, 0.55)
    score_lo     = max(0.0, target_score - 0.04)
    score_hi     = min(1.0, target_score + 0.04)

    try:
        pg  = get_pg()
        cur = pg.cursor()

        angle_filter = "AND br.board_angle_deg = %s" if angle else ""
        angle_params = [angle] if angle else []

        # Base query: routes at the target grade with send counts
        cur.execute(f"""
            SELECT br.id, br.external_id, br.name, br.community_grade,
                   br.board_angle_deg, br.send_count, br.avg_quality_rating,
                   EXISTS(
                       SELECT 1 FROM pose_frames pf WHERE pf.climb_uuid = br.external_id LIMIT 1
                   ) AS has_pose
            FROM board_routes br
            WHERE br.difficulty_score BETWEEN %s AND %s
              AND br.community_grade IS NOT NULL
              AND br.send_count >= 20
              {angle_filter}
            ORDER BY br.avg_quality_rating DESC NULLS LAST, br.send_count DESC
            LIMIT %s
        """, [score_lo, score_hi] + angle_params + [limit * 3])
        rows = cur.fetchall()
        cur.close(); pg.close()

        routes = []
        for r in rows:
            rid, ext_id, name, grade, ang, sends, quality, has_pose = r
            rec = {
                "id":            rid,
                "climb_uuid":    ext_id,
                "name":          name or "Unnamed",
                "grade":         grade,
                "angle":         ang,
                "send_count":    sends,
                "quality":       round(float(quality), 2) if quality else None,
                "has_pose":      has_pose,
                "why":           _recommendation_reason(grade, current_grade, weakness, has_pose),
            }
            routes.append(rec)

        # Prioritise pose routes and quality, then trim to limit
        routes.sort(key=lambda r: (not r["has_pose"], -(r["quality"] or 0)))
        routes = routes[:limit]

        return jsonify({
            "current_grade": current_grade,
            "target_grade":  target_grade,
            "weakness":      weakness,
            "routes":        routes,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _recommendation_reason(grade, current_grade, weakness, has_pose):
    """Build a short human-readable reason string for why this route is recommended."""
    parts = []
    if grade == current_grade:
        parts.append("Consolidate your " + grade)
    else:
        parts.append("Push into " + grade)
    if weakness:
        labels = {
            "dynamic":    "trains explosive movement",
            "lateral":    "builds lateral reach",
            "reach":      "extends your max reach",
            "power":      "develops contact strength",
            "sustained":  "builds endurance",
            "complexity": "trains sequencing",
        }
        parts.append(labels.get(weakness, "targets your weakness"))
    if has_pose:
        parts.append("beta video available")
    return " · ".join(parts)


@app.route("/api/climber/weak-points", methods=["POST"])
def climber_weak_points():
    """
    Identify technique weak points based on how the climber's target grade
    benchmarks compare to the overall average.

    Body JSON: { "current_grade": "V5", "target_grade": "V6" }

    Returns a ranked list of technique dimensions where the target grade
    demands significantly more than the current grade — the specific gaps
    the climber needs to close.
    """
    data          = request.get_json(silent=True) or {}
    current_grade = data.get("current_grade", "V4")
    target_grade  = data.get("target_grade", "V5")

    try:
        pg  = get_pg()
        cur = pg.cursor()
        cur.execute("""
            SELECT
                br.community_grade,
                ROUND(AVG(pf.hip_angle_deg)::numeric, 1)          AS avg_hip_angle,
                ROUND(AVG(pf.tension_score)::numeric, 3)           AS avg_tension,
                ROUND(AVG((pf.left_arm_reach_norm+pf.right_arm_reach_norm)/2)::numeric, 3) AS avg_reach,
                ROUND(AVG(CASE WHEN pf.is_straight_arm_l OR pf.is_straight_arm_r THEN 1.0 ELSE 0.0 END)::numeric, 3) AS pct_straight,
                ROUND(AVG(pf.com_height_norm)::numeric, 3)         AS avg_com,
                ROUND(AVG(pf.hip_spread_deg)::numeric, 1)          AS avg_hip_spread,
                ROUND(AVG(ABS(pf.left_arm_reach_norm - pf.right_arm_reach_norm))::numeric, 3) AS reach_asymmetry
            FROM pose_frames pf
            JOIN board_routes br ON br.external_id = pf.climb_uuid
            WHERE br.community_grade = ANY(%s)
              AND pf.hip_angle_deg IS NOT NULL
            GROUP BY br.community_grade
        """, [[current_grade, target_grade]])
        rows = cur.fetchall()
        cur.close(); pg.close()

        by_grade = {}
        for row in rows:
            by_grade[row[0]] = {
                "hip_angle":      row[1],
                "tension":        row[2],
                "arm_reach":      row[3],
                "pct_straight":   row[4],
                "com_height":     row[5],
                "hip_spread":     row[6],
                "reach_asymmetry":row[7],
            }

        current = by_grade.get(current_grade, {})
        target  = by_grade.get(target_grade, {})

        if not current or not target:
            return jsonify({"error": f"Not enough pose data for {current_grade} or {target_grade} yet"}), 404

        # Compute gaps: positive = target demands more/higher value
        DIMENSIONS = [
            ("tension",        "Body Tension",        +1,
             "core tightness across the climb — harder routes demand more lock-off strength"),
            ("arm_reach",      "Arm Extension",       +1,
             "how far arms extend relative to torso — longer reach = more efficient movement"),
            ("hip_spread",     "Hip Spread",          +1,
             "how open your hips stay — wider spread = better footwork and balance"),
            ("com_height",     "Center of Mass Height", +1,
             "how high you stay on the wall — higher COM = better body positioning"),
            ("pct_straight",   "Straight-Arm Resting", +1,
             "resting on straight arms saves energy — harder routes require more of this"),
            ("reach_asymmetry","L/R Reach Balance",   -1,
             "lower asymmetry = more balanced technique (negative gap = good)"),
        ]

        weak_points = []
        for key, label, direction, desc in DIMENSIONS:
            c_val = current.get(key)
            t_val = target.get(key)
            if c_val is None or t_val is None:
                continue
            gap = float(t_val) - float(c_val)  # how much target differs from current
            # Scale gap to 0-100 severity
            severity = min(100, round(abs(gap) * direction * 200))
            weak_points.append({
                "dimension":  key,
                "label":      label,
                "description": desc,
                "current_val": float(c_val),
                "target_val":  float(t_val),
                "gap":         round(gap, 3),
                "severity":    severity,
                "needs_more":  (gap * direction) > 0,
            })

        weak_points.sort(key=lambda w: -w["severity"])

        return jsonify({
            "current_grade": current_grade,
            "target_grade":  target_grade,
            "weak_points":   weak_points,
            "data_available": bool(current and target),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Gym API ───────────────────────────────────────────────────────────────────
#
# These endpoints are for gym operators / route setters who want to understand
# how their board is performing and what routes to set next.
#
# In production: auth-gated to gym admin role. For now open for prototyping.

@app.route("/api/gym/dashboard")
def gym_dashboard():
    """
    High-level gym health metrics for the gym operator dashboard.
    Returns route counts by grade, quality distribution, pose coverage,
    and top/bottom performing routes.
    """
    try:
        pg  = get_pg()
        cur = pg.cursor()

        # Grade distribution
        cur.execute("""
            SELECT community_grade,
                   COUNT(*)                            AS route_count,
                   ROUND(AVG(send_count)::numeric, 0)  AS avg_sends,
                   ROUND(AVG(avg_quality_rating)::numeric, 2) AS avg_quality
            FROM board_routes
            WHERE community_grade IS NOT NULL AND source = 'kilter'
            GROUP BY community_grade
            ORDER BY MIN(difficulty_score)
        """)
        grade_dist = [
            {"grade": r[0], "count": r[1], "avg_sends": r[2],
             "avg_quality": float(r[3]) if r[3] else None}
            for r in cur.fetchall()
        ]

        # Top 5 routes by quality
        cur.execute("""
            SELECT name, community_grade, board_angle_deg, send_count, avg_quality_rating
            FROM board_routes
            WHERE avg_quality_rating IS NOT NULL AND send_count >= 50 AND source = 'kilter'
            ORDER BY avg_quality_rating DESC LIMIT 5
        """)
        top_routes = [
            {"name": r[0], "grade": r[1], "angle": r[2],
             "sends": r[3], "quality": float(r[4])}
            for r in cur.fetchall()
        ]

        # Pose coverage by grade
        cur.execute("""
            SELECT br.community_grade, COUNT(DISTINCT pf.climb_uuid) AS climbs_with_pose
            FROM board_routes br
            JOIN pose_frames pf ON pf.climb_uuid = br.external_id
            WHERE br.community_grade IS NOT NULL
            GROUP BY br.community_grade
            ORDER BY MIN(br.difficulty_score)
        """)
        pose_cov = {r[0]: r[1] for r in cur.fetchall()}

        # Add pose coverage into grade_dist
        for g in grade_dist:
            g["pose_routes"] = pose_cov.get(g["grade"], 0)

        # Overall counts
        cur.execute("SELECT COUNT(*), COUNT(*) FILTER (WHERE send_count >= 5) FROM board_routes WHERE source='kilter'")
        total, active = cur.fetchone()

        cur.close(); pg.close()

        return jsonify({
            "total_routes":  total,
            "active_routes": active,
            "grade_distribution": grade_dist,
            "top_routes":    top_routes,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gym/setting-recommendations")
def setting_recommendations():
    """
    Route setting recommendations — what grades and styles the board needs more of.

    Compares current grade distribution to an 'ideal' bell curve centered at V4-V5,
    identifies under-served grades, and uses the DNA model to flag what route
    styles are missing (e.g., 'Need more dynamic V6, board is heavy on crimpy V6').
    """
    try:
        pg  = get_pg()
        cur = pg.cursor()

        # Current distribution
        cur.execute("""
            SELECT community_grade, COUNT(*) AS n,
                   ROUND(AVG(difficulty_score)::numeric, 4) AS avg_score
            FROM board_routes
            WHERE community_grade IS NOT NULL AND source = 'kilter' AND send_count >= 5
            GROUP BY community_grade
            ORDER BY MIN(difficulty_score)
        """)
        rows = cur.fetchall()
        cur.close(); pg.close()

        if not rows:
            return jsonify({"recommendations": []})

        total = sum(r[1] for r in rows)
        dist  = {r[0]: r[1] for r in rows}

        # Ideal distribution: roughly bell curve V3-V6 core, tails at extremes
        IDEAL_PCT = {
            "V0": 3, "V1": 5, "V2": 6, "V3": 12, "V4": 15,
            "V5": 15, "V6": 13, "V7": 10, "V8": 9, "V9": 5,
            "V10": 3, "V11": 2, "V12": 1, "V13": 0.5, "V14": 0.5,
        }

        recs = []
        for grade, ideal_pct in IDEAL_PCT.items():
            actual_n   = dist.get(grade, 0)
            actual_pct = actual_n / total * 100 if total else 0
            gap_pct    = ideal_pct - actual_pct
            if gap_pct > 2.0:   # under-served by more than 2 percentage points
                recs.append({
                    "grade":       grade,
                    "actual_pct":  round(actual_pct, 1),
                    "ideal_pct":   ideal_pct,
                    "gap":         round(gap_pct, 1),
                    "actual_count": actual_n,
                    "action":      f"Set {max(1, round(gap_pct / 100 * total))} more {grade} routes",
                    "priority":    "high" if gap_pct > 5 else "medium",
                })

        recs.sort(key=lambda r: -r["gap"])

        return jsonify({
            "total_active_routes": total,
            "recommendations":     recs,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gym/route-performance")
def route_performance():
    """
    Route performance analytics — which routes are getting sends and quality ratings.
    Useful for setters to understand what's working.
    Query params: grade (optional), angle (optional), limit (default 20)
    """
    grade = request.args.get("grade")
    angle = request.args.get("angle", type=int)
    limit = min(request.args.get("limit", 20, type=int), 100)

    try:
        pg  = get_pg()
        cur = pg.cursor()

        filters = ["source = 'kilter'", "send_count >= 5"]
        params  = []
        if grade:
            filters.append("community_grade = %s"); params.append(grade)
        if angle:
            filters.append("board_angle_deg = %s"); params.append(angle)

        where = "WHERE " + " AND ".join(filters)
        cur.execute(f"""
            SELECT id, name, community_grade, board_angle_deg,
                   send_count, avg_quality_rating, setter_name,
                   EXISTS(SELECT 1 FROM pose_frames pf WHERE pf.climb_uuid = external_id LIMIT 1) AS has_pose
            FROM board_routes
            {where}
            ORDER BY avg_quality_rating DESC NULLS LAST, send_count DESC
            LIMIT %s
        """, params + [limit])
        rows = cur.fetchall()
        cur.close(); pg.close()

        return jsonify({
            "routes": [
                {
                    "id":      r[0], "name": r[1] or "Unnamed",
                    "grade":   r[2], "angle": r[3],
                    "sends":   r[4],
                    "quality": round(float(r[5]), 2) if r[5] else None,
                    "setter":  r[6],
                    "has_pose": r[7],
                }
                for r in rows
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pose/predictions")
def pose_predictions():
    """
    Returns pose-based difficulty predictions for every route that has
    pose data, alongside the actual difficulty from kilter.db.
    This is the main endpoint for seeing the pose model working.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from ml.pose_difficulty_model import load_model, kilter_to_vgrade
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    try:
        pipe, feature_cols, col_medians = load_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    # Pull pose aggregates from PostgreSQL
    pg  = get_pg()
    cur = pg.cursor()
    agg_cols = ", ".join([
        "climb_uuid",
        "COUNT(DISTINCT attempt_id) AS video_count",
        "COUNT(*) AS frame_count",
        "AVG(hip_angle_deg) AS pose_avg_hip_angle",
        "MIN(hip_angle_deg) AS pose_min_hip_angle",
        "MAX(hip_angle_deg) - MIN(hip_angle_deg) AS pose_hip_angle_range",
        "PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY hip_angle_deg) AS pose_p10_hip_angle",
        "AVG(hip_spread_deg) AS pose_avg_hip_spread",
        "MAX(hip_spread_deg) AS pose_max_hip_spread",
        "AVG(elbow_l_deg) AS pose_avg_elbow_l",
        "AVG(elbow_r_deg) AS pose_avg_elbow_r",
        "MIN(elbow_l_deg) AS pose_min_elbow_l",
        "MIN(elbow_r_deg) AS pose_min_elbow_r",
        "ABS(AVG(elbow_l_deg) - AVG(elbow_r_deg)) AS pose_elbow_asymmetry",
        "AVG(CASE WHEN is_straight_arm_l THEN 1.0 ELSE 0.0 END) AS pose_pct_straight_l",
        "AVG(CASE WHEN is_straight_arm_r THEN 1.0 ELSE 0.0 END) AS pose_pct_straight_r",
        "AVG(shoulder_l_deg) AS pose_avg_shoulder_l",
        "AVG(shoulder_r_deg) AS pose_avg_shoulder_r",
        "AVG(knee_l_deg) AS pose_avg_knee_l",
        "AVG(knee_r_deg) AS pose_avg_knee_r",
        "MIN(LEAST(knee_l_deg, knee_r_deg)) AS pose_min_knee",
        "AVG(foot_hand_height_diff) AS pose_avg_foot_hand_diff",
        "AVG(CASE WHEN foot_hand_height_diff > 0 THEN 1.0 ELSE 0.0 END) AS pose_pct_high_feet",
        "AVG(tension_score) AS pose_avg_tension",
        "PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY tension_score) AS pose_p90_tension",
        "MAX(com_height_norm) - MIN(com_height_norm) AS pose_com_travel",
        "AVG(com_to_hands_dist) AS pose_avg_com_to_hands",
        "ABS(AVG(left_arm_reach_norm) - AVG(right_arm_reach_norm)) AS pose_reach_asymmetry",
        "AVG(com_velocity) AS pose_avg_com_vel",
        "MAX(com_velocity) AS pose_peak_com_vel",
        "AVG((hand_l_velocity + hand_r_velocity) / 2.0) AS pose_avg_hand_vel",
        "MAX(GREATEST(hand_l_velocity, hand_r_velocity)) AS pose_peak_hand_vel",
        "AVG(hip_ang_vel) AS pose_avg_hip_ang_vel",
        "MAX(GREATEST(elbow_l_ang_vel, elbow_r_ang_vel)) AS pose_peak_elbow_ang_vel",
    ])
    cur.execute(f"""
        SELECT {agg_cols}
        FROM pose_frames
        WHERE climb_uuid IS NOT NULL
          AND hip_angle_deg IS NOT NULL
          AND elbow_l_deg IS NOT NULL
        GROUP BY climb_uuid
    """)
    rows = cur.fetchall()
    col_names = [d[0] for d in cur.description]
    cur.close(); pg.close()

    # Look up actual difficulty from kilter.db
    uuids = [r[0] for r in rows]
    con   = sqlite3.connect(SQLITE_PATH)
    diff_rows = con.execute(f"""
        SELECT cs.climb_uuid, c.name, c.setter_username,
               cs.difficulty_average, cs.quality_average, cs.ascensionist_count
        FROM climb_stats cs JOIN climbs c ON c.uuid = cs.climb_uuid
        WHERE cs.climb_uuid IN ({','.join('?'*len(uuids))})
        GROUP BY cs.climb_uuid HAVING difficulty_average = MAX(cs.difficulty_average)
    """, uuids).fetchall()
    con.close()
    diff_lookup = {r[0]: r for r in diff_rows}

    results = []
    for row in rows:
        feat = dict(zip(col_names, row))
        c_uuid = feat["climb_uuid"]
        row_vec = [feat.get(col, col_medians.get(col, 0.0)) or col_medians.get(col, 0.0)
                   for col in feature_cols]
        pred_score = float(pipe.predict([row_vec])[0])
        pred_score = max(14.0, min(33.0, pred_score))

        dl = diff_lookup.get(c_uuid)
        actual_difficulty = float(dl[3]) if dl else None

        results.append({
            "climb_uuid":           c_uuid,
            "route_name":           dl[1] if dl else None,
            "setter":               dl[2] if dl else None,
            "actual_difficulty":    actual_difficulty,
            "actual_vgrade":        kilter_to_vgrade(actual_difficulty) if actual_difficulty else None,
            "predicted_difficulty": round(pred_score, 2),
            "predicted_vgrade":     kilter_to_vgrade(pred_score),
            "error":                round(pred_score - actual_difficulty, 2) if actual_difficulty else None,
            "video_count":          feat["video_count"],
            "frame_count":          feat["frame_count"],
            "top_signals": {
                "avg_shoulder_l":  round(feat.get("pose_avg_shoulder_l") or 0, 1),
                "avg_tension":     round(feat.get("pose_avg_tension") or 0, 3),
                "peak_com_vel":    round(feat.get("pose_peak_com_vel") or 0, 3),
                "pct_straight_arm": round((feat.get("pose_pct_straight_l") or 0 +
                                           feat.get("pose_pct_straight_r") or 0) / 2, 3),
            },
        })

    results.sort(key=lambda r: r["actual_difficulty"] or 99)
    return jsonify({"routes": results, "total": len(results)})


@app.route("/api/pose/stats")
def pose_stats():
    """Overall pose data coverage — how many routes have video data scraped."""
    try:
        pg  = get_pg()
        cur = pg.cursor()
        cur.execute("""
            SELECT
                COUNT(*)                              AS total_frames,
                COUNT(DISTINCT attempt_id)            AS total_attempts,
                COUNT(DISTINCT climb_uuid)            AS routes_with_pose,
                COUNT(*) FILTER (WHERE climb_uuid IS NOT NULL) AS beta_frames,
                ROUND(AVG(tension_score)::numeric, 3)          AS avg_tension,
                ROUND(AVG(hip_angle_deg)::numeric, 1)          AS avg_hip_angle,
                ROUND(AVG((left_arm_reach_norm + right_arm_reach_norm) / 2)::numeric, 3) AS avg_arm_reach,
                ROUND(AVG(com_height_norm)::numeric, 3)        AS avg_com_height
            FROM pose_frames
        """)
        row = cur.fetchone()
        cur.close(); pg.close()
        return jsonify({
            "total_frames":     row[0],
            "total_attempts":   row[1],
            "routes_with_pose": row[2],
            "beta_frames":      row[3],
            "avg_tension":      float(row[4]) if row[4] is not None else None,
            "avg_hip_angle":    float(row[5]) if row[5] is not None else None,
            "avg_arm_reach":    float(row[6]) if row[6] is not None else None,
            "avg_com_height":   float(row[7]) if row[7] is not None else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pose/<climb_uuid>")
def pose_for_climb(climb_uuid):
    """
    Aggregated pose metrics for a specific Kilter climb UUID.
    Returns per-metric stats (mean, p50, p90, max) across all scraped beta videos
    for that route — useful for showing 'how people climb this route' in the UI.
    """
    try:
        pg  = get_pg()
        cur = pg.cursor()
        cur.execute("""
            SELECT
                COUNT(*)                            AS frame_count,
                COUNT(DISTINCT attempt_id)          AS video_count,
                ROUND(AVG(hip_angle_deg)::numeric, 1)          AS avg_hip_angle,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY hip_angle_deg)::numeric, 1) AS p50_hip_angle,
                ROUND(AVG(tension_score)::numeric, 3)          AS avg_tension,
                ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY tension_score)::numeric, 3) AS p90_tension,
                ROUND(AVG(com_height_norm)::numeric, 3)        AS avg_com_height,
                ROUND(AVG(shoulder_rot_deg)::numeric, 1)       AS avg_shoulder_rot,
                ROUND(AVG(left_arm_reach_norm)::numeric, 3)    AS avg_left_reach,
                ROUND(AVG(right_arm_reach_norm)::numeric, 3)   AS avg_right_reach,
                ROUND(MIN(com_height_norm)::numeric, 3)        AS min_com_height,
                ROUND(MAX(com_height_norm)::numeric, 3)        AS max_com_height
            FROM pose_frames
            WHERE UPPER(climb_uuid) = UPPER(%s)
        """, (climb_uuid,))
        row = cur.fetchone()
        cur.close(); pg.close()

        if not row or row[0] == 0:
            return jsonify({"error": "No pose data for this climb yet"}), 404

        cols = [
            "frame_count", "video_count",
            "avg_hip_angle", "p50_hip_angle",
            "avg_tension", "p90_tension",
            "avg_com_height", "avg_shoulder_rot",
            "avg_left_reach", "avg_right_reach",
            "min_com_height", "max_com_height",
        ]
        return jsonify(dict(zip(cols, (float(v) if v is not None else None for v in row))))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pose/frames/<climb_uuid>")
def pose_frames_for_climb(climb_uuid):
    """
    Per-frame pose data for a climb UUID, used to animate the stick figure.
    Returns sampled frames (every 2nd) with raw_landmarks + key metrics.
    Caps at 300 frames.
    """
    try:
        pg  = get_pg()
        cur = pg.cursor()
        cur.execute("""
            SELECT timestamp_sec, raw_landmarks,
                   hip_angle_deg, tension_score, com_height_norm,
                   elbow_l_deg, elbow_r_deg, knee_l_deg, knee_r_deg,
                   com_velocity, hand_l_velocity, hand_r_velocity
            FROM pose_frames
            WHERE UPPER(climb_uuid) = UPPER(%s)
              AND raw_landmarks IS NOT NULL
            ORDER BY attempt_id, timestamp_sec
        """, (climb_uuid,))
        rows = cur.fetchall()
        cur.close(); pg.close()

        if not rows:
            return jsonify({"error": "No pose frames for this climb"}), 404

        # Sample every 2nd frame, cap at 300
        sampled = rows[::2][:300]

        frames = []
        for row in sampled:
            ts, raw_lm_json = row[0], row[1]
            if raw_lm_json is None:
                continue
            try:
                lm = json.loads(raw_lm_json) if isinstance(raw_lm_json, str) else raw_lm_json
            except Exception:
                continue
            frames.append({
                "ts":         float(ts),
                "landmarks":  lm,
                "hip_angle":  float(row[2]) if row[2] is not None else None,
                "tension":    float(row[3]) if row[3] is not None else None,
                "com_height": float(row[4]) if row[4] is not None else None,
                "elbow_l":    float(row[5]) if row[5] is not None else None,
                "elbow_r":    float(row[6]) if row[6] is not None else None,
                "knee_l":     float(row[7]) if row[7] is not None else None,
                "knee_r":     float(row[8]) if row[8] is not None else None,
                "com_vel":    float(row[9]) if row[9] is not None else None,
            })

        return jsonify({"climb_uuid": climb_uuid, "total_frames": len(rows), "frames": frames})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return jsonify({"status": "ok", "endpoints": [
        "/api/board", "/api/routes", "/api/route/<id>",
        "/api/predict", "/api/suggest", "/api/auto_generate", "/api/stats",
        "/api/pose/stats", "/api/pose/<climb_uuid>", "/api/pose/frames/<climb_uuid>",
        "/api/pose/predictions",
        "/api/climber/benchmarks", "/api/climber/recommendations", "/api/climber/weak-points",
        "/api/gym/dashboard", "/api/gym/setting-recommendations", "/api/gym/route-performance",
    ]})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true", default=True)
    args = parser.parse_args()

    print(f"\nClimbing Intelligence API")
    print(f"  Loading board holds...")
    get_board_holds()
    print(f"  Loading ML model...")
    get_model()
    print(f"  Starting on http://localhost:{args.port}")
    print(f"  Routes API: http://localhost:{args.port}/api/routes")
    print(f"  Predict:    http://localhost:{args.port}/api/predict\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)

import random

import numpy as np
from flask import Blueprint, jsonify, request

from api.board_config import get_board_holds
from api.db import get_pg
from api.ml_engine import (
    cache_get, cache_set, classify_style, compute_features, explain_prediction,
    find_similar_routes, get_model, holds_query_vec, make_cache_key,
    route_query_vec_by_ext, score_to_grade, weight_adjust_score, _dist,
)
from api.pdscore import compute_pdscore

bp = Blueprint("predict", __name__)


@bp.route("/api/predict", methods=["POST"])
def predict():
    """
    Live grade prediction from placed holds.

    Body: {"holds": [{x_cm, y_cm, role, hold_type?}], "angle": int}
    Returns: {grade, score, confidence, dna, features}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    holds      = data.get("holds", [])
    angle      = int(data.get("angle", 40))
    climber_id = data.get("climber_id")      # optional UUID from localStorage
    profile_in = data.get("profile") or {}   # optional inline profile dict

    if len(holds) < 2:
        return jsonify({"error": "Need at least 2 holds"}), 400

    valid_roles = {"start", "hand", "foot", "finish"}
    for h in holds:
        if h.get("role") not in valid_roles:
            h["role"] = "hand"

    cache_key = make_cache_key(holds, angle)
    cached = cache_get(cache_key)
    if cached:
        return jsonify(cached)

    features = compute_features(holds, angle)
    if features is None:
        return jsonify({"error": "Need at least 2 hand holds (start/hand/finish)"}), 400

    model, q10_model, q90_model, boundaries, eval_data = get_model()
    feature_cols = eval_data["feature_cols"]

    feat_vec = np.array([[features.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if "reach" in col:
            feat_vec[0][i] = min(feat_vec[0][i], 300.0)

    score     = float(model.predict(feat_vec)[0])
    score     = max(0.0, min(1.0, score))
    q10_score = float(q10_model.predict(feat_vec)[0])
    q90_score = float(q90_model.predict(feat_vec)[0])

    grade      = score_to_grade(score, boundaries)
    grade_low  = score_to_grade(min(q10_score, score), boundaries)
    grade_high = score_to_grade(max(q90_score, score), boundaries)

    info = boundaries.get(grade, {})
    if info:
        lo, hi = info.get("lo", 0), info.get("hi", 1)
        margin = hi - lo
        dist_to_edge = min(abs(score - lo), abs(score - hi))
        confidence = round(min(dist_to_edge / (margin / 2), 1.0), 3) if margin > 0 else 0.5
    else:
        confidence = 0.5

    dna = {
        "reach":      min(100, round(features["avg_reach_cm"] / 90 * 100)),
        "sustained":  min(100, round(features["moves_count"] / 12 * 100)),
        "dynamic":    min(100, round(features["dyno_score"] * 50)),
        "lateral":    min(100, round(features["lateral_ratio"] * 100)),
        "complexity": min(100, round(features["zigzag_ratio"] * 120 + features.get("move_angle_std", 0) * 0.8)),
        "power":      min(100, round(features.get("crux_cluster_compactness", 0) * 2500 + features.get("max_reach_z", 0) * 12)),
    }

    style_tags  = classify_style(features)
    explanation = explain_prediction(feat_vec, feature_cols, boundaries, score)

    result = {
        "grade":       grade,
        "grade_low":   grade_low,
        "grade_high":  grade_high,
        "score":       round(score, 4),
        "confidence":  confidence,
        "style_tags":  style_tags,
        "explanation": explanation,
        "dna":         dna,
        "features":    {k: round(v, 3) if isinstance(v, float) else v for k, v in features.items()},
    }

    cache_set(cache_key, result)

    # ── Personal difficulty score (optional, not cached — it's per-climber) ────
    profile    = dict(profile_in)
    affinities = {}
    if climber_id:
        try:
            with get_pg() as pg:
                cur = pg.cursor()
                cur.execute(
                    "SELECT height_cm, wingspan_cm, weight_kg FROM climber_profiles WHERE id = %s",
                    (climber_id,),
                )
                row = cur.fetchone()
                if row:
                    profile.setdefault("height_cm",   row[0])
                    profile.setdefault("wingspan_cm",  row[1])
                    profile.setdefault("weight_kg",    row[2])
                cur.execute(
                    "SELECT move_type, success_rate FROM climber_move_affinity WHERE climber_id = %s",
                    (climber_id,),
                )
                affinities = {r[0]: float(r[1]) for r in cur.fetchall() if r[1] is not None}
        except Exception:
            pass

    if profile.get("height_cm") and profile.get("wingspan_cm"):
        try:
            pd = compute_pdscore(score, features, profile, affinities, boundaries)
            result["pd_personal"]  = pd["pd_personal"]
            result["pdscore"]      = pd["pdscore"]
            result["pd_modifiers"] = pd["modifiers"]
        except Exception:
            pass

    # ── Weight-adjusted grade (optional) ─────────────────────────────────────
    weight_kg = (profile.get("weight_kg") or
                 (data.get("profile") or {}).get("weight_kg"))
    if weight_kg:
        try:
            adj_score = weight_adjust_score(float(score), float(weight_kg), features)
            adj_grade = score_to_grade(adj_score, boundaries)
            result["grade_weight_adjusted"] = adj_grade
            result["score_weight_adjusted"] = round(adj_score, 4)
            result["weight_kg_used"] = float(weight_kg)
        except Exception:
            pass

    return jsonify(result)


@bp.route("/api/suggest", methods=["POST"])
def suggest():
    """
    Suggest next hand holds given placed holds and angle.
    Body: {"holds": [...], "angle": int, "count": int}
    """
    data       = request.get_json(silent=True) or {}
    holds      = data.get("holds", [])
    angle      = int(data.get("angle", 40))
    count      = min(int(data.get("count", 6)), 12)
    target_grade = data.get("target_grade", "")
    board_data = get_board_holds()

    hand_holds = [h for h in holds if h.get("role") in ("start", "hand")]
    if not hand_holds:
        return jsonify({"suggestions": []})

    last = hand_holds[-1]
    lx, ly = last.get("x_cm", 0), last.get("y_cm", 0)
    placed = {(h.get("x_cm"), h.get("y_cm")) for h in holds}

    # Grade-aware reach bands; steeper angles increase natural reach
    _grade_num = int(target_grade.replace("V", "").replace("~", "")) if target_grade and target_grade.replace("V","").replace("~","").isdigit() else 3
    if _grade_num <= 2:
        min_reach, max_reach = 20, 65
    elif _grade_num <= 5:
        min_reach, max_reach = (28, 85) if angle <= 40 else (30, 95)
    else:
        min_reach, max_reach = (35, 95) if angle <= 40 else (40, 115)

    center_x  = (board_data["x_min"] + board_data["x_max"]) / 2

    candidates = []
    for hold in board_data["holds"]:
        if hold["hold_type"] == "foot" or (hold["x"], hold["y"]) in placed:
            continue
        d = _dist(lx, ly, hold["x"], hold["y"])
        if min_reach <= d <= max_reach and hold["y"] >= ly:
            lateral_penalty = abs(hold["x"] - center_x) / 100
            # At harder grades jugs are welcome rests; at easier grades they just mean easier
            jug_adj = -0.05 if _grade_num >= 6 else 0
            hold_bonus = {"jug": jug_adj, "crimp": 0.05, "pinch": 0.1, "sloper": 0.15, "unknown": 0.2}
            score = d + lateral_penalty * 10 + hold_bonus.get(hold["hold_type"], 0.2) * 20

            dx, dy = abs(hold["x"] - lx), hold["y"] - ly
            impact = []
            if d > 65:   impact.append("big reach")
            elif d > 48: impact.append("medium reach")
            else:        impact.append("close move")
            if dx > 35:  impact.append(f"lateral +{round(dx)}cm")
            if dy > 50:  impact.append("high step")
            if hold["hold_type"] in ("crimp", "sloper", "pinch"):
                impact.append(f"{hold['hold_type']} (harder)")
            elif hold["hold_type"] == "jug":
                impact.append("jug (rest)")
            if d > 60 and dx > 25:
                impact.append("increases difficulty")

            candidates.append({**hold, "dist": round(d, 1), "score": round(score, 2), "impact": impact})

    candidates.sort(key=lambda h: h["score"])
    return jsonify({"suggestions": candidates[:count]})


@bp.route("/api/auto_generate", methods=["POST"])
def auto_generate():
    """
    Return a real community route at the requested grade.
    Body: {"grade": "V3", "angle": int, "exclude_ids": [...]}

    Returns a randomly selected real route (with holds) from the DB at that grade,
    so the grade is always accurate and the route is community-validated.
    """
    data       = request.get_json(silent=True) or {}
    grade_str  = data.get("grade", "V3").strip().upper()
    angle      = data.get("angle")       # optional filter
    exclude_ids = data.get("exclude_ids", [])  # UUIDs already shown this session
    board_data  = get_board_holds()

    try:
        with get_pg() as pg:
            import psycopg2.extras
            cur = pg.cursor(cursor_factory=psycopg2.extras.DictCursor)

            filters = ["br.community_grade = %s", "br.source = 'kilter'",
                       "br.send_count >= 100"]
            params  = [grade_str]

            if angle is not None:
                filters.append("br.board_angle_deg = %s")
                params.append(int(angle))
            if exclude_ids:
                filters.append("br.id != ALL(%s)")
                params.append(exclude_ids)

            where = " AND ".join(filters)
            cur.execute(f"""
                SELECT br.id, br.name, br.setter_name, br.community_grade,
                       br.difficulty_score, br.board_angle_deg, br.send_count,
                       br.avg_quality_rating, br.external_id
                FROM board_routes br
                WHERE {where}
                ORDER BY random()
                LIMIT 1
            """, params)
            row = cur.fetchone()

            if not row:
                # Fall back: no angle filter
                cur.execute("""
                    SELECT br.id, br.name, br.setter_name, br.community_grade,
                           br.difficulty_score, br.board_angle_deg, br.send_count,
                           br.avg_quality_rating, br.external_id
                    FROM board_routes br
                    WHERE br.community_grade = %s AND br.source = 'kilter'
                    ORDER BY random()
                    LIMIT 1
                """, (grade_str,))
                row = cur.fetchone()

            if not row:
                return jsonify({"error": f"No routes found for {grade_str}"}), 404

            route = dict(row)
            route["id"] = str(route["id"])

            # Fetch holds — try direct UUID, fall back to external_id join
            cur.execute("""
                SELECT grid_col, grid_row, position_x_cm AS x_cm, position_y_cm AS y_cm,
                       role, hand_sequence, hold_type
                FROM route_holds WHERE route_id = %s
                ORDER BY hand_sequence NULLS LAST, position_y_cm
            """, (route["id"],))
            raw = cur.fetchall()

            if not raw and route.get("external_id"):
                cur.execute("""
                    SELECT DISTINCT ON (rh.grid_col, rh.grid_row, rh.role)
                           rh.grid_col, rh.grid_row, rh.position_x_cm AS x_cm,
                           rh.position_y_cm AS y_cm, rh.role, rh.hand_sequence, rh.hold_type
                    FROM route_holds rh
                    JOIN board_routes br ON br.id = rh.route_id
                    WHERE br.external_id = %s
                    ORDER BY rh.grid_col, rh.grid_row, rh.role,
                             rh.hand_sequence NULLS LAST, rh.position_y_cm
                    LIMIT 200
                """, (route["external_id"],))
                raw = cur.fetchall()

            holds = []
            for h in raw:
                hd = dict(h)
                if hd.get("x_cm") and hd.get("y_cm"):
                    hd["x_pct"] = round((hd["x_cm"] - board_data["x_min"]) / board_data["x_range"] * 100, 2)
                    hd["y_pct"] = round(100 - (hd["y_cm"] - board_data["y_min"]) / board_data["y_range"] * 100, 2)
                holds.append(hd)

            route["holds"] = holds
            return jsonify(route)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/routes/similar", methods=["GET", "POST"])
def similar_routes():
    """
    Route similarity search using cosine distance in feature space.

    Query params:
      id    — route_id (integer) for a known DB route
      limit — number of results (max 20, default 10)

    Or POST body:
      holds — raw holds array (for Creator routes not yet in DB)
      angle — board angle
      limit — number of results
    """
    limit = min(int(request.args.get("limit", 10)), 20)
    ext_id = request.args.get("id", "").strip()  # external kilter UUID

    # Determine query vector
    exclude_ext = ""
    if ext_id:
        query_vec = route_query_vec_by_ext(ext_id)
        if query_vec is None:
            return jsonify({"error": "Route not found in similarity corpus"}), 404
        exclude_ext = ext_id
    else:
        # Accept holds from POST body (Creator mode)
        data = request.get_json(silent=True) or {}
        holds = data.get("holds", [])
        angle = int(data.get("angle", 40))
        limit = min(int(data.get("limit", limit)), 20)
        if len(holds) < 2:
            return jsonify({"error": "Provide ?id=<external_uuid> or POST {holds, angle}"}), 400
        query_vec = holds_query_vec(holds, angle)
        if query_vec is None:
            return jsonify({"error": "Could not compute features from holds"}), 400

    pairs = find_similar_routes(query_vec, exclude_ext_id=exclude_ext, limit=limit)
    if not pairs:
        return jsonify({"similar": []})

    sim_ext_ids = [p[0] for p in pairs]
    sim_scores  = {p[0]: p[1] for p in pairs}

    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("""
                SELECT id, external_id, name, setter_name, community_grade,
                       board_angle_deg, send_count, avg_quality_rating
                FROM board_routes
                WHERE UPPER(external_id) = ANY(%s) AND source = 'kilter'
            """, ([e.upper() for e in sim_ext_ids],))
            rows = {r[1].upper() if r[1] else "": r for r in cur.fetchall()}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = []
    for ext in sim_ext_ids:
        r = rows.get(ext.upper())
        if r is None:
            continue
        results.append({
            "id":         r[0],
            "external_id": r[1],
            "name":       r[2],
            "setter":     r[3],
            "grade":      r[4],
            "angle":      r[5],
            "send_count": r[6],
            "quality":    round(float(r[7]), 2) if r[7] else None,
            "similarity": round(sim_scores[ext.upper()], 4),
        })

    return jsonify({"similar": results})

import random

import numpy as np
from flask import Blueprint, jsonify, request

from api.board_config import get_board_holds
from api.db import get_pg
from api.ml_engine import (
    cache_get, cache_set, compute_features, get_model,
    make_cache_key, score_to_grade, _dist,
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

    model, boundaries, eval_data = get_model()
    feature_cols = eval_data["feature_cols"]

    feat_vec = np.array([[features.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if "reach" in col:
            feat_vec[0][i] = min(feat_vec[0][i], 300.0)

    score = float(model.predict(feat_vec)[0])
    score = max(0.0, min(1.0, score))
    grade = score_to_grade(score, boundaries)

    info = boundaries.get(grade, {})
    if info:
        lo, hi = info.get("lo", 0), info.get("hi", 1)
        margin = hi - lo
        dist_to_edge = min(abs(score - lo), abs(score - hi))
        confidence = round(min(dist_to_edge / (margin / 2), 1.0), 3) if margin > 0 else 0.5
    else:
        confidence = 0.5

    dna = {
        "reach":      min(100, round(features["avg_reach_cm"] / 100 * 100)),
        "sustained":  min(100, round(features["moves_count"] / 12 * 100)),
        "dynamic":    min(100, round(features["dyno_score"] * 50)),
        "lateral":    min(100, round(features["lateral_ratio"] * 100)),
        "complexity": min(100, round(features["zigzag_ratio"] * 120 + features.get("move_angle_std", 0) * 0.8)),
        "power":      min(100, round(features["sloper_ratio"] * 60 + features["pinch_ratio"] * 40 + features["crimp_ratio"] * 30)),
    }

    result = {
        "grade":      grade,
        "score":      round(score, 4),
        "confidence": confidence,
        "dna":        dna,
        "features":   {k: round(v, 3) if isinstance(v, float) else v for k, v in features.items()},
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
    board_data = get_board_holds()

    hand_holds = [h for h in holds if h.get("role") in ("start", "hand")]
    if not hand_holds:
        return jsonify({"suggestions": []})

    last = hand_holds[-1]
    lx, ly = last.get("x_cm", 0), last.get("y_cm", 0)
    placed = {(h.get("x_cm"), h.get("y_cm")) for h in holds}

    min_reach = 30 if angle <= 40 else 25
    max_reach = 90 if angle <= 40 else 75
    center_x  = (board_data["x_min"] + board_data["x_max"]) / 2

    candidates = []
    for hold in board_data["holds"]:
        if hold["hold_type"] == "foot" or (hold["x"], hold["y"]) in placed:
            continue
        d = _dist(lx, ly, hold["x"], hold["y"])
        if min_reach <= d <= max_reach and hold["y"] >= ly:
            lateral_penalty = abs(hold["x"] - center_x) / 100
            hold_bonus = {"jug": 0, "crimp": 0.05, "pinch": 0.1, "sloper": 0.15, "unknown": 0.2}
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


_GRADE_TO_SCORE = {
    "V0": 0.30, "V1": 0.35, "V2": 0.40, "V3": 0.45, "V4": 0.50,
    "V5": 0.55, "V6": 0.60, "V7": 0.65, "V8": 0.70, "V9": 0.74,
    "V10": 0.78, "V11": 0.82, "V12": 0.85, "V13": 0.89, "V14": 0.94,
}


@bp.route("/api/auto_generate", methods=["POST"])
def auto_generate():
    """
    Auto-generate a complete route.
    Body: {"angle": int, "difficulty": float 0–1, "grade": "V6", "hold_count": int}

    `grade` takes precedence over `difficulty` when provided.
    """
    data       = request.get_json(silent=True) or {}
    angle      = int(data.get("angle", 40))
    hold_count = min(int(data.get("hold_count", 8)), 16)

    # Grade string → difficulty float (with small random jitter so each generation differs)
    grade_str  = data.get("grade", "").strip().upper()
    if grade_str in _GRADE_TO_SCORE:
        base_diff  = _GRADE_TO_SCORE[grade_str]
        difficulty = max(0.05, min(0.95, base_diff + random.uniform(-0.03, 0.03)))
    else:
        difficulty = float(data.get("difficulty", 0.4))
    board_data = get_board_holds()

    all_holds = [h for h in board_data["holds"] if h["hold_type"] != "foot"]
    y_min, y_max = board_data["y_min"], board_data["y_max"]
    y_range   = max(y_max - y_min, 1)
    center_x  = (board_data["x_min"] + board_data["x_max"]) / 2

    # Start holds: bottom 30% of board
    bottom_holds = [h for h in all_holds if (h["y"] - y_min) / y_range < 0.30]
    if len(bottom_holds) < 4:
        bottom_holds = sorted(all_holds, key=lambda h: h["y"])[:max(20, len(all_holds) // 4)]

    target_spread = 20 + difficulty * 35
    best_pair, best_diff = None, float("inf")
    sample = random.sample(bottom_holds, min(40, len(bottom_holds)))
    for i, a in enumerate(sample):
        for b in sample[i + 1:]:
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

    reach_min = max(20, 30 - difficulty * 10)
    reach_max = min(95, 55 + difficulty * 35)
    hand_slots = hold_count - 3

    hold_bonus_diff = {"jug": -0.1, "crimp": 0.1, "pinch": 0.15, "sloper": 0.2, "unknown": 0.05}
    for _ in range(hand_slots):
        candidates = []
        for h in all_holds:
            if (h["x"], h["y"]) in placed:
                continue
            d = _dist(last_x, last_y, h["x"], h["y"])
            if reach_min <= d <= reach_max and h["y"] >= last_y:
                lateral    = abs(h["x"] - center_x) / 100
                diff_match = abs(hold_bonus_diff.get(h["hold_type"], 0.05) - difficulty * 0.15)
                score      = d * 0.4 + lateral * 8 + diff_match * 20 + random.uniform(0, 5)
                candidates.append((score, h))
        if not candidates:
            break
        candidates.sort(key=lambda t: t[0])
        pick = random.choice(candidates[:min(3, len(candidates))])[1]
        route_holds.append({**pick, "role": "hand"})
        placed.add((pick["x"], pick["y"]))
        last_x, last_y = pick["x"], pick["y"]

    top_holds = [h for h in all_holds
                 if (h["y"] - y_min) / y_range > 0.75 and (h["x"], h["y"]) not in placed]
    if top_holds:
        finish = min(top_holds, key=lambda h: abs(h["x"] - center_x) + abs(h["y"] - y_max) * 0.5)
        route_holds.append({**finish, "role": "finish"})

    return jsonify({"holds": route_holds})

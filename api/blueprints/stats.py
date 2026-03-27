import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from api.board_config import get_board_holds
from api.db import get_pg, DATA_DIR, SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY
from api.ml_engine import classify_style, compute_features, get_model, score_to_grade

bp = Blueprint("stats", __name__)

_DAILY_CHALLENGE_PATH = DATA_DIR / "daily_challenges.json"


@bp.route("/api/config")
def frontend_config():
    """
    Public config endpoint — returns Supabase URL and publishable key so the
    frontend can initialise the Supabase JS client without hardcoding values.
    Only publishable (non-secret) credentials are exposed here.
    """
    return jsonify({
        "supabase_url":  SUPABASE_URL,
        "supabase_key":  SUPABASE_PUBLISHABLE_KEY,
    })


@bp.route("/api/stats")
def stats():
    """Quick DB counts for the UI header — route totals and pose coverage."""
    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("SELECT COUNT(*) FROM board_routes WHERE source='kilter'")
            n_routes = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM board_routes WHERE source='kilter' AND community_grade IS NOT NULL")
            n_graded = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT climb_uuid), COUNT(*) FROM pose_frames WHERE climb_uuid IS NOT NULL")
            pose_row = cur.fetchone()

        try:
            _, _, _, _, eval_data = get_model()
            model_accuracy = f"{round(eval_data.get('grade_within1_acc', 0) * 100, 1)}% ±1"
        except Exception:
            model_accuracy = None

        return jsonify({
            "total_routes":   n_routes,
            "graded_routes":  n_graded,
            "grade_range":    "V0–V14",
            "hold_count":     get_board_holds()["count"],
            "pose_climbs":    pose_row[0],
            "pose_frames":    pose_row[1],
            "model_accuracy": model_accuracy,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/health/probe")
def health_probe():
    """
    Synthetic probe: runs 4 checks to catch silent failures.
    Returns {ok, checks, latency_ms}.
    """
    results = {}
    latency = {}

    # ── Check 1: predict a V3-ish 3-hold route ────────────────────────────────
    _V3_HOLDS = [
        {"x_cm": 56, "y_cm": 24, "role": "start"},
        {"x_cm": 64, "y_cm": 72, "role": "hand"},
        {"x_cm": 72, "y_cm": 112, "role": "finish"},
    ]
    try:
        t0 = time.monotonic()
        model, q10_m, q90_m, boundaries, eval_data = get_model()
        feats = compute_features(_V3_HOLDS, 40)
        fcols = eval_data["feature_cols"]
        fv = np.array([[feats.get(c, 0.0) for c in fcols]], dtype=np.float32)
        score = float(model.predict(fv)[0])
        grade = score_to_grade(score, boundaries)
        sorted_grades = sorted(boundaries, key=lambda g: boundaries[g]["mean"])
        idx = {g: i for i, g in enumerate(sorted_grades)}
        v3_idx = idx.get("V3", 3)
        ok_range = abs(idx.get(grade, 0) - v3_idx) <= 2  # V1–V5 acceptable
        results["predict_v3"] = "pass" if ok_range else f"fail (got {grade}, expected V1-V5)"
        latency["predict_v3_ms"] = round((time.monotonic() - t0) * 1000, 1)
    except Exception as e:
        results["predict_v3"] = f"error: {e}"
        latency["predict_v3_ms"] = -1

    # ── Check 2: predict a V7-ish 7-hold route ────────────────────────────────
    _V7_HOLDS = [
        {"x_cm": 40, "y_cm": 16, "role": "start"},
        {"x_cm": 88, "y_cm": 40, "role": "hand"},
        {"x_cm": 32, "y_cm": 64, "role": "hand"},
        {"x_cm": 96, "y_cm": 88, "role": "hand"},
        {"x_cm": 24, "y_cm": 104, "role": "hand"},
        {"x_cm": 104, "y_cm": 128, "role": "hand"},
        {"x_cm": 64, "y_cm": 152, "role": "finish"},
    ]
    try:
        t0 = time.monotonic()
        feats7 = compute_features(_V7_HOLDS, 40)
        fv7 = np.array([[feats7.get(c, 0.0) for c in fcols]], dtype=np.float32)
        score7 = float(model.predict(fv7)[0])
        grade7 = score_to_grade(score7, boundaries)
        v7_idx = idx.get("V7", 7)
        ok7 = abs(idx.get(grade7, 0) - v7_idx) <= 3  # V4–V10 acceptable
        results["predict_v7"] = "pass" if ok7 else f"fail (got {grade7}, expected V4-V10)"
        latency["predict_v7_ms"] = round((time.monotonic() - t0) * 1000, 1)
    except Exception as e:
        results["predict_v7"] = f"error: {e}"
        latency["predict_v7_ms"] = -1

    # ── Check 3: DB connectivity ──────────────────────────────────────────────
    try:
        t0 = time.monotonic()
        with get_pg() as pg:
            pg.cursor().execute("SELECT 1")
        results["db"] = "pass"
        latency["db_ms"] = round((time.monotonic() - t0) * 1000, 1)
    except Exception as e:
        results["db"] = f"fail: {e}"
        latency["db_ms"] = -1

    # ── Check 4: suggest endpoint sanity (at least 1 board hold exists) ──────
    try:
        t0 = time.monotonic()
        board = get_board_holds()
        results["board"] = "pass" if board["count"] > 100 else f"fail (only {board['count']} holds)"
        latency["board_ms"] = round((time.monotonic() - t0) * 1000, 1)
    except Exception as e:
        results["board"] = f"error: {e}"
        latency["board_ms"] = -1

    all_pass = all(v == "pass" for v in results.values())
    return jsonify({"ok": all_pass, "checks": results, "latency_ms": latency})


# ── Daily Challenge ───────────────────────────────────────────────────────────

def _load_challenges() -> dict:
    if _DAILY_CHALLENGE_PATH.exists():
        with open(_DAILY_CHALLENGE_PATH) as f:
            return json.load(f)
    return {}


def _save_challenges(data: dict) -> None:
    _DAILY_CHALLENGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_DAILY_CHALLENGE_PATH, "w") as f:
        json.dump(data, f, indent=2)


@bp.route("/api/daily_challenge")
def daily_challenge():
    """
    GET /api/daily_challenge?grade=V4

    Returns today's pre-generated challenge route for the requested grade.
    If no challenge exists for today (or grade), returns 404.
    The ?all=1 flag returns all of today's challenges across grades.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    challenges = _load_challenges()

    if request.args.get("all"):
        day_data = challenges.get(today, {})
        if not day_data:
            return jsonify({"error": "No challenges generated for today yet"}), 404
        return jsonify({"date": today, "challenges": day_data})

    grade = (request.args.get("grade") or "V4").strip().upper()
    day_data = challenges.get(today, {})
    route = day_data.get(grade)
    if not route:
        return jsonify({"error": f"No {grade} challenge for {today}"}), 404

    return jsonify({"date": today, "grade": grade, **route})


@bp.route("/api/daily_challenge/generate", methods=["POST"])
def generate_daily_challenge():
    """
    POST /api/daily_challenge/generate

    Generates today's challenges for a set of target grades by randomly
    selecting a high-quality community route from the DB.

    Body: {"grades": ["V2", "V4", "V6"], "admin_secret": "..."}
    Idempotent: running twice on the same day keeps the first run's routes.
    """
    import os
    secret = (request.get_json(silent=True) or {}).get("admin_secret", "")
    if secret != os.environ.get("ADMIN_SECRET", ""):
        return jsonify({"error": "unauthorized"}), 401

    data   = request.get_json(silent=True) or {}
    grades = data.get("grades") or ["V2", "V4", "V6"]
    today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    challenges = _load_challenges()
    day_data   = challenges.get(today, {})
    generated  = {}

    try:
        with get_pg() as pg:
            import psycopg2.extras
            cur = pg.cursor(cursor_factory=psycopg2.extras.DictCursor)

            for grade in grades:
                grade = grade.strip().upper()
                if grade in day_data:
                    generated[grade] = "already_exists"
                    continue

                cur.execute("""
                    SELECT br.id, br.name, br.setter_name, br.community_grade,
                           br.board_angle_deg, br.send_count, br.avg_quality_rating
                    FROM board_routes br
                    WHERE br.community_grade = %s
                      AND br.source = 'kilter'
                      AND br.send_count >= 500
                      AND br.avg_quality_rating >= 2.0
                    ORDER BY random()
                    LIMIT 1
                """, (grade,))
                row = cur.fetchone()
                if not row:
                    generated[grade] = "not_found"
                    continue

                route = dict(row)
                route["id"] = str(route["id"])

                # Fetch holds
                cur.execute("""
                    SELECT position_x_cm AS x_cm, position_y_cm AS y_cm, role, hand_sequence, hold_type
                    FROM route_holds WHERE route_id = %s
                    ORDER BY hand_sequence NULLS LAST, position_y_cm
                """, (route["id"],))
                route["holds"] = [dict(h) for h in cur.fetchall()]

                day_data[grade]  = route
                generated[grade] = "created"

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if day_data:
        challenges[today] = day_data
        _save_challenges(challenges)

    return jsonify({"date": today, "generated": generated})

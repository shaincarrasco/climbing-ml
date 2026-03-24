from flask import Blueprint, jsonify, request

from api.db import get_pg

bp = Blueprint("climber", __name__)

# V-grade → difficulty_score midpoints (used for target grade lookup)
_GRADE_SCORES = {
    "V0": 0.30, "V1": 0.35, "V2": 0.40, "V3": 0.45, "V4": 0.50,
    "V5": 0.55, "V6": 0.60, "V7": 0.65, "V8": 0.70, "V9": 0.74,
    "V10": 0.78, "V11": 0.82, "V12": 0.85, "V13": 0.89, "V14": 0.94,
}
_GRADES = list(_GRADE_SCORES)


@bp.route("/api/climber/benchmarks")
def climber_benchmarks():
    """
    Body mechanics benchmarks by V-grade.
    Median hip angle, tension, arm reach, and straight-arm % per grade band.
    """
    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("""
                SELECT
                    br.community_grade,
                    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pf.hip_angle_deg)::numeric, 1)   AS p50_hip_angle,
                    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pf.tension_score)::numeric, 3)   AS p50_tension,
                    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                        (pf.left_arm_reach_norm + pf.right_arm_reach_norm) / 2)::numeric, 3)           AS p50_arm_reach,
                    ROUND(AVG(CASE WHEN pf.is_straight_arm_l OR pf.is_straight_arm_r THEN 1.0 ELSE 0.0 END)::numeric, 3) AS pct_straight_arm,
                    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pf.com_height_norm)::numeric, 3) AS p50_com_height,
                    ROUND(AVG(pf.hip_spread_deg)::numeric, 1)                                          AS avg_hip_spread,
                    COUNT(DISTINCT pf.climb_uuid)                                                        AS route_count,
                    COUNT(*)                                                                              AS frame_count
                FROM pose_frames pf
                JOIN board_routes br ON br.external_id = pf.climb_uuid
                WHERE br.community_grade IS NOT NULL
                  AND pf.hip_angle_deg   IS NOT NULL
                  AND pf.tension_score   IS NOT NULL
                GROUP BY br.community_grade
                ORDER BY MIN(br.difficulty_score)
            """)
            rows = cur.fetchall()

        cols = ["grade", "hip_angle", "tension", "arm_reach", "pct_straight_arm",
                "com_height", "hip_spread", "route_count", "frame_count"]
        return jsonify({
            "benchmarks": [dict(zip(cols, row)) for row in rows]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/climber/recommendations", methods=["POST"])
def climber_recommendations():
    """
    Personalised route recommendations.

    Body: {current_grade, target_grade?, angle?, weakness?, limit?}
    Returns routes at the target grade, prioritised by pose availability + quality.
    """
    data          = request.get_json(silent=True) or {}
    current_grade = data.get("current_grade", "V4")
    target_grade  = data.get("target_grade")
    angle         = data.get("angle")
    weakness      = data.get("weakness")
    limit         = min(int(data.get("limit", 10)), 30)

    if not target_grade:
        idx          = _GRADES.index(current_grade) if current_grade in _GRADES else 4
        target_grade = _GRADES[min(idx + 1, len(_GRADES) - 1)]

    target_score = _GRADE_SCORES.get(target_grade, 0.55)
    score_lo     = max(0.0, target_score - 0.04)
    score_hi     = min(1.0, target_score + 0.04)

    try:
        angle_filter = "AND br.board_angle_deg = %s" if angle else ""
        angle_params = [angle] if angle else []

        with get_pg() as pg:
            cur = pg.cursor()
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

        routes = []
        for rid, ext_id, name, grade, ang, sends, quality, has_pose in rows:
            routes.append({
                "id":         rid,
                "climb_uuid": ext_id,
                "name":       name or "Unnamed",
                "grade":      grade,
                "angle":      ang,
                "send_count": sends,
                "quality":    round(float(quality), 2) if quality else None,
                "has_pose":   has_pose,
                "why":        _recommendation_reason(grade, current_grade, weakness, has_pose),
            })

        routes.sort(key=lambda r: (not r["has_pose"], -(r["quality"] or 0)))
        return jsonify({
            "current_grade": current_grade,
            "target_grade":  target_grade,
            "weakness":      weakness,
            "routes":        routes[:limit],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/climber/weak-points", methods=["POST"])
def climber_weak_points():
    """
    Identify technique gaps between current grade and target grade.
    Body: {current_grade, target_grade}
    """
    data          = request.get_json(silent=True) or {}
    current_grade = data.get("current_grade", "V4")
    target_grade  = data.get("target_grade", "V5")

    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("""
                SELECT
                    br.community_grade,
                    ROUND(AVG(pf.hip_angle_deg)::numeric, 1)                                          AS avg_hip_angle,
                    ROUND(AVG(pf.tension_score)::numeric, 3)                                          AS avg_tension,
                    ROUND(AVG((pf.left_arm_reach_norm+pf.right_arm_reach_norm)/2)::numeric, 3)        AS avg_reach,
                    ROUND(AVG(CASE WHEN pf.is_straight_arm_l OR pf.is_straight_arm_r THEN 1.0 ELSE 0.0 END)::numeric, 3) AS pct_straight,
                    ROUND(AVG(pf.com_height_norm)::numeric, 3)                                        AS avg_com,
                    ROUND(AVG(pf.hip_spread_deg)::numeric, 1)                                         AS avg_hip_spread,
                    ROUND(AVG(ABS(pf.left_arm_reach_norm - pf.right_arm_reach_norm))::numeric, 3)     AS reach_asymmetry
                FROM pose_frames pf
                JOIN board_routes br ON br.external_id = pf.climb_uuid
                WHERE br.community_grade = ANY(%s)
                  AND pf.hip_angle_deg IS NOT NULL
                GROUP BY br.community_grade
            """, [[current_grade, target_grade]])
            rows = cur.fetchall()

        by_grade = {}
        for row in rows:
            by_grade[row[0]] = {
                "hip_angle":       row[1],
                "tension":         row[2],
                "arm_reach":       row[3],
                "pct_straight":    row[4],
                "com_height":      row[5],
                "hip_spread":      row[6],
                "reach_asymmetry": row[7],
            }

        current = by_grade.get(current_grade, {})
        target  = by_grade.get(target_grade, {})
        if not current or not target:
            return jsonify({"error": f"Not enough pose data for {current_grade} or {target_grade} yet"}), 404

        DIMENSIONS = [
            ("tension",         "Body Tension",          +1,
             "core tightness across the climb — harder routes demand more lock-off strength"),
            ("arm_reach",       "Arm Extension",         +1,
             "how far arms extend relative to torso — longer reach = more efficient movement"),
            ("hip_spread",      "Hip Spread",            +1,
             "how open your hips stay — wider spread = better footwork and balance"),
            ("com_height",      "Center of Mass Height", +1,
             "how high you stay on the wall — higher COM = better body positioning"),
            ("pct_straight",    "Straight-Arm Resting",  +1,
             "resting on straight arms saves energy — harder routes require more of this"),
            ("reach_asymmetry", "L/R Reach Balance",     -1,
             "lower asymmetry = more balanced technique (negative gap = good)"),
        ]

        weak_points = []
        for key, label, direction, desc in DIMENSIONS:
            c_val, t_val = current.get(key), target.get(key)
            if c_val is None or t_val is None:
                continue
            gap      = float(t_val) - float(c_val)
            severity = min(100, round(abs(gap) * direction * 200))
            weak_points.append({
                "dimension":   key,
                "label":       label,
                "description": desc,
                "current_val": float(c_val),
                "target_val":  float(t_val),
                "gap":         round(gap, 3),
                "severity":    severity,
                "needs_more":  (gap * direction) > 0,
            })

        weak_points.sort(key=lambda w: -w["severity"])
        return jsonify({
            "current_grade":  current_grade,
            "target_grade":   target_grade,
            "weak_points":    weak_points,
            "data_available": bool(current and target),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def _recommendation_reason(grade: str, current_grade: str, weakness: str | None, has_pose: bool) -> str:
    labels = {
        "dynamic":    "trains explosive movement",
        "lateral":    "builds lateral reach",
        "reach":      "extends your max reach",
        "power":      "develops contact strength",
        "sustained":  "builds endurance",
        "complexity": "trains sequencing",
    }
    parts = ["Consolidate your " + grade if grade == current_grade else "Push into " + grade]
    if weakness:
        parts.append(labels.get(weakness, "targets your weakness"))
    if has_pose:
        parts.append("beta video available")
    return " · ".join(parts)

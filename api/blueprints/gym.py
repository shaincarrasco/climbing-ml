from flask import Blueprint, jsonify, request

from api.db import get_pg

bp = Blueprint("gym", __name__)

_IDEAL_PCT = {
    "V0": 3, "V1": 5, "V2": 6, "V3": 12, "V4": 15,
    "V5": 15, "V6": 13, "V7": 10, "V8": 9, "V9": 5,
    "V10": 3, "V11": 2, "V12": 1, "V13": 0.5, "V14": 0.5,
}


@bp.route("/api/gym/dashboard")
def gym_dashboard():
    """Grade distribution, top routes, and pose coverage for gym operators."""
    try:
        with get_pg() as pg:
            cur = pg.cursor()

            cur.execute("""
                SELECT community_grade,
                       COUNT(*)                                   AS route_count,
                       ROUND(AVG(send_count)::numeric, 0)         AS avg_sends,
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

            cur.execute("""
                SELECT br.community_grade, COUNT(DISTINCT pf.climb_uuid) AS climbs_with_pose
                FROM board_routes br
                JOIN pose_frames pf ON pf.climb_uuid = br.external_id
                WHERE br.community_grade IS NOT NULL
                GROUP BY br.community_grade
                ORDER BY MIN(br.difficulty_score)
            """)
            pose_cov = {r[0]: r[1] for r in cur.fetchall()}
            for g in grade_dist:
                g["pose_routes"] = pose_cov.get(g["grade"], 0)

            cur.execute("""
                SELECT COUNT(*), COUNT(*) FILTER (WHERE send_count >= 5)
                FROM board_routes WHERE source='kilter'
            """)
            total, active = cur.fetchone()

        return jsonify({
            "total_routes":       total,
            "active_routes":      active,
            "grade_distribution": grade_dist,
            "top_routes":         top_routes,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/gym/setting-recommendations")
def setting_recommendations():
    """What grades and styles the board needs more of, compared to an ideal distribution."""
    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("""
                SELECT community_grade, COUNT(*) AS n,
                       ROUND(AVG(difficulty_score)::numeric, 4) AS avg_score
                FROM board_routes
                WHERE community_grade IS NOT NULL AND source = 'kilter' AND send_count >= 5
                GROUP BY community_grade
                ORDER BY MIN(difficulty_score)
            """)
            rows = cur.fetchall()

        if not rows:
            return jsonify({"recommendations": []})

        total = sum(r[1] for r in rows)
        dist  = {r[0]: r[1] for r in rows}

        recs = []
        for grade, ideal_pct in _IDEAL_PCT.items():
            actual_pct = dist.get(grade, 0) / total * 100 if total else 0
            gap_pct    = ideal_pct - actual_pct
            if gap_pct > 2.0:
                recs.append({
                    "grade":        grade,
                    "actual_pct":   round(actual_pct, 1),
                    "ideal_pct":    ideal_pct,
                    "gap":          round(gap_pct, 1),
                    "actual_count": dist.get(grade, 0),
                    "action":       f"Set {max(1, round(gap_pct / 100 * total))} more {grade} routes",
                    "priority":     "high" if gap_pct > 5 else "medium",
                })

        recs.sort(key=lambda r: -r["gap"])
        return jsonify({"total_active_routes": total, "recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/gym/route-performance")
def route_performance():
    """Which routes are getting sends and quality ratings. Filterable by grade and angle."""
    grade = request.args.get("grade")
    angle = request.args.get("angle", type=int)
    limit = min(request.args.get("limit", 20, type=int), 100)

    filters = ["source = 'kilter'", "send_count >= 5"]
    params  = []
    if grade:
        filters.append("community_grade = %s"); params.append(grade)
    if angle:
        filters.append("board_angle_deg = %s"); params.append(angle)

    where = "WHERE " + " AND ".join(filters)

    try:
        with get_pg() as pg:
            cur = pg.cursor()
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

        return jsonify({
            "routes": [
                {
                    "id":       r[0], "name":  r[1] or "Unnamed",
                    "grade":    r[2], "angle": r[3],
                    "sends":    r[4],
                    "quality":  round(float(r[5]), 2) if r[5] else None,
                    "setter":   r[6],
                    "has_pose": r[7],
                }
                for r in rows
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

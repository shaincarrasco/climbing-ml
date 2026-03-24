import json

import psycopg2.extras
from flask import Blueprint, jsonify, request

from api.board_config import get_board_holds
from api.db import get_pg

bp = Blueprint("routes", __name__)


@bp.route("/api/routes")
def routes():
    """
    Return routes from board_routes.
    Query params: grade, angle, min_sends, limit (≤200), offset, q (name search), has_pose
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
        filters.append(
            "EXISTS (SELECT 1 FROM pose_frames pf "
            "WHERE UPPER(pf.climb_uuid) = UPPER(br.external_id) LIMIT 1)"
        )

    where = " AND ".join(filters)

    try:
        with get_pg() as pg:
            cur = pg.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute(f"SELECT COUNT(*) FROM board_routes br WHERE {where}", params)
            total = cur.fetchone()[0]

            cur.execute(f"""
                SELECT br.id, br.name, br.setter_name, br.community_grade, br.difficulty_score,
                       br.board_angle_deg, br.send_count, br.avg_quality_rating, br.external_id,
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
            r["board_angle_deg"]    = int(r["board_angle_deg"])    if r["board_angle_deg"]    else None
            r["difficulty_score"]   = round(float(r["difficulty_score"]), 4) if r["difficulty_score"]   else None
            r["avg_quality_rating"] = round(float(r["avg_quality_rating"]), 2) if r["avg_quality_rating"] else None
            r["has_pose"] = bool(r.get("has_pose", False))

        return jsonify({"total": total, "routes": rows, "limit": limit, "offset": offset})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/route/<route_id>")
def route_detail(route_id):
    """Return a single route with all its holds and display percentages."""
    board_data  = get_board_holds()

    try:
        with get_pg() as pg:
            cur = pg.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute("""
                SELECT id, name, setter_name, community_grade, difficulty_score,
                       board_angle_deg, send_count, avg_quality_rating, external_id
                FROM board_routes WHERE id = %s
            """, (route_id,))
            row = cur.fetchone()
            if not row:
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
                if hd["position_x_cm"] and hd["position_y_cm"]:
                    hd["x_pct"] = round((hd["position_x_cm"] - board_data["x_min"]) / board_data["x_range"] * 100, 2)
                    hd["y_pct"] = round(100 - (hd["position_y_cm"] - board_data["y_min"]) / board_data["y_range"] * 100, 2)
                holds.append(hd)

            route["holds"] = holds

        return jsonify(route)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/routes/saved", methods=["GET"])
def get_saved_routes():
    """Return all user-saved routes."""
    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("""
                SELECT id, name, angle, predicted_grade, confidence, hold_count,
                       holds_json, created_at
                FROM saved_routes
                ORDER BY created_at DESC
                LIMIT 100
            """)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

        routes = []
        for row in rows:
            r = dict(zip(cols, row))
            if r.get("holds_json"):
                try:
                    r["holds"] = json.loads(r["holds_json"])
                except Exception:
                    r["holds"] = []
            del r["holds_json"]
            if r.get("created_at"):
                r["created_at"] = str(r["created_at"])
            routes.append(r)

        return jsonify({"routes": routes, "total": len(routes)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/routes/saved", methods=["POST"])
def save_route():
    """Save a user-created route."""
    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS saved_routes (
                    id              SERIAL PRIMARY KEY,
                    name            TEXT,
                    angle           INT DEFAULT 40,
                    predicted_grade TEXT,
                    confidence      FLOAT,
                    hold_count      INT,
                    holds_json      TEXT,
                    created_at      TIMESTAMP DEFAULT NOW()
                )
            """)
            body  = request.get_json(force=True)
            name  = body.get("name") or "Untitled Route"
            angle = int(body.get("angle") or 40)
            grade = body.get("predicted_grade") or ""
            conf  = float(body.get("confidence") or 0)
            holds = body.get("holds") or []

            cur.execute("""
                INSERT INTO saved_routes (name, angle, predicted_grade, confidence, hold_count, holds_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (name, angle, grade, conf, len(holds), json.dumps(holds)))
            new_id = cur.fetchone()[0]
            pg.commit()

        return jsonify({"id": new_id, "saved": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/routes/saved/<int:route_id>", methods=["DELETE"])
def delete_saved_route(route_id):
    """Delete a saved route by ID."""
    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("DELETE FROM saved_routes WHERE id = %s", (route_id,))
            pg.commit()
        return jsonify({"deleted": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from flask import Blueprint, jsonify

from api.board_config import get_board_holds
from api.db import get_pg, SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY

bp = Blueprint("stats", __name__)


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

        return jsonify({
            "total_routes":  n_routes,
            "graded_routes": n_graded,
            "grade_range":   "V0–V14",
            "hold_count":    get_board_holds()["count"],
            "pose_climbs":   pose_row[0],
            "pose_frames":   pose_row[1],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

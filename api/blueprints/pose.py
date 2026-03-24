import json
import os
import sys
import tempfile
import uuid as _uuid

import sqlite3
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from api.db import DATA_DIR, MODEL_DIR, SQLITE_PATH, get_pg

bp = Blueprint("pose", __name__)


@bp.route("/api/pose/stats")
def pose_stats():
    """Overall pose data coverage — frames, attempts, and aggregate body metrics."""
    try:
        with get_pg() as pg:
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


@bp.route("/api/pose/correlations")
def pose_correlations():
    """
    Pearson correlations between pose metrics and difficulty_score + per-grade benchmarks.
    Serves pre-computed data/pose_correlations.json when available; falls back to live queries.
    """
    corr_cache = DATA_DIR / "pose_correlations.json"
    if corr_cache.exists():
        try:
            payload = json.loads(corr_cache.read_text())
            payload["source"] = "cached"
            return jsonify(payload)
        except Exception:
            pass

    try:
        with get_pg() as pg:
            cur = pg.cursor()

            cur.execute("""
                SELECT
                    br.community_grade,
                    MIN(br.difficulty_score)                                                AS diff_score,
                    ROUND(AVG(pf.hip_angle_deg)::numeric, 1)                               AS avg_hip_angle,
                    ROUND(AVG(pf.tension_score)::numeric, 3)                               AS avg_tension,
                    ROUND(AVG(pf.com_height_norm)::numeric, 3)                             AS avg_com_height,
                    ROUND(AVG((pf.left_arm_reach_norm+pf.right_arm_reach_norm)/2)::numeric,3) AS avg_reach,
                    ROUND(AVG(pf.hip_spread_deg)::numeric, 1)                              AS avg_hip_spread,
                    ROUND(AVG(pf.elbow_l_deg)::numeric, 1)                                 AS avg_elbow_l,
                    ROUND(AVG(pf.elbow_r_deg)::numeric, 1)                                 AS avg_elbow_r,
                    ROUND(AVG(pf.knee_l_deg)::numeric, 1)                                  AS avg_knee,
                    ROUND(AVG(pf.com_velocity)::numeric, 4)                                AS avg_com_vel,
                    COUNT(DISTINCT pf.attempt_id)                                           AS videos,
                    COUNT(*)                                                                AS frames
                FROM pose_frames pf
                JOIN board_routes br ON UPPER(br.external_id) = UPPER(pf.climb_uuid)
                WHERE br.community_grade IS NOT NULL
                  AND br.difficulty_score IS NOT NULL
                GROUP BY br.community_grade
                HAVING COUNT(*) >= 20
                ORDER BY MIN(br.difficulty_score)
            """)
            grade_rows = cur.fetchall()
            grade_cols = [d[0] for d in cur.description]
            grade_benchmarks = [dict(zip(grade_cols, row)) for row in grade_rows]
            for row in grade_benchmarks:
                for k, v in row.items():
                    if hasattr(v, "__float__"):
                        row[k] = float(v)

            METRICS = [
                "hip_angle_deg", "tension_score", "com_height_norm",
                "left_arm_reach_norm", "right_arm_reach_norm",
                "hip_spread_deg", "elbow_l_deg", "elbow_r_deg",
                "knee_l_deg", "knee_r_deg", "com_velocity", "shoulder_rot_deg",
            ]
            correlations = {}
            for metric in METRICS:
                cur.execute(f"""
                    SELECT CORR(pf.{metric}, br.difficulty_score)
                    FROM pose_frames pf
                    JOIN board_routes br ON UPPER(br.external_id) = UPPER(pf.climb_uuid)
                    WHERE pf.{metric} IS NOT NULL AND br.difficulty_score IS NOT NULL
                """)
                val = cur.fetchone()[0]
                correlations[metric] = round(float(val), 4) if val is not None else None

        sorted_corr = sorted(
            [{"metric": k, "r": v, "abs_r": abs(v) if v else 0} for k, v in correlations.items()],
            key=lambda x: x["abs_r"], reverse=True,
        )
        return jsonify({
            "correlations":     sorted_corr,
            "grade_benchmarks": grade_benchmarks,
            "source":           "live",
            "note": "r = Pearson correlation with difficulty_score. |r|>0.3 = meaningful signal for model.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/pose/<climb_uuid>")
def pose_for_climb(climb_uuid):
    """
    Aggregated pose metrics for a specific Kilter climb UUID.
    Returns per-metric stats across all scraped beta videos for that route.
    """
    try:
        with get_pg() as pg:
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

        if not row or row[0] == 0:
            return jsonify({"error": "No pose data for this climb yet"}), 404

        cols = [
            "frame_count", "video_count",
            "avg_hip_angle", "p50_hip_angle",
            "avg_tension",   "p90_tension",
            "avg_com_height","avg_shoulder_rot",
            "avg_left_reach","avg_right_reach",
            "min_com_height","max_com_height",
        ]
        return jsonify(dict(zip(cols, (float(v) if v is not None else None for v in row))))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/pose/frames/<climb_uuid>")
def pose_frames_for_climb(climb_uuid):
    """
    Per-frame pose data for stick figure animation.
    Returns sampled frames (every 2nd, capped at 300) with raw landmarks + key metrics.
    """
    try:
        with get_pg() as pg:
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

        if not rows:
            return jsonify({"error": "No pose frames for this climb"}), 404

        sampled = rows[::2][:300]
        frames  = []
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
                "hip_angle":  float(row[2])  if row[2]  is not None else None,
                "tension":    float(row[3])  if row[3]  is not None else None,
                "com_height": float(row[4])  if row[4]  is not None else None,
                "elbow_l":    float(row[5])  if row[5]  is not None else None,
                "elbow_r":    float(row[6])  if row[6]  is not None else None,
                "knee_l":     float(row[7])  if row[7]  is not None else None,
                "knee_r":     float(row[8])  if row[8]  is not None else None,
                "com_vel":    float(row[9])  if row[9]  is not None else None,
            })

        return jsonify({"climb_uuid": climb_uuid, "total_frames": len(rows), "frames": frames})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/pose/predictions")
def pose_predictions():
    """
    Pose-model predictions for every route with video data.
    Shows predicted vs actual grade using the pose-only sklearn model.
    """
    project_root = str(MODEL_DIR.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from ml.pose_difficulty_model import load_model, kilter_to_vgrade
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        pipe, feature_cols, col_medians = load_model()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute(f"""
                SELECT {agg_cols}
                FROM pose_frames
                WHERE climb_uuid IS NOT NULL
                  AND hip_angle_deg IS NOT NULL
                  AND elbow_l_deg   IS NOT NULL
                GROUP BY climb_uuid
            """)
            rows      = cur.fetchall()
            col_names = [d[0] for d in cur.description]

        uuids = [r[0] for r in rows]
        with sqlite3.connect(str(SQLITE_PATH)) as con:
            diff_rows = con.execute(f"""
                SELECT cs.climb_uuid, c.name, c.setter_username,
                       cs.difficulty_average, cs.quality_average, cs.ascensionist_count
                FROM climb_stats cs JOIN climbs c ON c.uuid = cs.climb_uuid
                WHERE cs.climb_uuid IN ({','.join('?'*len(uuids))})
                GROUP BY cs.climb_uuid HAVING difficulty_average = MAX(cs.difficulty_average)
            """, uuids).fetchall()
        diff_lookup = {r[0]: r for r in diff_rows}

        results = []
        for row in rows:
            feat       = dict(zip(col_names, row))
            c_uuid     = feat["climb_uuid"]
            row_vec    = [feat.get(col, col_medians.get(col, 0.0)) or col_medians.get(col, 0.0)
                          for col in feature_cols]
            pred_score = max(14.0, min(33.0, float(pipe.predict([row_vec])[0])))
            dl         = diff_lookup.get(c_uuid)

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
                    "avg_shoulder_l":   round(feat.get("pose_avg_shoulder_l") or 0, 1),
                    "avg_tension":      round(feat.get("pose_avg_tension") or 0, 3),
                    "peak_com_vel":     round(feat.get("pose_peak_com_vel") or 0, 3),
                    "pct_straight_arm": round((feat.get("pose_pct_straight_l") or 0 +
                                               feat.get("pose_pct_straight_r") or 0) / 2, 3),
                },
            })

        results.sort(key=lambda r: r["actual_difficulty"] or 99)
        return jsonify({"routes": results, "total": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Video Upload + Coaching ───────────────────────────────────────────────────

_ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
_MAX_UPLOAD_MB      = 200


@bp.route("/api/pose/upload", methods=["POST"])
def upload_video():
    """
    Upload a climbing video for pose extraction + coaching analysis.

    Multipart form fields:
      video             — video file (required)
      climber_id        — UUID from localStorage (optional)
      board_angle_deg   — int, default 40 (optional)
      self_reported_grade — e.g. 'V6' (optional, improves coaching benchmarks)

    Returns immediately with {session_id, status: 'processing'}.
    Poll GET /api/pose/session/<session_id> for results.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file attached (field name: 'video')"}), 400

    f         = request.files["video"]
    filename  = secure_filename(f.filename or "upload.mp4")
    ext       = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_VIDEO_EXTS:
        return jsonify({"error": f"Unsupported format {ext}. Use mp4/mov/avi."}), 400

    # Check file size before saving
    f.seek(0, 2)
    size_bytes = f.tell()
    f.seek(0)
    if size_bytes > _MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify({"error": f"File too large (max {_MAX_UPLOAD_MB}MB)"}), 413

    session_id  = str(_uuid.uuid4())
    climber_id  = request.form.get("climber_id")
    angle       = int(request.form.get("board_angle_deg", 40))
    grade       = request.form.get("self_reported_grade", "").strip() or None

    # Save to temp file
    tmp_dir  = tempfile.mkdtemp(prefix="climb_upload_")
    tmp_path = os.path.join(tmp_dir, f"{session_id}{ext}")
    f.save(tmp_path)

    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("""
                INSERT INTO video_upload_sessions
                    (id, climber_id, original_filename, file_size_bytes,
                     board_angle_deg, self_reported_grade, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            """, (session_id, climber_id, filename, size_bytes, angle, grade))
            pg.commit()
    except Exception as e:
        return jsonify({"error": f"DB error: {e}"}), 500

    # Process in background thread so the HTTP response returns immediately
    import threading
    t = threading.Thread(
        target=_process_upload,
        args=(session_id, tmp_path, angle, grade, climber_id),
        daemon=True,
    )
    t.start()

    return jsonify({
        "session_id": session_id,
        "status":     "processing",
        "message":    "Upload received. Poll /api/pose/session/" + session_id + " for results.",
    })


@bp.route("/api/pose/session/<session_id>")
def get_session(session_id):
    """Poll for upload processing status and coaching results."""
    try:
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("""
                SELECT status, overall_verdict, summary_text, frames_extracted,
                       processed_at, error_message, self_reported_grade,
                       board_angle_deg, original_filename
                FROM video_upload_sessions WHERE id = %s
            """, (session_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Session not found"}), 404

            status = row[0]
            result = {
                "session_id":       session_id,
                "status":           status,
                "overall_verdict":  row[1],
                "summary":          row[2],
                "frames_extracted": row[3],
                "processed_at":     row[4].isoformat() if row[4] else None,
                "error":            row[5],
                "grade":            row[6],
                "angle":            row[7],
                "filename":         row[8],
                "insights":         [],
            }

            if status == "done":
                cur.execute("""
                    SELECT category, severity, message, score, benchmark, drills
                    FROM coaching_insights WHERE session_id = %s
                    ORDER BY
                        CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END
                """, (session_id,))
                result["insights"] = [
                    {"category": r[0], "severity": r[1], "message": r[2],
                     "score": r[3], "benchmark": r[4], "drills": r[5] or []}
                    for r in cur.fetchall()
                ]

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _process_upload(session_id: str, video_path: str, angle: int,
                    grade: str | None, climber_id: str | None):
    """Background worker: extract pose frames and run coaching analysis."""
    import traceback
    attempt_id = session_id  # reuse session UUID as the attempt grouping key

    def _set_status(status, **kwargs):
        try:
            with get_pg() as pg:
                cur = pg.cursor()
                sets = ["status = %s"] + [f"{k} = %s" for k in kwargs]
                vals = [status] + list(kwargs.values()) + [session_id]
                cur.execute(
                    f"UPDATE video_upload_sessions SET {', '.join(sets)} WHERE id = %s",
                    vals,
                )
                pg.commit()
        except Exception:
            pass

    _set_status("processing")

    try:
        # ── Step 1: Run pose extractor ─────────────────────────────────────
        project_root = str(DATA_DIR.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from pipeline.pose_extractor import process_video
        n_frames = process_video(video_path, attempt_id=attempt_id, climb_uuid=None)

        # ── Step 2: Fetch extracted frames ────────────────────────────────
        with get_pg() as pg:
            cur = pg.cursor()
            cur.execute("""
                SELECT tension_score, com_height_norm, hip_angle_deg,
                       is_straight_arm_l, is_straight_arm_r,
                       left_arm_reach_norm, right_arm_reach_norm,
                       hip_spread_deg
                FROM pose_frames WHERE attempt_id = %s
            """, (attempt_id,))
            frame_rows = [
                {
                    "tension_score":        r[0],
                    "com_height_norm":      r[1],
                    "hip_angle_deg":        r[2],
                    "is_straight_arm_l":    r[3],
                    "is_straight_arm_r":    r[4],
                    "left_arm_reach_norm":  r[5],
                    "right_arm_reach_norm": r[6],
                    "hip_spread_deg":       r[7],
                }
                for r in cur.fetchall()
            ]

        # ── Step 3: Run coaching analysis ─────────────────────────────────
        from api.video_coach import analyse_attempt
        analysis = analyse_attempt(frame_rows, grade=grade)

        # ── Step 4: Persist insights + update session ─────────────────────
        with get_pg() as pg:
            cur = pg.cursor()
            for insight in analysis["insights"]:
                cur.execute("""
                    INSERT INTO coaching_insights
                        (session_id, category, severity, message, score, benchmark, drills)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    session_id,
                    insight["category"],
                    insight["severity"],
                    insight["message"],
                    insight.get("score"),
                    insight.get("benchmark"),
                    insight.get("drills", []),
                ))
            cur.execute("""
                UPDATE video_upload_sessions
                SET status = 'done',
                    frames_extracted = %s,
                    overall_verdict  = %s,
                    summary_text     = %s,
                    attempt_id       = %s,
                    processed_at     = NOW()
                WHERE id = %s
            """, (n_frames, analysis["overall"], analysis["summary"], attempt_id, session_id))
            pg.commit()

    except Exception:
        err = traceback.format_exc()
        _set_status("failed", error_message=err[:1000])
    finally:
        # Always delete the video file — only pose data is retained
        try:
            os.remove(video_path)
            os.rmdir(os.path.dirname(video_path))
        except Exception:
            pass

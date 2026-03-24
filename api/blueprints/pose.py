import json
import os
import sys

import sqlite3
from flask import Blueprint, jsonify, request

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

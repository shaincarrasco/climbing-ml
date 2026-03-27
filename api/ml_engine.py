"""
api/ml_engine.py
----------------
Model loading, feature computation, and prediction caching for the difficulty model.

Loaded once at startup; shared across all blueprint handlers.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from api.db import MODEL_DIR, DATA_DIR

# ── Human-readable feature labels for SHAP explanations ──────────────────────

FEATURE_LABELS = {
    "avg_reach_cm":              "Average reach between holds",
    "max_reach_cm":              "Longest single move",
    "avg_hand_to_foot_cm":       "Hand-to-foot distance",
    "board_angle_deg":           "Board angle",
    "total_hold_count":          "Number of holds",
    "moves_count":               "Move count",
    "dyno_score":                "Dynamic/explosive moves",
    "dyno_flag":                 "Contains a dyno",
    "zigzag_ratio":              "Direction changes (technical)",
    "direction_changes":         "Direction reversals",
    "crux_cluster_compactness":  "Crux hold proximity",
    "max_reach_z":               "Outlier big move",
    "lateral_ratio":             "Lateral movement ratio",
    "vertical_ratio":            "Vertical movement ratio",
    "height_span_cm":            "Route height span",
    "lateral_span_cm":           "Route width span",
    "width_height_ratio":        "Width-to-height ratio",
    "reach_cv":                  "Move-length variability",
    "path_linearity":            "Path straightness",
    "avg_foot_spread_cm":        "Foot spread",
    "avg_foot_hand_x_offset_cm": "Foot-hand lateral offset",
    "hold_density":              "Hold density",
    "finish_height_pct":         "Finish hold height",
    "start_height_pct":          "Start hold height",
    "hand_hold_count":           "Number of hand holds",
    "foot_hold_count":           "Number of foot holds",
    "foot_ratio":                "Foot hold ratio",
    "zone_transitions":          "Wall zone transitions",
    "seq_angle_avg":             "Average move angle",
    "max_consec_vert_gain_cm":   "Max consecutive vertical gain",
}


# ── SHAP explainer (lazy-loaded) ──────────────────────────────────────────────

_shap_explainer = None


def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is not None:
        return _shap_explainer
    try:
        import shap as shap_lib
        model, _, _, _, _ = get_model()
        _shap_explainer = shap_lib.TreeExplainer(model)
        return _shap_explainer
    except ImportError:
        return None
    except Exception:
        return None


def explain_prediction(feat_vec: np.ndarray, feature_cols: list,
                       boundaries: dict, score: float, top_n: int = 5) -> list:
    """
    Return top-N SHAP contributors as [{feature, label, direction, delta_grades}].
    Returns empty list if shap is not installed.
    """
    explainer = get_shap_explainer()
    if explainer is None:
        return []

    sorted_grades = sorted(boundaries, key=lambda g: boundaries[g]["mean"])
    grade_scores  = [boundaries[g]["mean"] for g in sorted_grades]

    shap_values = explainer.shap_values(feat_vec)[0]  # shape: (n_features,)

    pairs = sorted(enumerate(shap_values), key=lambda x: abs(x[1]), reverse=True)

    result = []
    for idx, sv in pairs[:top_n]:
        col   = feature_cols[idx]
        label = FEATURE_LABELS.get(col, col.replace("_", " "))
        # Convert shap value (in difficulty_score units) to approximate grade delta
        # 1.0 difficulty_score spans ~14 grades (V0–V14)
        delta_grades = round(float(sv) * 14, 2)
        direction = "harder" if sv > 0 else "easier"
        result.append({
            "feature":      col,
            "label":        label,
            "direction":    direction,
            "delta_grades": abs(delta_grades),
            "shap_value":   round(float(sv), 5),
        })

    return result


# ── Model state (loaded once) ─────────────────────────────────────────────────

_model:      Optional[xgb.XGBRegressor] = None
_q10_model:  Optional[xgb.XGBRegressor] = None
_q90_model:  Optional[xgb.XGBRegressor] = None
_boundaries: Optional[dict]             = None
_eval_data:  Optional[dict]             = None


def get_model() -> tuple[xgb.XGBRegressor, xgb.XGBRegressor, xgb.XGBRegressor, dict, dict]:
    global _model, _q10_model, _q90_model, _boundaries, _eval_data
    if _model is None:
        _model = xgb.XGBRegressor()
        _model.load_model(str(MODEL_DIR / "difficulty_model.xgb"))
        # Quantile models for confidence bands (q10 = optimistic, q90 = pessimistic)
        _q10_model = xgb.XGBRegressor()
        _q10_model.load_model(str(MODEL_DIR / "difficulty_model_q10.xgb"))
        _q90_model = xgb.XGBRegressor()
        _q90_model.load_model(str(MODEL_DIR / "difficulty_model_q90.xgb"))
        with open(MODEL_DIR / "grade_boundaries.json") as f:
            _boundaries = json.load(f)
        with open(MODEL_DIR / "evaluation.json") as f:
            _eval_data = json.load(f)
    return _model, _q10_model, _q90_model, _boundaries, _eval_data


# ── Similarity feature matrix (lazy-loaded for /api/routes/similar) ───────────

_sim_matrix:    Optional[np.ndarray] = None   # shape (N, n_features), L2-normalised
_sim_ext_ids:   Optional[list]       = None   # external UUID strings matching matrix rows


def get_similarity_matrix() -> tuple[Optional[np.ndarray], Optional[list]]:
    """
    Return (L2-normalised feature matrix, list of external_id strings).
    Built from routes_features_kilter.csv using the same feature_cols the model uses.
    Loaded once; subsequent calls return cached arrays.
    """
    global _sim_matrix, _sim_ext_ids
    if _sim_matrix is not None and _sim_ext_ids is not None:
        return _sim_matrix, _sim_ext_ids

    csv_path = DATA_DIR / "routes_features_kilter.csv"
    if not csv_path.exists():
        return None, None

    try:
        import pandas as pd
        _, _, _, _, eval_data = get_model()
        feature_cols = eval_data["feature_cols"]

        df = pd.read_csv(csv_path)
        if "route_id" not in df.columns:
            return None, None

        # Align to model feature columns, fill missing with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        df[feature_cols] = df[feature_cols].fillna(0.0)

        # Clip reach outliers (same as training)
        for col in feature_cols:
            if "reach" in col:
                df[col] = df[col].clip(upper=300.0)

        mat = df[feature_cols].values.astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms   # L2-normalise: dot product = cosine similarity

        ext_ids = df["route_id"].astype(str).str.upper().tolist()
        _sim_matrix  = mat
        _sim_ext_ids = ext_ids
        print(f"[similarity] feature matrix loaded: {mat.shape}")
        return _sim_matrix, _sim_ext_ids
    except Exception as e:
        print(f"[similarity] failed to build feature matrix: {e}")
        return None, None


def find_similar_routes(query_vec: np.ndarray, exclude_ext_id: str = "",
                        limit: int = 10) -> list[tuple[str, float]]:
    """
    Given an L2-normalised query feature vector, return (external_id, similarity) pairs
    sorted by descending cosine similarity.
    """
    mat, ext_ids = get_similarity_matrix()
    if mat is None or ext_ids is None:
        return []

    sims = (mat @ query_vec).ravel()
    if exclude_ext_id:
        excl_upper = exclude_ext_id.upper()
        for i, eid in enumerate(ext_ids):
            if eid == excl_upper:
                sims[i] = -1.0

    top_idxs = np.argsort(sims)[::-1][:limit]
    return [(ext_ids[i], float(sims[i])) for i in top_idxs]


def route_query_vec_by_ext(ext_id: str) -> Optional[np.ndarray]:
    """Return the L2-normalised feature vector for a known external_id, or None."""
    mat, ext_ids = get_similarity_matrix()
    if mat is None or ext_ids is None:
        return None
    target = ext_id.upper()
    for i, eid in enumerate(ext_ids):
        if eid == target:
            return mat[i]
    return None


def holds_query_vec(holds: list, angle: int) -> Optional[np.ndarray]:
    """Compute and L2-normalise a feature vector from a raw holds array."""
    _, _, _, _, eval_data = get_model()
    feature_cols = eval_data["feature_cols"]
    features = compute_features(holds, angle)
    if features is None:
        return None
    vec = np.array([features.get(c, 0.0) for c in feature_cols], dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if "reach" in col:
            vec[i] = min(vec[i], 300.0)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


# ── Prediction cache ──────────────────────────────────────────────────────────

_prediction_cache: dict = {}
_CACHE_MAX = 20_000


def cache_get(key: str) -> Optional[dict]:
    return _prediction_cache.get(key)


def cache_set(key: str, value: dict) -> None:
    if len(_prediction_cache) >= _CACHE_MAX:
        del _prediction_cache[next(iter(_prediction_cache))]
    _prediction_cache[key] = value


def make_cache_key(holds: list, angle: int) -> str:
    payload = sorted(
        [(h.get("x_cm", 0), h.get("y_cm", 0), h.get("role", "")) for h in holds],
        key=str,
    ) + [angle]
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ── Grade utilities ───────────────────────────────────────────────────────────

def score_to_grade(score: float, boundaries: dict) -> str:
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


# ── Feature computation ───────────────────────────────────────────────────────

def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _safe_mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _safe_std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    m = _safe_mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


_DYNO_WEIGHTS = {0:2.8, 20:2.6, 35:2.3, 40:2.2, 45:2.1, 50:2.0,
                 55:1.95, 60:1.9, 65:1.85, 70:1.8}


def _dyno_threshold(angle: int) -> float:
    angles = sorted(_DYNO_WEIGHTS)
    if angle <= angles[0]:  return _DYNO_WEIGHTS[angles[0]]
    if angle >= angles[-1]: return _DYNO_WEIGHTS[angles[-1]]
    for i in range(len(angles) - 1):
        if angles[i] <= angle <= angles[i + 1]:
            lo, hi = angles[i], angles[i + 1]
            t = (angle - lo) / (hi - lo)
            return _DYNO_WEIGHTS[lo] + t * (_DYNO_WEIGHTS[hi] - _DYNO_WEIGHTS[lo])
    return 1.8


def compute_features(holds: list, angle: int) -> Optional[dict]:
    """
    Compute all ML features from a list of hold dicts.
    Each hold: {x_cm, y_cm, role, hold_type (optional)}
    Returns feature dict or None if not enough holds.
    """
    if not holds or len(holds) < 2:
        return None

    hand_holds = [h for h in holds if h.get("role") in ("start", "hand", "finish")]
    foot_holds  = [h for h in holds if h.get("role") == "foot"]
    if len(hand_holds) < 2:
        return None

    sequenced = [h for h in hand_holds if h.get("hand_sequence") is not None]
    hand_seq = (sorted(sequenced, key=lambda h: h["hand_sequence"])
                if len(sequenced) >= 2
                else sorted(hand_holds, key=lambda h: h.get("y_cm") or 0))

    reaches, laterals, verticals, move_angles = [], [], [], []
    direction_changes, prev_lat_sign = 0, None

    for i in range(1, len(hand_seq)):
        h1, h2 = hand_seq[i - 1], hand_seq[i]
        x1, y1 = h1.get("x_cm") or 0, h1.get("y_cm") or 0
        x2, y2 = h2.get("x_cm") or 0, h2.get("y_cm") or 0
        d = _dist(x1, y1, x2, y2)
        reaches.append(d)
        laterals.append(abs(x2 - x1))
        verticals.append(abs(y2 - y1))
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

    foot_spreads = []
    if len(foot_holds) >= 2:
        for i in range(1, len(foot_holds)):
            f1, f2 = foot_holds[i - 1], foot_holds[i]
            foot_spreads.append(_dist(
                f1.get("x_cm") or 0, f1.get("y_cm") or 0,
                f2.get("x_cm") or 0, f2.get("y_cm") or 0,
            ))

    hand_to_foot_dists, foot_hand_x_offsets = [], []
    for hh in hand_seq:
        hx, hy = hh.get("x_cm") or 0, hh.get("y_cm") or 0
        if foot_holds:
            nf = min(foot_holds, key=lambda f: _dist(hx, hy, f.get("x_cm") or 0, f.get("y_cm") or 0))
            hand_to_foot_dists.append(_dist(hx, hy, nf.get("x_cm") or 0, nf.get("y_cm") or 0))
        feet_below = [f for f in foot_holds if (f.get("y_cm") or 0) < hy]
        if feet_below:
            nb = min(feet_below, key=lambda f: _dist(hx, hy, f.get("x_cm") or 0, f.get("y_cm") or 0))
            foot_hand_x_offsets.append(abs(hx - (nb.get("x_cm") or 0)))

    all_xs = [h.get("x_cm") or 0 for h in hand_seq]
    all_ys = [h.get("y_cm") or 0 for h in hand_seq]
    all_hx = [h.get("x_cm") or 0 for h in holds]
    all_hy = [h.get("y_cm") or 0 for h in holds]
    height_span  = max(all_ys) - min(all_ys)
    lateral_span = max(all_xs) - min(all_xs)

    max_reach_z    = (max(reaches) - avg_reach) / reach_std if reach_std > 0 else 0
    dyno_threshold = _dyno_threshold(angle)
    dyno_ratio     = max(reaches) / avg_reach if avg_reach > 0 else 0
    dyno_score     = round(dyno_ratio / dyno_threshold, 4)
    dyno_flag      = 1 if (dyno_ratio >= dyno_threshold and max_reach_z >= 2.0 and max(reaches) > 150) else 0

    n_hand = len(hand_holds)
    n_foot = len(foot_holds)
    avg_v  = avg_reach if avg_reach > 0 else 1

    # ── Sequence angle features ───────────────────────────────────────────────
    seq_angles = []
    for i in range(1, len(hand_seq)):
        x1 = hand_seq[i-1].get("x_cm") or 0
        y1 = hand_seq[i-1].get("y_cm") or 0
        x2 = hand_seq[i].get("x_cm") or 0
        y2 = hand_seq[i].get("y_cm") or 0
        dx, dy = x2 - x1, y2 - y1
        seq_angles.append(math.degrees(math.atan2(dx, dy if dy != 0 else 0.001)))
    seq_angle_avg = _safe_mean(seq_angles)
    seq_angle_std = _safe_std(seq_angles)

    # ── Wall zone features ────────────────────────────────────────────────────
    _MID_X, _MID_Y = 70.0, 70.0  # board midpoints (140cm board)
    zone_counts = {"BL": 0, "BR": 0, "TL": 0, "TR": 0}
    zone_transitions = 0
    prev_zone = None
    for h in hand_seq:
        hx = h.get("x_cm") or 0
        hy = h.get("y_cm") or 0
        z = ("T" if hy > _MID_Y else "B") + ("R" if hx > _MID_X else "L")
        zone_counts[z] += 1
        if prev_zone is not None and z != prev_zone:
            zone_transitions += 1
        prev_zone = z

    # ── Crux cluster compactness ──────────────────────────────────────────────
    if len(hand_seq) >= 3:
        min_mean_dist = float("inf")
        for i in range(len(hand_seq) - 2):
            cluster = hand_seq[i:i + 3]
            pairs = [
                _dist(cluster[a].get("x_cm") or 0, cluster[a].get("y_cm") or 0,
                      cluster[b].get("x_cm") or 0, cluster[b].get("y_cm") or 0)
                for a in range(3) for b in range(a + 1, 3)
            ]
            md = _safe_mean(pairs)
            if md < min_mean_dist:
                min_mean_dist = md
        crux_cluster_compactness = round(1.0 / (min_mean_dist + 1.0), 6)
    elif len(hand_seq) == 2:
        d2 = _dist(hand_seq[0].get("x_cm") or 0, hand_seq[0].get("y_cm") or 0,
                   hand_seq[1].get("x_cm") or 0, hand_seq[1].get("y_cm") or 0)
        crux_cluster_compactness = round(1.0 / (d2 + 1.0), 6)
    else:
        crux_cluster_compactness = 0.0

    # ── Path efficiency ───────────────────────────────────────────────────────
    total_path = sum(reaches)
    if total_path > 0:
        euclidean = _dist(
            hand_seq[0].get("x_cm") or 0, hand_seq[0].get("y_cm") or 0,
            hand_seq[-1].get("x_cm") or 0, hand_seq[-1].get("y_cm") or 0,
        )
        path_efficiency = round(euclidean / total_path, 4)
    else:
        path_efficiency = 0.0

    # ── Max consecutive vertical gain ─────────────────────────────────────────
    max_consec_vert_gain_cm = 0.0
    for i in range(1, len(hand_seq)):
        dy = (hand_seq[i].get("y_cm") or 0) - (hand_seq[i-1].get("y_cm") or 0)
        if dy > max_consec_vert_gain_cm:
            max_consec_vert_gain_cm = dy
    max_consec_vert_gain_cm = round(max_consec_vert_gain_cm, 2)

    hand_typed = [h for h in hand_holds if h.get("hold_type") and h["hold_type"] not in ("unknown", "foot")]
    n_typed    = len(hand_typed)

    def _tr(t):
        return round(sum(1 for h in hand_typed if h.get("hold_type") == t) / n_typed, 4) if n_typed > 0 else 0.0

    all_typed   = [h for h in holds if h.get("hold_type") and h["hold_type"] not in ("unknown",)]
    typed_ratio = round(len(all_typed) / len(holds), 4) if holds else 0.0

    features = {
        "board_angle_deg":           float(angle),
        "hand_hold_count":           n_hand,
        "foot_hold_count":           n_foot,
        "total_hold_count":          len(holds),
        "moves_count":               moves,
        "foot_ratio":                round(n_foot / len(holds), 4) if holds else 0,
        "hand_foot_ratio":           round(n_hand / n_foot, 4) if n_foot > 0 else float(n_hand),
        "avg_reach_cm":              round(avg_reach, 2),
        "max_reach_cm":              round(max(reaches), 2),
        "min_reach_cm":              round(min(reaches), 2),
        "reach_std_cm":              round(reach_std, 2),
        "reach_range_cm":            round(max(reaches) - min(reaches), 2),
        "reach_cv":                  round(reach_cv, 4),
        "avg_lateral_cm":            round(_safe_mean(laterals), 2),
        "avg_vertical_cm":           round(_safe_mean(verticals), 2),
        "max_lateral_cm":            round(max(laterals), 2),
        "max_vertical_cm":           round(max(verticals), 2),
        "vertical_ratio":            round(_safe_mean(verticals) / avg_v, 4),
        "lateral_ratio":             round(_safe_mean(laterals) / avg_v, 4),
        "avg_foot_spread_cm":        round(_safe_mean(foot_spreads), 2),
        "avg_hand_to_foot_cm":       round(_safe_mean(hand_to_foot_dists), 2),
        "avg_foot_hand_x_offset_cm": round(_safe_mean(foot_hand_x_offsets), 2),
        "max_foot_hand_x_offset_cm": round(max(foot_hand_x_offsets) if foot_hand_x_offsets else 0, 2),
        "height_span_cm":            round(height_span, 2),
        "lateral_span_cm":           round(lateral_span, 2),
        "width_height_ratio":        round(lateral_span / height_span, 4) if height_span > 0 else 0,
        "hold_density":              round(len(holds) / (height_span * lateral_span) * 100, 6) if height_span > 0 and lateral_span > 0 else 0,
        "direction_changes":         direction_changes,
        "zigzag_ratio":              round(direction_changes / moves, 4) if moves > 0 else 0,
        "dyno_score":                dyno_score,
        "dyno_flag":                 dyno_flag,
        "max_reach_z":               round(max_reach_z, 4),
        "start_x_cm":                round(hand_seq[0].get("x_cm") or 0, 2),
        "start_y_cm":                round(hand_seq[0].get("y_cm") or 0, 2),
        "finish_x_cm":               round(hand_seq[-1].get("x_cm") or 0, 2),
        "finish_y_cm":               round(hand_seq[-1].get("y_cm") or 0, 2),
        "start_height_pct":          round((hand_seq[0].get("y_cm") or 0) / 140.0, 4),
        "finish_height_pct":         round((hand_seq[-1].get("y_cm") or 0) / 140.0, 4),
        "centroid_x_cm":             round(_safe_mean(all_xs), 2),
        "centroid_y_cm":             round(_safe_mean(all_ys), 2),
        "total_centroid_x_cm":       round(_safe_mean(all_hx), 2),
        "total_centroid_y_cm":       round(_safe_mean(all_hy), 2),
        "hold_spread_x":             round(_safe_std(all_xs), 2),
        "hold_spread_y":             round(_safe_std(all_ys), 2),
        "crimp_ratio":               _tr("crimp"),
        "sloper_ratio":              _tr("sloper"),
        "jug_ratio":                 _tr("jug"),
        "pinch_ratio":               _tr("pinch"),
        "typed_ratio":               typed_ratio,
        "avg_move_angle_deg":        round(_safe_mean(move_angles), 2),
        "max_move_angle_deg":        round(max(move_angles) if move_angles else 0, 2),
        "move_angle_std":            round(_safe_std(move_angles), 2),
        "crux_reach_ratio":          round(max(reaches) / (_safe_mean(reaches) or 1), 4),
        "path_linearity":            round(
            (hand_seq[-1].get("y_cm", 0) - hand_seq[0].get("y_cm", 0)) /
            sum(reaches) if sum(reaches) > 0 else 0, 4),
        "seq_angle_avg":             round(seq_angle_avg, 4),
        "seq_angle_std":             round(seq_angle_std, 4),
        "zone_bl_count":             zone_counts["BL"],
        "zone_br_count":             zone_counts["BR"],
        "zone_tl_count":             zone_counts["TL"],
        "zone_tr_count":             zone_counts["TR"],
        "zone_transitions":          zone_transitions,
        "crux_cluster_compactness":  crux_cluster_compactness,
        "path_efficiency":           path_efficiency,
        "max_consec_vert_gain_cm":   max_consec_vert_gain_cm,
    }
    return features


# ── Body-weight grade adjustment ─────────────────────────────────────────────

_BASELINE_WEIGHT_KG = 70.0  # community grade baseline (approximate avg)
_WEIGHT_PENALTY_PER_KG = 0.08 / 70.0  # 8% grade delta per 70kg over baseline


def weight_adjust_score(score: float, weight_kg: float, features: dict) -> float:
    """
    Return a weight-adjusted difficulty score.
    Only applied to crimp/technical routes — dynos and jug routes are weight-neutral.
    Score is clamped to [0, 1].
    """
    if weight_kg is None or weight_kg <= 0:
        return score

    delta_kg = weight_kg - _BASELINE_WEIGHT_KG
    if abs(delta_kg) < 2:
        return score  # within 2kg of baseline — no adjustment

    # Route type: determine if weight-sensitive
    dyno_score     = features.get("dyno_score", 0)
    avg_reach      = features.get("avg_reach_cm", 50)
    crux_compact   = features.get("crux_cluster_compactness", 0)
    hold_density   = features.get("hold_density", 0)

    # Dynos and jumpy routes: weight-neutral (power-to-weight not grip-strength limited)
    if dyno_score > 0.5:
        return score

    # Jug/high-density routes: low weight penalty
    if hold_density > 0.15 and avg_reach < 45:
        weight_factor = 0.3
    elif crux_compact > 0.02 and avg_reach < 48:
        # Crimp/compression: full weight penalty
        weight_factor = 1.0
    else:
        weight_factor = 0.6

    # Penalty: positive delta_kg makes route harder (higher score)
    penalty = delta_kg * _WEIGHT_PENALTY_PER_KG * weight_factor
    adjusted = score + penalty
    return float(max(0.0, min(1.0, adjusted)))


# ── Style classification ───────────────────────────────────────────────────────

def classify_style(features: dict) -> list:
    """
    Rule-based style tag classifier. Returns a list of 0–3 style tags
    derived entirely from the feature vector — no trained model required.

    Tags: dynamic, technical, endurance, compression, slab
    """
    tags = []

    dyno_score     = features.get("dyno_score", 0)
    dyno_flag      = features.get("dyno_flag", 0)
    zigzag_ratio   = features.get("zigzag_ratio", 0)
    direction_chg  = features.get("direction_changes", 0)
    avg_reach      = features.get("avg_reach_cm", 0)
    moves          = features.get("moves_count", 0)
    path_lin       = features.get("path_linearity", 0)
    compactness    = features.get("crux_cluster_compactness", 0)
    width_h_ratio  = features.get("width_height_ratio", 0)
    board_angle    = features.get("board_angle_deg", 40)
    avg_lateral    = features.get("avg_lateral_cm", 0)
    max_lateral    = features.get("max_lateral_cm", 0)

    # Dynamic: big move relative to average reach, or explicit dyno
    if dyno_flag == 1 or dyno_score > 0.7:
        tags.append("dynamic")

    # Technical: lots of direction changes, short precise moves
    if zigzag_ratio > 0.4 and avg_reach < 50 and direction_chg >= 2:
        tags.append("technical")

    # Endurance / pumpy: many moves in a relatively straight line upward
    if moves >= 8 and path_lin > 0.5 and "dynamic" not in tags:
        tags.append("endurance")

    # Compression: narrow route, tight cluster of crux holds
    if width_h_ratio < 0.5 and compactness > 0.018 and "technical" not in tags:
        tags.append("compression")

    # Slab: low angle with high lateral movement (balance-dependent)
    if board_angle < 15 and avg_lateral > 20:
        tags.append("slab")

    # Span-dependent: requires wide reach between holds
    if avg_lateral > 55 or max_lateral > 75:
        if "technical" not in tags and "dynamic" not in tags:
            tags.append("span")

    return tags[:3]  # cap at 3 tags


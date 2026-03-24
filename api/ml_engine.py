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

from api.db import MODEL_DIR

# ── Model state (loaded once) ─────────────────────────────────────────────────

_model:      Optional[xgb.XGBRegressor] = None
_boundaries: Optional[dict]             = None
_eval_data:  Optional[dict]             = None


def get_model() -> tuple[xgb.XGBRegressor, dict, dict]:
    global _model, _boundaries, _eval_data
    if _model is None:
        _model = xgb.XGBRegressor()
        _model.load_model(str(MODEL_DIR / "difficulty_model.xgb"))
        with open(MODEL_DIR / "grade_boundaries.json") as f:
            _boundaries = json.load(f)
        with open(MODEL_DIR / "evaluation.json") as f:
            _eval_data = json.load(f)
    return _model, _boundaries, _eval_data


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

    hand_typed = [h for h in hand_holds if h.get("hold_type") and h["hold_type"] not in ("unknown", "foot")]
    n_typed    = len(hand_typed)

    def _tr(t):
        return round(sum(1 for h in hand_typed if h.get("hold_type") == t) / n_typed, 4) if n_typed > 0 else 0.0

    all_typed   = [h for h in holds if h.get("hold_type") and h["hold_type"] not in ("unknown",)]
    typed_ratio = round(len(all_typed) / len(holds), 4) if holds else 0.0

    return {
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
    }

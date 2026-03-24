"""
api/pdscore.py
--------------
Personal Difficulty Score (PDScore) — adjusts the ML grade prediction based on
a specific climber's body dimensions and move-type success history.

Formula
-------
  base_score   = XGBoost difficulty_score (0–1 float)
  body_delta   = Σ feature_weight × normalised_body_delta
  move_delta   = Σ move_weight × (1 - climber_success_rate_on_type)
  pdscore      = clamp(base_score + body_delta + move_delta, 0.0, 1.0)

Body modifiers
--------------
  Reach advantage: routes with big avg_reach are easier for long-wingspan climbers
  Weight penalty:  overhanging routes are harder for heavier climbers (angle > 40°)
  Height:          tall climbers can skip moves (lower effective move count)

Move modifiers
--------------
  Each route is classified into move types based on its features.
  A climber with low success on that type gets a positive delta (harder for them).
  A climber with high success gets a small negative delta (easier for them).
"""

from __future__ import annotations

from typing import Optional


# ── Body modifier weights ─────────────────────────────────────────────────────
# Each entry: (feature_key, weight, direction)
# direction=+1 means "larger value → harder for this climber given their body"
# direction=-1 means "larger value → easier for this climber given their body"

_BODY_MODIFIERS = [
    # Reach: route avg_reach_cm vs climber effective reach
    # Effective reach ≈ wingspan_cm * 0.44 (arm length fraction)
    # positive ape_index → easier big moves
    ("avg_reach_cm",      0.0003,  +1),   # per cm of reach above climber's span
    # Overhang weight penalty: each kg over 70 adds ~0.001 per 10° angle
    ("board_angle_deg",   0.00015, +1),   # steeper + heavier = harder
    # Sustained routes: taller climbers skip moves
    ("moves_count",      -0.0015,  +1),   # each move easier with height advantage
]

# ── Move type classifiers ─────────────────────────────────────────────────────
# Each returns True if a route's features match this style.
# Used to look up the climber's success rate on that type.

def _classify_moves(features: dict) -> list[str]:
    """Return list of move types that apply to this route."""
    types = []
    angle = features.get("board_angle_deg", 40)

    if features.get("dyno_flag", 0):
        types.append("dyno")
    if features.get("pose_avg_tension", features.get("avg_tension", 0)) > 0.65:
        types.append("high_tension")
    if features.get("crimp_ratio", 0) > 0.40:
        types.append("crimp")
    if features.get("sloper_ratio", 0) > 0.30:
        types.append("sloper")
    if features.get("pinch_ratio", 0) > 0.25:
        types.append("pinch")
    if features.get("lateral_ratio", 0) > 0.50:
        types.append("lateral")
    if features.get("moves_count", 0) > 9:
        types.append("sustained")
    if features.get("zigzag_ratio", 0) > 0.30:
        types.append("coordination")
    if 35 <= angle <= 44:
        types.append("overhang_40")
    elif 45 <= angle <= 54:
        types.append("overhang_50")
    elif angle >= 55:
        types.append("overhang_60")

    return types or ["general"]


# ── Body delta computation ────────────────────────────────────────────────────

def _body_delta(features: dict, profile: dict) -> float:
    """
    Compute how much harder/easier this route is for this climber's body.
    Returns a signed delta to add to the base difficulty_score.

    profile keys: height_cm, wingspan_cm, weight_kg (all optional)
    """
    height_cm   = profile.get("height_cm")   or 175.0
    wingspan_cm = profile.get("wingspan_cm") or height_cm  # neutral ape index if missing
    weight_kg   = profile.get("weight_kg")   or 70.0

    ape_index = wingspan_cm - height_cm  # negative = short arms, positive = long arms
    reach_cm  = wingspan_cm * 0.44       # effective one-arm reach

    delta = 0.0

    # Reach: each cm the route's avg_reach exceeds your reach adds difficulty
    avg_reach = features.get("avg_reach_cm", 60)
    reach_gap = avg_reach - reach_cm   # positive = route needs more reach than you have
    delta += reach_gap * 0.0006        # ~+0.03 per 50cm gap (half a V-grade)

    # Ape index bonus: long arms help on big moves
    if avg_reach > 55:
        delta -= ape_index * 0.0008    # +5cm wingspan = -0.004 on big-move routes

    # Weight penalty on steep angles
    angle = features.get("board_angle_deg", 40)
    if angle >= 40:
        weight_excess = max(0, weight_kg - 70)
        steepness     = (angle - 40) / 30          # 0 at 40°, 1 at 70°
        delta += weight_excess * steepness * 0.0012

    # Height advantage on sustained routes (tall climbers can skip moves)
    moves = features.get("moves_count", 5)
    if height_cm > 175 and moves > 7:
        height_bonus = (height_cm - 175) / 30      # 0 at 175cm, 1 at 205cm
        delta -= height_bonus * moves * 0.001

    return round(delta, 4)


# ── Move delta computation ────────────────────────────────────────────────────

def _move_delta(features: dict, affinities: dict[str, float]) -> float:
    """
    Compute difficulty modifier from personal move success rates.
    affinities: {move_type → success_rate (0–1)}
    """
    move_types = _classify_moves(features)

    # Weights per move type — how much the success rate matters
    _WEIGHTS = {
        "dyno":        0.060,
        "high_tension":0.040,
        "crimp":       0.025,
        "sloper":      0.025,
        "pinch":       0.020,
        "lateral":     0.015,
        "sustained":   0.015,
        "coordination":0.015,
        "overhang_40": 0.010,
        "overhang_50": 0.020,
        "overhang_60": 0.030,
        "general":     0.005,
    }

    delta = 0.0
    for mt in move_types:
        weight       = _WEIGHTS.get(mt, 0.010)
        success_rate = affinities.get(mt)

        if success_rate is None:
            # Unknown affinity → small neutral penalty (benefit of the doubt)
            delta += weight * 0.15
        else:
            # Shift centred at 0.5 success rate: below 0.5 = harder, above = easier
            skill_gap = 0.5 - success_rate   # positive = you struggle with this
            delta += weight * skill_gap * 2

    return round(delta, 4)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_pdscore(
    base_score: float,
    features: dict,
    profile: dict,
    affinities: dict[str, float],
    boundaries: Optional[dict] = None,   # kept for backwards-compat, unused
) -> dict:
    """
    Compute the personal difficulty score for a route.

    Parameters
    ----------
    base_score   : ML difficulty_score from main model (0–1)
    features     : route feature dict (from compute_features or DB)
    profile      : climber profile dict (height_cm, wingspan_cm, weight_kg, ...)
    affinities   : {move_type → success_rate} from climber_move_affinity table

    Returns
    -------
    {
        "pdscore":      float,  # 0–1 raw
        "pd_personal":  float,  # 1.0–10.0 personal scale
        "body_delta":   float,  # signed
        "move_delta":   float,  # signed
        "move_types":   list,   # what styles this route tests
        "modifiers":    dict,   # breakdown for display
    }
    """
    body  = _body_delta(features, profile)
    moves = _move_delta(features, affinities)
    pd    = max(0.0, min(1.0, base_score + body + moves))

    # Personal scale: 1.0–10.0 with one decimal place.
    # Maps the 0–1 difficulty float to a climber-relative scale where:
    #   1.0  = very easy for you  (≈ V0 equivalent)
    #   5.5  = moderate challenge (≈ mid V5–V6)
    #   10.0 = at or beyond your limit
    pd_personal = round(pd * 9.0 + 1.0, 1)

    move_types = _classify_moves(features)

    return {
        "pdscore":     round(pd, 4),
        "pd_personal": pd_personal,       # 1.0–10.0 personal scale
        "body_delta":  body,
        "move_delta":  moves,
        "move_types":  move_types,
        "modifiers": {
            "body":  round(body, 4),
            "moves": round(moves, 4),
            "total": round(body + moves, 4),
        },
    }

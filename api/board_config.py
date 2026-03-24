"""
api/board_config.py
-------------------
Board layout configs and hold loading for Kilter Original and Homewall.

Holds are loaded once per layout at startup and cached in memory.
"""

import json
import pandas as pd
import sqlite3

from api.db import SQLITE_PATH, DATA_DIR

# ── Board layout configs ──────────────────────────────────────────────────────

BOARD_CONFIGS: dict = {
    "original": {
        "layout_id":    1,
        "name":         "Kilter Board Original",
        "description":  "Commercial/gym layout. Larger ergonomic holds, dynamic gymnastic style.",
        "hold_character": "Bolt Ons (jugs/slopers) + Screw Ons (crimps/pinches)",
        "climbology_x_offset": 4,
        "climbology_y_offset": 4,
        "sets": {
            "all":       None,
            "bolt-ons":  [1],
            "screw-ons": [20],
        },
        "default_set": "all",
    },
    "homewall": {
        "layout_id":    8,
        "name":         "Kilter Board Homewall",
        "description":  "Home/smaller space layout. More crimps, pinches, and flakes. Technical and skin-friendly.",
        "hold_character": "Mainline + Auxiliary. Higher density (4.34 vs 2.20 holds/sqft). More technical/static.",
        "climbology_x_offset": -60,
        "climbology_y_offset": -8,
        "sets": {
            "all":           None,
            "mainline":      [26, 28],
            "auxiliary":     [27, 29],
            "full-ride":     [26, 27, 28, 29],
        },
        "default_set": "mainline",
    },
}

# ── Hold cache (populated at startup) ─────────────────────────────────────────

_board_cache: dict = {}


def get_board_holds(board_type: str = "original", set_filter: str = "all") -> dict:
    """
    Return board holds with Climbology type annotations and display percentages.
    Results are cached in memory for the lifetime of the process.
    """
    cache_key = f"{board_type}:{set_filter}"
    if cache_key in _board_cache:
        return _board_cache[cache_key]

    cfg = BOARD_CONFIGS.get(board_type, BOARD_CONFIGS["original"])

    climbology = pd.read_csv(DATA_DIR / "kilter_hold_types.csv")
    climbology["x_mapped"] = climbology["x_coordinate"] * 4 + cfg["climbology_x_offset"]
    climbology["y_mapped"] = climbology["y_coordinate"] * 4 + cfg["climbology_y_offset"]

    set_ids = cfg["sets"].get(set_filter, cfg["sets"].get(cfg["default_set"]))
    set_clause = ""
    if set_ids:
        ids_str = ",".join(str(i) for i in set_ids)
        set_clause = f"AND p.set_id IN ({ids_str})"

    with sqlite3.connect(str(SQLITE_PATH)) as con:
        holes = pd.read_sql(f"""
            SELECT h.id AS hole_id, h.x, h.y, p.id AS placement_id,
                   s.name AS set_name, s.hsm AS set_hsm
            FROM holes h
            JOIN placements p ON p.hole_id = h.id
            JOIN sets s ON p.set_id = s.id
            WHERE p.layout_id = {cfg['layout_id']}
            {set_clause}
            ORDER BY h.y, h.x
        """, con)

    merged = holes.merge(
        climbology[["x_mapped", "y_mapped", "type", "depth", "size"]],
        left_on=["x", "y"], right_on=["x_mapped", "y_mapped"], how="left",
    )
    merged["hold_type"] = merged["type"].fillna("unknown")

    if board_type == "homewall":
        merged.loc[merged["set_hsm"] == 4, "hold_type"] = "foot"

    x_min, x_max = int(merged["x"].min()), int(merged["x"].max())
    y_min, y_max = int(merged["y"].min()), int(merged["y"].max())
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    holds = []
    for _, row in merged.iterrows():
        x_pct = round((row["x"] - x_min) / x_range * 100, 2)
        y_pct = round(100 - (row["y"] - y_min) / y_range * 100, 2)
        holds.append({
            "id":        int(row["hole_id"]),
            "pid":       int(row["placement_id"]),
            "x":         int(row["x"]),
            "y":         int(row["y"]),
            "x_pct":     x_pct,
            "y_pct":     y_pct,
            "hold_type": row["hold_type"],
            "set_name":  row["set_name"],
        })

    result = {
        "board_type":      board_type,
        "board_name":      cfg["name"],
        "description":     cfg["description"],
        "hold_character":  cfg["hold_character"],
        "set_filter":      set_filter,
        "sets_available":  list(cfg["sets"].keys()),
        "holds":   holds,
        "x_min":   x_min, "x_max": x_max,
        "y_min":   y_min, "y_max": y_max,
        "x_range": x_range, "y_range": y_range,
        "count":   len(holds),
        "type_counts": {
            t: int((merged["hold_type"] == t).sum())
            for t in ["jug", "crimp", "sloper", "pinch", "foot", "unknown"]
        },
    }
    _board_cache[cache_key] = result
    return result

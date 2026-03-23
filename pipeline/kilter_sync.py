"""
pipeline/kilter_sync.py

Reads the local Kilter Board SQLite database (kilter.db) downloaded via BoardLib
and populates PostgreSQL tables:
  - board_routes
  - route_holds

Holds without a known position are inserted with NULL x/y coordinates
but are still counted for role-based features (foot_hold_count etc).
Grid col/row also NULL for unknown positions — schema allows this.

Usage:
    python3 pipeline/kilter_sync.py
    python3 pipeline/kilter_sync.py --angle 40 --limit 5000
    python3 pipeline/kilter_sync.py --all-angles
"""

import os
import re
import sys
import sqlite3
import argparse
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

SQLITE_PATH = os.path.expanduser(
    os.getenv("KILTER_DB_PATH", "~/Desktop/Climbing ML/kilter.db")
)

STANDARD_ANGLES = [0, 20, 35, 40, 45, 50, 55, 60, 65, 70]

# ── Grade mapping ─────────────────────────────────────────────────────────────

def extract_vgrade(boulder_col):
    if not boulder_col:
        return None
    match = re.search(r'(V\d+)', boulder_col)
    return match.group(1) if match else None

def normalise_difficulty(raw, max_grade=33):
    return round(min(max(float(raw) / max_grade, 0.0), 1.0), 4)

# ── Hold role mapping ─────────────────────────────────────────────────────────

ROLE_MAP = {12: "start", 13: "hand", 14: "foot", 15: "finish"}

def map_role(role_id):
    return ROLE_MAP.get(role_id, "hand")

# ── Frames parser ─────────────────────────────────────────────────────────────

FRAMES_RE = re.compile(r'p(\d+)r(\d+)')

def parse_frames(frames):
    return [(int(m.group(1)), int(m.group(2))) for m in FRAMES_RE.finditer(frames or "")]

# ── SQLite loaders ────────────────────────────────────────────────────────────

def load_grade_map(cur):
    cur.execute("SELECT difficulty, boulder_name FROM difficulty_grades WHERE is_listed = 1")
    grade_map = {}
    for row in cur.fetchall():
        vgrade = extract_vgrade(row[1])
        if vgrade:
            grade_map[row[0]] = vgrade
    return grade_map

def load_hole_positions(cur):
    """Returns dict: placement_id -> (x_cm, y_cm) via placements → holes join."""
    cur.execute("""
        SELECT p.id, h.x, h.y
        FROM placements p
        JOIN holes h ON h.id = p.hole_id
    """)
    return {row[0]: (float(row[1]), float(row[2])) for row in cur.fetchall()}

# ── PostgreSQL connection ─────────────────────────────────────────────────────

def get_pg_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "climbing_platform"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD", ""),
    )

# ── SQL ───────────────────────────────────────────────────────────────────────

INSERT_ROUTE = """
    INSERT INTO board_routes (
        source, external_id, board_type, board_angle_deg,
        name, setter_name, community_grade, difficulty_score,
        send_count, avg_quality_rating, metadata
    )
    VALUES (
        'kilter', %(external_id)s, 'kilter_original', %(angle)s,
        %(name)s, %(setter_name)s, %(community_grade)s, %(difficulty_score)s,
        %(send_count)s, %(avg_quality_rating)s, %(metadata)s
    )
    ON CONFLICT (source, external_id, board_angle_deg) DO UPDATE SET
        community_grade     = EXCLUDED.community_grade,
        difficulty_score    = EXCLUDED.difficulty_score,
        send_count          = EXCLUDED.send_count,
        avg_quality_rating  = EXCLUDED.avg_quality_rating,
        synced_at           = NOW()
    RETURNING id;
"""

INSERT_HOLD = """
    INSERT INTO route_holds (
        route_id, grid_col, grid_row,
        position_x_cm, position_y_cm,
        role, board_angle_deg, hand_sequence
    )
    VALUES (
        %(route_id)s, %(grid_col)s, %(grid_row)s,
        %(position_x_cm)s, %(position_y_cm)s,
        %(role)s, %(board_angle_deg)s, %(hand_sequence)s
    );
"""

# ── Core sync ─────────────────────────────────────────────────────────────────

def sync_angle(sqlite_con, pg_cur, grade_map, hole_positions,
               angle, limit, offset=0, min_ascents=5):

    query = """
        SELECT
            c.uuid, c.name, c.setter_username, c.frames, c.angle,
            c.is_draft, c.frames_count,
            cs.ascensionist_count, cs.difficulty_average, cs.quality_average
        FROM climbs c
        LEFT JOIN climb_stats cs ON cs.climb_uuid = c.uuid AND cs.angle = ?
        WHERE
            c.is_draft = 0
            AND c.is_listed = 1
            AND (c.angle = ? OR c.angle IS NULL)
            AND (cs.ascensionist_count IS NULL OR cs.ascensionist_count >= ?)
            AND c.frames IS NOT NULL
            AND c.frames != ''
        ORDER BY cs.ascensionist_count DESC
        LIMIT ? OFFSET ?
    """

    sqlite_cur = sqlite_con.cursor()
    sqlite_cur.execute(query, (angle, angle, min_ascents, limit, offset))
    rows = sqlite_cur.fetchall()

    routes_done = 0
    holds_done  = 0

    for row in rows:
        (uuid, name, setter, frames, climb_angle, is_draft,
         frames_count, ascensionist_count, difficulty_avg, quality_avg) = row

        effective_angle  = climb_angle if climb_angle is not None else angle
        grade_id         = round(float(difficulty_avg)) if difficulty_avg else None
        community_grade  = grade_map.get(grade_id) if grade_id else None
        difficulty_score = normalise_difficulty(difficulty_avg) if difficulty_avg else None

        pg_cur.execute(INSERT_ROUTE, {
            "external_id":        uuid,
            "angle":              effective_angle,
            "name":               name or None,
            "setter_name":        setter or None,
            "community_grade":    community_grade,
            "difficulty_score":   difficulty_score,
            "send_count":         int(ascensionist_count) if ascensionist_count else 0,
            "avg_quality_rating": float(quality_avg) if quality_avg else None,
            "metadata":           psycopg2.extras.Json({
                "frames_count": frames_count,
                "is_draft":     bool(is_draft),
                "source":       "boardlib_sqlite",
            }),
        })
        result = pg_cur.fetchone()
        if not result:
            continue
        route_pg_id = str(result[0])

        pg_cur.execute("DELETE FROM route_holds WHERE route_id = %s", (route_pg_id,))

        placements = parse_frames(frames)
        hand_seq   = 1

        for placement_id, role_id in placements:
            role = map_role(role_id)

            # Look up physical position — may be None for unknown placements
            pos  = hole_positions.get(placement_id)
            if pos is not None:
                x_cm, y_cm   = pos
                grid_col     = max(0, min(17, round((x_cm / 140.0) * 17)))
                grid_row     = max(0, min(17, round((y_cm / 140.0) * 17)))
                x_cm_rounded = round(x_cm, 2)
                y_cm_rounded = round(y_cm, 2)
            else:
                # Position unknown — store NULL for geometry columns
                # Hold still counts toward role-based features (foot_hold_count etc)
                x_cm_rounded = None
                y_cm_rounded = None
                grid_col     = 0   # schema requires int — use 0 as sentinel
                grid_row     = 0   # feature extractor filters on NULL position_x_cm

            seq = None
            if role in ("start", "hand", "finish"):
                seq = hand_seq
                hand_seq += 1

            pg_cur.execute(INSERT_HOLD, {
                "route_id":        route_pg_id,
                "grid_col":        grid_col,
                "grid_row":        grid_row,
                "position_x_cm":   x_cm_rounded,
                "position_y_cm":   y_cm_rounded,
                "role":            role,
                "board_angle_deg": float(effective_angle),
                "hand_sequence":   seq,
            })
            holds_done += 1

        routes_done += 1

    return routes_done, holds_done

# ── Main ──────────────────────────────────────────────────────────────────────

def main(angle=40, limit=10000, all_angles=False, min_ascents=5):
    angles = STANDARD_ANGLES if all_angles else [angle]

    print(f"\nKilter Board sync — reading: {SQLITE_PATH}")
    print(f"Angles: {angles}  |  Limit per angle: {limit:,}  |  Min ascents: {min_ascents}")
    print("─" * 60)

    if not os.path.exists(SQLITE_PATH):
        print(f"\nERROR: kilter.db not found at:\n  {SQLITE_PATH}")
        print("Run: boardlib database kilter kilter.db --username YOUR_EMAIL")
        sys.exit(1)

    sqlite_con = sqlite3.connect(SQLITE_PATH)
    sqlite_cur = sqlite_con.cursor()

    print("\nLoading lookup tables...")
    grade_map      = load_grade_map(sqlite_cur)
    hole_positions = load_hole_positions(sqlite_cur)
    print(f"  {len(grade_map)} grades  |  {len(hole_positions)} hole positions")

    pg_con = get_pg_connection()
    pg_con.autocommit = False
    pg_cur = pg_con.cursor()

    total_routes = 0
    total_holds  = 0

    try:
        for a in angles:
            print(f"\nAngle {a}°...")
            offset = 0
            page   = 1

            while True:
                batch = min(1000, limit)
                r, h  = sync_angle(sqlite_con, pg_cur, grade_map, hole_positions,
                                   angle=a, limit=batch, offset=offset,
                                   min_ascents=min_ascents)
                if r == 0:
                    print(f"  Done at {a}°.")
                    break

                total_routes += r
                total_holds  += h
                pg_con.commit()
                print(f"  Page {page}: +{r:,} routes  +{h:,} holds  (running: {total_routes:,})")

                if r < batch:
                    break
                if total_routes >= limit and not all_angles:
                    break

                offset += batch
                page   += 1

    except KeyboardInterrupt:
        print("\nInterrupted — saving progress...")
        pg_con.commit()
    except Exception as e:
        pg_con.rollback()
        print(f"\nERROR — rolled back: {e}")
        raise
    finally:
        pg_cur.close()
        pg_con.close()
        sqlite_con.close()

    print("\n" + "─" * 60)
    print(f"Sync complete:  {total_routes:,} routes  |  {total_holds:,} holds")

    # Verification
    vcon = get_pg_connection()
    vcur = vcon.cursor()
    vcur.execute("SELECT COUNT(*) FROM board_routes WHERE source = 'kilter'")
    print(f"\nPostgreSQL board_routes (kilter): {vcur.fetchone()[0]:,}")
    vcur.execute("SELECT COUNT(*) FROM route_holds")
    print(f"PostgreSQL route_holds total:     {vcur.fetchone()[0]:,}")
    vcur.execute("""
        SELECT community_grade, COUNT(*) as n
        FROM board_routes WHERE source='kilter' AND community_grade IS NOT NULL
        GROUP BY community_grade ORDER BY community_grade
    """)
    print("\nRoutes by grade:")
    for row in vcur.fetchall():
        bar = "█" * min(50, row[1] // 500)
        print(f"  {row[0]:>4}  {row[1]:>6,}  {bar}")
    vcur.close()
    vcon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Kilter Board SQLite into PostgreSQL")
    parser.add_argument("--angle",       type=int, default=40)
    parser.add_argument("--limit",       type=int, default=10000)
    parser.add_argument("--min-ascents", type=int, default=5)
    parser.add_argument("--all-angles",  action="store_true")
    args = parser.parse_args()
    main(angle=args.angle, limit=args.limit, all_angles=args.all_angles,
         min_ascents=args.min_ascents)

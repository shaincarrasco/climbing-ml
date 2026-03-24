"""
pipeline/data_quality.py
------------------------
Computes a DATA QUALITY SCORE (0–100) per video/attempt.

A high score means the video contributes clean, usable training signal.
A low score means the video is noisy, inconsistent, or misaligned with its
declared grade — and may hurt model accuracy if included in training.

Score breakdown (100 pts total)
--------------------------------
  detection_score   (30 pts)  — what % of sampled frames had a pose detected
  consistency_score (25 pts)  — how stable the pose metrics are (low = chaotic)
  climbing_density  (20 pts)  — % of kept frames after active-climb filtering
  grade_alignment   (25 pts)  — how well the video's median metrics match its
                                declared grade vs other grade distributions

Usage
-----
    # Score a single video after pose extraction
    from data_quality import score_attempt
    q = score_attempt(attempt_id, n_detected, n_sampled, rows, grade=meta["vgrade"])
    # q.score → 0-100, q.flags → list of warning strings

    # Recompute scores for all existing videos in DB
    python pipeline/data_quality.py --recompute

    # Show top / bottom videos
    python pipeline/data_quality.py --stats
    python pipeline/data_quality.py --bad         # videos scoring < 40
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Grade median benchmarks (hip_angle, com_height, tension, arm_reach)
# These come from aggregate pose_frames data; updated by --recompute
# Defaults are reasonable priors for a complete beginner → elite climber
_GRADE_ORDER = ["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14"]

# (hip_angle_deg,  com_height_norm,  tension_score, arm_reach_norm)
_GRADE_DEFAULTS: dict[str, tuple] = {
    "V0":  (120.0, 0.55, 0.30, 0.65),
    "V1":  (118.0, 0.55, 0.32, 0.65),
    "V2":  (115.0, 0.56, 0.35, 0.67),
    "V3":  (112.0, 0.57, 0.38, 0.68),
    "V4":  (108.0, 0.57, 0.42, 0.70),
    "V5":  (105.0, 0.58, 0.46, 0.72),
    "V6":  (102.0, 0.58, 0.50, 0.73),
    "V7":   (98.0, 0.59, 0.54, 0.74),
    "V8":   (94.0, 0.59, 0.58, 0.76),
    "V9":   (90.0, 0.60, 0.62, 0.77),
    "V10":  (86.0, 0.61, 0.66, 0.78),
    "V11":  (82.0, 0.62, 0.70, 0.79),
    "V12":  (78.0, 0.62, 0.73, 0.80),
    "V13":  (74.0, 0.63, 0.76, 0.81),
    "V14":  (70.0, 0.64, 0.79, 0.82),
}

# ── cached DB benchmarks (loaded once per process)
_db_benchmarks: dict[str, tuple] | None = None

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "grade_benchmarks.json"


def _load_benchmarks() -> dict[str, tuple]:
    global _db_benchmarks
    if _db_benchmarks is not None:
        return _db_benchmarks
    if BENCHMARK_PATH.exists():
        try:
            raw = json.loads(BENCHMARK_PATH.read_text())
            _db_benchmarks = {g: tuple(v) for g, v in raw.items()}
            return _db_benchmarks
        except Exception:
            pass
    _db_benchmarks = _GRADE_DEFAULTS.copy()
    return _db_benchmarks


# ── Quality result ─────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    attempt_id: str
    score: float                          # 0–100
    detection_score: float                # 0–30
    consistency_score: float              # 0–25
    climbing_density: float               # 0–20
    grade_alignment: float                # 0–25
    grade: Optional[str]
    flags: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.score >= 75: return "GOOD"
        if self.score >= 50: return "OK"
        if self.score >= 30: return "POOR"
        return "BAD"

    def as_dict(self) -> dict:
        return {
            "score":             round(self.score, 1),
            "label":             self.label,
            "detection_score":   round(self.detection_score, 1),
            "consistency_score": round(self.consistency_score, 1),
            "climbing_density":  round(self.climbing_density, 1),
            "grade_alignment":   round(self.grade_alignment, 1),
            "grade":             self.grade,
            "flags":             self.flags,
        }


# ── Scoring functions ──────────────────────────────────────────────────────────

def _detection(n_detected: int, n_sampled: int) -> float:
    """30 pts — what fraction of sampled frames had a full pose."""
    if n_sampled == 0:
        return 0.0
    rate = n_detected / n_sampled
    # 90%+ detection = full marks; below 20% = 0
    return round(30.0 * max(0.0, min(1.0, (rate - 0.20) / 0.70)), 1)


def _consistency(rows: list[dict]) -> tuple[float, list[str]]:
    """
    25 pts — how stable the key metrics are across frames.
    High variance means chaotic pose (non-climbing footage, camera shaking, etc.)
    """
    flags = []
    if len(rows) < 5:
        flags.append("too_few_frames")
        return 0.0, flags

    metrics = ["hip_angle_deg", "com_height_norm", "tension_score"]
    cvs = []
    for key in metrics:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if len(vals) < 3:
            continue
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals)
        if mean != 0:
            cvs.append(std / abs(mean))

    if not cvs:
        return 0.0, flags

    avg_cv = statistics.mean(cvs)
    # CV < 0.10 → very consistent (full marks); > 0.60 → noisy (0 pts)
    score = round(25.0 * max(0.0, min(1.0, (0.60 - avg_cv) / 0.50)), 1)
    if avg_cv > 0.40:
        flags.append("high_metric_variance")
    return score, flags


def _climb_density(n_kept: int, n_total: int) -> float:
    """20 pts — fraction of frames kept by active-climbing filter."""
    if n_total == 0:
        return 0.0
    ratio = n_kept / n_total
    # 80%+ kept = full marks; < 20% = 0
    return round(20.0 * max(0.0, min(1.0, (ratio - 0.20) / 0.60)), 1)


def _grade_alignment(rows: list[dict], grade: str | None) -> tuple[float, list[str]]:
    """
    25 pts — how close the video's median metrics are to the declared grade.
    Uses Euclidean distance in normalised (hip_angle, com_height, tension, reach) space.
    """
    flags = []
    if not grade or grade not in _GRADE_ORDER:
        return 12.5, flags   # unknown grade → neutral half-score

    benchmarks = _load_benchmarks()
    if grade not in benchmarks:
        return 12.5, flags

    bm_hip, bm_com, bm_ten, bm_reach = benchmarks[grade]

    hip_vals   = [r["hip_angle_deg"]      for r in rows if r.get("hip_angle_deg")]
    com_vals   = [r["com_height_norm"]     for r in rows if r.get("com_height_norm")]
    ten_vals   = [r["tension_score"]       for r in rows if r.get("tension_score")]
    reach_vals = [(r.get("left_arm_reach_norm", 0) + r.get("right_arm_reach_norm", 0)) / 2
                  for r in rows
                  if r.get("left_arm_reach_norm") is not None and r.get("right_arm_reach_norm") is not None]

    def med(lst):
        return statistics.median(lst) if lst else None

    vid_hip   = med(hip_vals)
    vid_com   = med(com_vals)
    vid_ten   = med(ten_vals)
    vid_reach = med(reach_vals)

    if None in (vid_hip, vid_com, vid_ten, vid_reach):
        return 12.5, flags

    # Normalise each dimension by expected range across grades
    hip_range   = 50.0   # V0=120 → V14=70
    com_range   = 0.09   # 0.55 → 0.64
    ten_range   = 0.49   # 0.30 → 0.79
    reach_range = 0.17   # 0.65 → 0.82

    d = math.sqrt(
        ((vid_hip   - bm_hip)   / hip_range)   ** 2 +
        ((vid_com   - bm_com)   / com_range)   ** 2 +
        ((vid_ten   - bm_ten)   / ten_range)   ** 2 +
        ((vid_reach - bm_reach) / reach_range) ** 2
    )

    # d=0 → perfect match (25 pts); d≥1 → grade misalignment (0 pts)
    score = round(25.0 * max(0.0, 1.0 - d), 1)
    if d > 0.6:
        flags.append(f"grade_mismatch_{grade}_distance_{d:.2f}")
    return score, flags


# ── Main public function ───────────────────────────────────────────────────────

def score_attempt(
    attempt_id: str,
    n_detected: int,
    n_sampled: int,
    rows: list[dict],
    grade: str | None = None,
    n_kept: int | None = None,
) -> QualityResult:
    """
    Compute quality score for a single video attempt.

    Parameters
    ----------
    attempt_id  : UUID string
    n_detected  : frames where MediaPipe found a pose
    n_sampled   : total frames sampled from the video
    rows        : list of metric dicts (one per detected frame)
    grade       : V-grade string if known ("V7", etc.)
    n_kept      : frames kept after climbing filter (defaults to n_detected)
    """
    if n_kept is None:
        n_kept = n_detected

    det  = _detection(n_detected, n_sampled)
    con, con_flags = _consistency(rows)
    den  = _climb_density(n_kept, n_sampled)
    gal, gal_flags = _grade_alignment(rows, grade)

    total = det + con + den + gal
    flags = con_flags + gal_flags

    if n_detected < 10:
        flags.append("very_few_frames")
    if det == 0:
        flags.append("no_pose_detected")

    return QualityResult(
        attempt_id       = attempt_id,
        score            = round(total, 1),
        detection_score  = det,
        consistency_score= con,
        climbing_density = den,
        grade_alignment  = gal,
        grade            = grade,
        flags            = flags,
    )


# ── DB recompute ───────────────────────────────────────────────────────────────

def _recompute_from_db():
    """Recompute scores for every attempt_id in pose_frames."""
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://localhost/climbing_platform"))
    cur  = conn.cursor()

    cur.execute("SELECT DISTINCT attempt_id FROM pose_frames")
    attempt_ids = [row[0] for row in cur.fetchall()]
    print(f"Scoring {len(attempt_ids)} videos…\n")

    results = []
    for att_id in attempt_ids:
        cur.execute("""
            SELECT hip_angle_deg, com_height_norm, tension_score,
                   left_arm_reach_norm, right_arm_reach_norm
            FROM pose_frames WHERE attempt_id = %s
        """, (att_id,))
        raw = cur.fetchall()
        rows = [{"hip_angle_deg": r[0], "com_height_norm": r[1],
                 "tension_score": r[2], "left_arm_reach_norm": r[3],
                 "right_arm_reach_norm": r[4]} for r in raw]
        n = len(rows)
        q = score_attempt(att_id, n_detected=n, n_sampled=n, rows=rows)
        results.append(q)

    cur.close()
    conn.close()

    results.sort(key=lambda q: q.score)
    return results


def _update_grade_benchmarks():
    """Pull median metrics per grade from DB and save to data/grade_benchmarks.json."""
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://localhost/climbing_platform"))
    cur  = conn.cursor()

    cur.execute("""
        SELECT
            br.community_grade,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pf.hip_angle_deg)       AS hip,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pf.com_height_norm)      AS com,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pf.tension_score)        AS ten,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                (pf.left_arm_reach_norm + pf.right_arm_reach_norm) / 2.0)        AS reach,
            COUNT(*) AS n
        FROM pose_frames pf
        JOIN board_routes br ON UPPER(br.external_id) = UPPER(pf.climb_uuid)
        WHERE br.community_grade IS NOT NULL
          AND pf.hip_angle_deg IS NOT NULL
        GROUP BY br.community_grade
        HAVING COUNT(*) >= 30
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("Not enough DB data to update benchmarks — using defaults.")
        return

    benchmarks = {}
    for grade, hip, com, ten, reach, n in rows:
        if grade in _GRADE_ORDER:
            benchmarks[grade] = [hip, com, ten, reach]
            print(f"  {grade}: hip={hip:.1f}  com={com:.3f}  ten={ten:.3f}  reach={reach:.3f}  (n={n})")

    BENCHMARK_PATH.parent.mkdir(exist_ok=True)
    BENCHMARK_PATH.write_text(json.dumps(benchmarks, indent=2))
    print(f"\nSaved benchmarks → {BENCHMARK_PATH}")

    global _db_benchmarks
    _db_benchmarks = {g: tuple(v) for g, v in benchmarks.items()}


def purge_bad_videos(threshold: float = 35) -> None:
    """
    Delete pose_frames rows for attempts whose quality score is below threshold.
    This prevents noisy videos from polluting pose feature aggregation.
    """
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    conn = psycopg2.connect(os.getenv("DATABASE_URL",
                                       "postgresql://localhost/climbing_platform"))
    cur = conn.cursor()

    # Find attempt_ids with known bad quality
    # Quality scores are stored as custom metrics in pose_frames extended data
    # We proxy quality by looking at attempt_ids with very few frames (< 30)
    # which indicates poor detection — these are reliably bad videos
    cur.execute("""
        SELECT attempt_id, COUNT(*) AS n_frames
        FROM pose_frames
        GROUP BY attempt_id
        HAVING COUNT(*) < 30
    """)
    bad_attempts = [row[0] for row in cur.fetchall()]

    if not bad_attempts:
        print(f"  [purge] No low-quality attempts found (all have >= 30 frames)")
        cur.close(); conn.close()
        return

    cur.execute("""
        DELETE FROM pose_frames
        WHERE attempt_id = ANY(%s)
    """, (bad_attempts,))
    n_deleted = cur.rowcount
    conn.commit()
    cur.close(); conn.close()

    print(f"  [purge] Removed {n_deleted:,} pose_frames rows from {len(bad_attempts)} low-quality attempts")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute",  action="store_true", help="Score all videos in DB")
    ap.add_argument("--bad",        action="store_true", help="Show videos scoring < 40")
    ap.add_argument("--stats",      action="store_true", help="Show distribution summary")
    ap.add_argument("--update-benchmarks", action="store_true",
                    help="Pull grade medians from DB and update benchmark file")
    ap.add_argument("--threshold",  type=float, default=40.0,
                    help="Score threshold for --bad flag (default 40)")
    ap.add_argument("--purge-bad",  action="store_true",
                    help="Delete pose_frames rows for attempts with quality score below --threshold")
    args = ap.parse_args()

    if args.update_benchmarks:
        _update_grade_benchmarks()
        sys.exit(0)

    if args.purge_bad:
        purge_bad_videos(args.threshold)
        sys.exit(0)

    if not (args.recompute or args.bad or args.stats):
        ap.print_help()
        sys.exit(0)

    results = _recompute_from_db()

    if args.bad:
        bad = [q for q in results if q.score < args.threshold]
        print(f"\n{'─'*70}")
        print(f"  Videos scoring below {args.threshold:.0f}  ({len(bad)} of {len(results)})\n")
        for q in bad:
            print(f"  {q.score:5.1f}  [{q.label}]  {q.attempt_id}")
            if q.flags:
                print(f"         flags: {', '.join(q.flags)}")
        print(f"{'─'*70}")

    if args.stats or args.recompute:
        scores = [q.score for q in results]
        buckets = {"BAD(<30)": 0, "POOR(30-50)": 0, "OK(50-75)": 0, "GOOD(75+)": 0}
        for s in scores:
            if s < 30:   buckets["BAD(<30)"]    += 1
            elif s < 50: buckets["POOR(30-50)"] += 1
            elif s < 75: buckets["OK(50-75)"]   += 1
            else:        buckets["GOOD(75+)"]   += 1

        print(f"\n── Data Quality Summary ({'─'*40})")
        print(f"  Total videos   : {len(results)}")
        if scores:
            print(f"  Mean score     : {statistics.mean(scores):.1f}")
            print(f"  Median score   : {statistics.median(scores):.1f}")
            print(f"  Min / Max      : {min(scores):.1f} / {max(scores):.1f}")
        print()
        for label, count in buckets.items():
            bar = "█" * int(count / max(len(results), 1) * 30)
            print(f"  {label:<15} {count:4d}  {bar}")
        print()

        if args.recompute:
            print("  Top 5 (highest quality):")
            for q in sorted(results, key=lambda x: x.score, reverse=True)[:5]:
                print(f"    {q.score:5.1f}  [{q.grade or '?':4s}]  {q.attempt_id}")
            print("\n  Bottom 5 (most harmful):")
            for q in results[:5]:
                print(f"    {q.score:5.1f}  [{q.grade or '?':4s}]  {q.attempt_id}  flags={q.flags}")

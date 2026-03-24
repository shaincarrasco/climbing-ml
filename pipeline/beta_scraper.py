"""
pipeline/beta_scraper.py
------------------------
Scrapes beta videos from kilter.db (32,839 Instagram/TikTok links),
downloads each via yt-dlp, runs pose_extractor, writes pose_frames to
PostgreSQL, then deletes the local video to preserve disk space.

Usage:
    # Dry run — download + extract but no DB writes
    python pipeline/beta_scraper.py --dry-run --limit 3

    # Full run, 10 videos
    python pipeline/beta_scraper.py --limit 10

    # Only process one specific climb
    python pipeline/beta_scraper.py --climb-uuid 00045F0E6F0340ACAE7CF216E1055B8B

    # Process already-downloaded local videos (e.g. from Instagram app)
    python pipeline/beta_scraper.py --local-dir ~/Downloads/climbing_videos

    # Progress stats
    python pipeline/beta_scraper.py --stats
"""

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
import uuid
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR       = Path(__file__).resolve().parent.parent
KILTER_DB       = _BASE_DIR / "kilter.db"
CHECKPOINT_FILE = _BASE_DIR / "data" / "beta_scrape_progress.json"
TMP_VIDEO_DIR   = Path("/tmp/beta_videos")

# ── Import pose_extractor from sibling module ──────────────────────────────────
sys.path.insert(0, str(_BASE_DIR / "pipeline"))
from pose_extractor import process_video
from data_quality import score_attempt


# ── Deterministic attempt_id ──────────────────────────────────────────────────
def make_attempt_id(climb_uuid: str, url: str) -> str:
    """SHA-256(climb_uuid + url), first 32 hex chars formatted as UUID."""
    raw = hashlib.sha256(f"{climb_uuid}{url}".encode()).hexdigest()[:32]
    return str(uuid.UUID(raw))


# ── Checkpoint (progress persistence) ─────────────────────────────────────────
def load_checkpoint() -> dict:
    """Returns {'completed': {attempt_id: climb_uuid}, 'failed': {url: reason}}"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_checkpoint(checkpoint: dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ── kilter.db queries ─────────────────────────────────────────────────────────
def _grade_to_difficulty_range(vgrade: str) -> tuple[int, int]:
    """
    Convert a V-grade string to the (min, max) difficulty integer range in
    kilter.db's difficulty_grades table. e.g. "V7" → (22, 24).
    """
    # Build a mapping from V-grade name to list of difficulty ints
    conn = sqlite3.connect(KILTER_DB)
    cur  = conn.cursor()
    cur.execute("SELECT boulder_name, difficulty FROM difficulty_grades")
    grade_map: dict[str, list[int]] = {}
    for name, diff in cur.fetchall():
        # boulder_name is like "6c+/V5" — extract the V-grade part
        vpart = name.split("/")[-1].strip()
        grade_map.setdefault(vpart, []).append(diff)
    conn.close()

    vgrade = vgrade.upper().strip()
    if vgrade not in grade_map:
        raise ValueError(f"Unknown grade '{vgrade}'. Valid: {sorted(grade_map)}")
    diffs = grade_map[vgrade]
    return min(diffs), max(diffs)


def get_beta_links(
    climb_uuid: str | None = None,
    limit: int | None = None,
    reels_only: bool = False,
    since: str | None = None,
    grades: list[str] | None = None,
) -> list[tuple]:
    """
    Returns list of (climb_uuid, link) tuples.
    Ordered newest-first so recent (still-live) posts are processed first.

    grades: filter to specific V-grades e.g. ['V7', 'V8', 'V9']
      Joins beta_links → climb_stats → difficulty_grades to match.
      Uses display_difficulty averaged across all angles for a climb.
    reels_only: only include /reel/ links (higher success rate than old /p/ posts)
    since: only include links created after this date e.g. '2022-01-01'
    """
    conn = sqlite3.connect(KILTER_DB)
    cur  = conn.cursor()

    filters = []
    params  = []

    base_from = "beta_links bl"

    if grades:
        # Resolve each grade to a difficulty range and build an IN clause
        # We join climb_stats and average display_difficulty across angles
        ranges = [_grade_to_difficulty_range(g) for g in grades]
        range_conditions = " OR ".join(
            f"(ROUND(avg_diff) BETWEEN {lo} AND {hi})" for lo, hi in ranges
        )
        base_from = (
            "beta_links bl "
            "JOIN ("
            "  SELECT climb_uuid, AVG(display_difficulty) as avg_diff "
            "  FROM climb_stats GROUP BY climb_uuid"
            ") cs ON cs.climb_uuid = bl.climb_uuid"
        )
        filters.append(f"({range_conditions})")

    if climb_uuid:
        filters.append("bl.climb_uuid = ?")
        params.append(climb_uuid)
    if reels_only:
        filters.append("bl.link LIKE '%/reel/%'")
    if since:
        filters.append("bl.created_at >= ?")
        params.append(since)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    cur.execute(
        f"SELECT bl.climb_uuid, bl.link FROM {base_from} {where} ORDER BY bl.created_at DESC",
        params,
    )
    rows = cur.fetchall()
    conn.close()
    if limit:
        rows = rows[:limit]
    return rows


# ── yt-dlp download ───────────────────────────────────────────────────────────
def download_video(url: str, out_path: Path) -> bool:
    """
    Downloads url to out_path using yt-dlp (max 720p, mp4).
    Returns True on success, False on failure.
    """
    import subprocess

    cookies_file = Path.home() / "Desktop" / "instagram_cookies.txt"
    cookie_args = (
        ["--cookies", str(cookies_file)]
        if cookies_file.exists()
        else ["--cookies-from-browser", "chrome"]
    )

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-playlist",
        "--socket-timeout", "60",
        *cookie_args,
        "--format", "best[height<=720]/best",
        "--output", str(out_path),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and out_path.exists():
            return True
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        print(f"    yt-dlp failed: {err}")
        return False
    except subprocess.TimeoutExpired:
        print("    yt-dlp timed out after 120s")
        return False
    except FileNotFoundError:
        print("    yt-dlp not found — install with: pip install yt-dlp")
        sys.exit(1)


# ── Main scrape loop ──────────────────────────────────────────────────────────
def _extract_pose(
    video_path: Path,
    attempt_id: str,
    climb_uuid: str,
    sample_fps: float,
    dry_run: bool,
) -> int:
    """Run pose_extractor on a video file. Returns frame count."""
    return process_video(
        str(video_path),
        attempt_id=attempt_id,
        sample_fps=sample_fps,
        min_confidence=0.4,
        dry_run=dry_run,
        verbose=False,
        climb_uuid=climb_uuid,
        filter_climbing=True,
    )


def _fetch_quality_data(attempt_id: str, climb_uuid: str | None) -> tuple[list[dict], str | None]:
    """
    After pose extraction, query back frame metrics + grade for quality scoring.
    Returns (rows, grade) where rows is a list of metric dicts and grade is the V-grade string.
    """
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv(_BASE_DIR / ".env")
        conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://localhost/climbing_platform"))
        cur  = conn.cursor()

        # Frame metrics for consistency + grade alignment scoring
        cur.execute("""
            SELECT hip_angle_deg, com_height_norm, tension_score,
                   left_arm_reach_norm, right_arm_reach_norm
            FROM pose_frames WHERE attempt_id = %s
        """, (attempt_id,))
        rows = [
            {"hip_angle_deg": r[0], "com_height_norm": r[1], "tension_score": r[2],
             "left_arm_reach_norm": r[3], "right_arm_reach_norm": r[4]}
            for r in cur.fetchall()
        ]

        # Grade lookup
        grade = None
        if climb_uuid:
            cur.execute(
                "SELECT community_grade FROM board_routes WHERE UPPER(external_id) = UPPER(%s)",
                (climb_uuid,),
            )
            row = cur.fetchone()
            grade = row[0] if row else None

        cur.close()
        conn.close()
        return rows, grade
    except Exception:
        return [], None


# ── Process a pre-downloaded local directory ───────────────────────────────────
def run_local(
    video_dir: Path,
    dry_run: bool = False,
    sample_fps: float = 5.0,
    max_consecutive_failures: int = 10,
):
    """
    Process already-downloaded videos from a local directory.

    Filename matching (in order of priority):
      1. Filename contains a 32-char hex Kilter climb UUID
         e.g.  00045F0E6F0340ACAE7CF216E1055B8B_reel.mp4
      2. Filename contains a known climb name from kilter.db (partial match)
      3. Falls back to a UUID derived from the filename — frames still saved,
         climb_uuid will be null and can be linked later.
    """
    import re as _re

    video_dir = Path(video_dir).expanduser()
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        sys.exit(1)

    videos = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".m4v")
    )
    if not videos:
        print(f"No video files found in {video_dir}")
        return

    checkpoint = load_checkpoint()
    completed  = checkpoint["completed"]
    failed     = checkpoint["failed"]

    # Build a name→uuid index from kilter.db for soft matching
    conn = sqlite3.connect(KILTER_DB)
    cur  = conn.cursor()
    cur.execute("SELECT uuid, name FROM climbs WHERE name IS NOT NULL")
    name_index = {row[1].lower(): row[0] for row in cur.fetchall()}
    conn.close()

    UUID_RE = _re.compile(r"[0-9a-fA-F]{32}")

    print(f"\nLocal video processing")
    print(f"  Directory : {video_dir}")
    print(f"  Videos    : {len(videos)}")
    if dry_run:
        print("  [dry-run] No DB writes.\n")

    processed = skipped = errors = 0
    total_frames = 0
    consecutive_failures = 0

    for i, vid in enumerate(videos, 1):
        stem = vid.stem

        # 1. 32-char hex UUID in filename?
        m = UUID_RE.search(stem)
        climb_uuid = m.group(0).upper() if m else None

        # 2. Name match?
        if not climb_uuid:
            stem_lower = stem.lower()
            for cname, cuuid in name_index.items():
                if cname in stem_lower or stem_lower in cname:
                    climb_uuid = cuuid
                    break

        # 3. Derive from filename
        if not climb_uuid:
            climb_uuid = None  # will be null in DB; can be linked later

        attempt_id = make_attempt_id(climb_uuid or stem, str(vid))

        if attempt_id in completed:
            skipped += 1
            print(f"[{i}/{len(videos)}] {vid.name} — already done, skip")
            continue

        print(f"[{i}/{len(videos)}] {vid.name}")
        if climb_uuid:
            print(f"    climb_uuid : {climb_uuid}")
        else:
            print(f"    climb_uuid : (unknown — will be null in DB)")

        try:
            n_frames = _extract_pose(vid, attempt_id, climb_uuid or "", sample_fps, dry_run)
            total_frames += n_frames
            print(f"    Pose frames: {n_frames}")
        except Exception as e:
            print(f"    pose_extractor error: {e}")
            failed[str(vid)] = str(e)
            save_checkpoint({"completed": completed, "failed": failed})
            errors += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n✗ {consecutive_failures} consecutive failures — aborting.")
                break
            continue
        else:
            consecutive_failures = 0
            if not dry_run:
                completed[attempt_id] = climb_uuid or stem
                save_checkpoint({"completed": completed, "failed": failed})
            processed += 1

    print(f"\n── Local run complete ───────────────────────────")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}  (already done)")
    print(f"  Errors    : {errors}")
    print(f"  Pose rows : {total_frames:,}")


def run_scrape(
    limit: int | None = None,
    climb_uuid_filter: str | None = None,
    dry_run: bool = False,
    sample_fps: float = 5.0,
    delay_sec: float = 2.0,
    reels_only: bool = False,
    since: str | None = None,
    max_consecutive_failures: int = 10,
    grades: list[str] | None = None,
):
    checkpoint = load_checkpoint()
    completed  = checkpoint["completed"]   # {attempt_id: climb_uuid}
    failed     = checkpoint["failed"]      # {url: reason}

    links = get_beta_links(
        climb_uuid=climb_uuid_filter,
        limit=limit,
        reels_only=reels_only,
        since=since,
        grades=grades,
    )
    print(f"\nBeta links to process : {len(links):,}")
    print(f"Already completed     : {len(completed):,}")
    print(f"Previously failed     : {len(failed):,}")
    if dry_run:
        print("[dry-run] No DB writes.\n")
    else:
        print()

    TMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    processed = skipped = errors = 0
    total_frames = 0
    consecutive_failures = 0

    for i, (c_uuid, url) in enumerate(links, 1):
        attempt_id = make_attempt_id(c_uuid, url)

        # Skip already-done
        if attempt_id in completed:
            skipped += 1
            continue

        print(f"[{i}/{len(links)}] {c_uuid[:12]}… | {url[:60]}")

        # Deterministic tmp filename
        slug    = hashlib.sha256(url.encode()).hexdigest()[:12]
        tmp_mp4 = TMP_VIDEO_DIR / f"{c_uuid[:8]}_{slug}.mp4"

        # Download
        ok = download_video(url, tmp_mp4)
        if not ok:
            print(f"    → skip (download failed)")
            failed[url] = "download_failed"
            save_checkpoint({"completed": completed, "failed": failed})
            errors += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n✗ {consecutive_failures} consecutive download failures — aborting.")
                print(f"  Instagram may be rate-limiting or your session has expired.")
                print(f"  Re-run later or refresh cookies (~/Desktop/instagram_cookies.txt).")
                break
            continue

        # Successful download resets the counter
        consecutive_failures = 0

        size_mb = tmp_mp4.stat().st_size / 1_048_576
        print(f"    Downloaded  {size_mb:.1f} MB")

        # Extract pose
        try:
            n_frames = _extract_pose(tmp_mp4, attempt_id, c_uuid, sample_fps, dry_run)
            total_frames += n_frames
        except Exception as e:
            import traceback
            print(f"    pose_extractor error: {e or type(e).__name__}")
            print(f"    {traceback.format_exc().strip().splitlines()[-1]}")
            failed[url] = str(e) or type(e).__name__
            save_checkpoint({"completed": completed, "failed": failed})
            errors += 1
        else:
            frame_rows, grade = _fetch_quality_data(attempt_id, c_uuid) if not dry_run else ([], None)
            quality = score_attempt(
                attempt_id = attempt_id,
                n_detected = n_frames,
                n_sampled  = n_frames,
                rows       = frame_rows,
                grade      = grade,
            )
            print(f"    Pose frames : {n_frames}  quality={quality.score:.0f}/100 [{quality.label}]")
            if quality.flags:
                print(f"    quality flags: {', '.join(quality.flags)}")
            if not dry_run:
                completed[attempt_id] = c_uuid
                save_checkpoint({"completed": completed, "failed": failed})
            processed += 1
        finally:
            # Always delete tmp file
            if tmp_mp4.exists():
                tmp_mp4.unlink()

        # Rate limiting: base delay + jitter
        jitter = random.uniform(0, 1.5)
        time.sleep(delay_sec + jitter)

    print(f"\n── Run complete ─────────────────────────────────")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}  (already done)")
    print(f"  Errors    : {errors}")
    print(f"  Pose rows : {total_frames:,}")
    print(f"  Checkpoint: {CHECKPOINT_FILE}")


# ── Stats ─────────────────────────────────────────────────────────────────────
def print_stats():
    checkpoint = load_checkpoint()
    completed  = checkpoint["completed"]
    failed     = checkpoint["failed"]

    total = len(get_beta_links())
    done  = len(completed)
    errs  = len(failed)
    left  = total - done - errs

    print(f"\nBeta scrape progress")
    print(f"  Total links  : {total:,}")
    print(f"  Completed    : {done:,}  ({100*done/max(total,1):.1f}%)")
    print(f"  Failed       : {errs:,}")
    print(f"  Remaining    : {left:,}")

    if CHECKPOINT_FILE.exists():
        print(f"  Checkpoint   : {CHECKPOINT_FILE}")

    # PostgreSQL pose_frames summary for beta videos
    try:
        import os
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "climbing_platform"),
            user=os.getenv("DB_USER", "shaincarrasco"),
            password=os.getenv("DB_PASSWORD", "") or None,
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT climb_uuid) FROM pose_frames WHERE climb_uuid IS NOT NULL")
        frames, climbs = cur.fetchone()
        cur.close()
        conn.close()
        print(f"\npose_frames (beta):")
        print(f"  Total rows   : {frames:,}")
        print(f"  Unique routes: {climbs:,}")
    except Exception as e:
        print(f"\n  (DB unavailable: {e})")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Kilter beta videos, extract pose, store in PostgreSQL"
    )
    parser.add_argument("--limit",      type=int,  help="Max number of videos to process")
    parser.add_argument("--climb-uuid", help="Only process links for this climb UUID")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Download and extract pose but do not write to DB or checkpoint")
    parser.add_argument("--stats",      action="store_true",
                        help="Show progress stats and exit")
    parser.add_argument("--sample-fps", type=float, default=5.0,
                        help="Pose sampling rate in fps (default: 5)")
    parser.add_argument("--delay",      type=float, default=2.0,
                        help="Base delay between downloads in seconds (default: 2)")
    parser.add_argument("--reels-only", action="store_true",
                        help="Only process /reel/ links (skip old /p/ posts, higher success rate)")
    parser.add_argument("--since",      default=None,
                        help="Only process links created after this date e.g. 2022-01-01")
    parser.add_argument("--grades",     default=None,
                        help="Comma-separated V-grades to target e.g. V8,V9,V10  "
                             "Only scrapes beta for routes at those grades.")
    parser.add_argument("--local-dir",  default=None,
                        help="Process already-downloaded videos from this directory instead of "
                             "fetching from kilter.db beta_links. Matches climb UUIDs from "
                             "filenames automatically.")
    parser.add_argument("--max-consecutive-failures", type=int, default=10,
                        help="Abort after this many consecutive download failures (default: 10). "
                             "Prevents wasting time when Instagram is rate-limiting.")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.local_dir:
        run_local(
            video_dir=Path(args.local_dir),
            dry_run=args.dry_run,
            sample_fps=args.sample_fps,
            max_consecutive_failures=args.max_consecutive_failures,
        )
        return

    grades = [g.strip().upper() for g in args.grades.split(",")] if args.grades else None

    run_scrape(
        limit=args.limit,
        climb_uuid_filter=args.climb_uuid,
        dry_run=args.dry_run,
        sample_fps=args.sample_fps,
        delay_sec=args.delay,
        reels_only=args.reels_only,
        since=args.since,
        max_consecutive_failures=args.max_consecutive_failures,
        grades=grades,
    )


if __name__ == "__main__":
    main()

"""
pipeline/youtube_scraper.py
---------------------------
Scans YouTube for real climbing videos — indoor gym climbs, outdoor boulder
problems, competition footage, technique breakdowns — and builds a broad pose
library that goes beyond Kilter Board data.

The goal is NOT to match videos to specific routes. It's to accumulate as many
real climbing movement patterns as possible across grades, styles, and settings
so the pose model learns what V6 movement looks like in general, not just on
one specific board.

Pose frames are stored in PostgreSQL with metadata (grade, style, setting,
source URL) tracked in data/climb_scan_progress.json alongside the checkpoint.

Search strategies
-----------------
  --mode grade        Search for climbing by V-grade or font grade
                      e.g.  "V8 bouldering", "Font 7c boulder"

  --mode style        Search by movement style / technique
                      e.g.  "heel hook bouldering", "drop knee climbing"

  --mode outdoor      Outdoor locations and crags
                      e.g.  "Bishop bouldering", "Magic Wood", "Fontainebleau"

  --mode competition  Competition climbing footage (IFSC, nationals, etc.)

  --mode channel      Curated climbing channels (default — most reliable)

  --mode all          Run all strategies in sequence

Usage
-----
    # Scan by grade, 30 videos
    python pipeline/youtube_scraper.py --mode grade --limit 30

    # Outdoor climbing only
    python pipeline/youtube_scraper.py --mode outdoor --limit 50

    # Everything, no limit — run overnight
    python pipeline/youtube_scraper.py --mode all

    # See what's been collected
    python pipeline/youtube_scraper.py --stats

    # Dry run — download + pose but no DB writes
    python pipeline/youtube_scraper.py --mode grade --limit 5 --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR       = Path(__file__).resolve().parent.parent
CHECKPOINT_FILE = _BASE_DIR / "data" / "climb_scan_progress.json"
TMP_VIDEO_DIR   = Path("/tmp/climb_scan_videos")

sys.path.insert(0, str(_BASE_DIR / "pipeline"))
from pose_extractor import process_video
from data_quality import score_attempt


# ── Search query banks ────────────────────────────────────────────────────────

GRADE_QUERIES = [
    # V-scale sends — explicit grade + send language maximises grade-aligned pose data
    "V3 boulder send", "V4 boulder send", "V5 boulder send",
    "V6 boulder send", "V7 boulder send", "V8 boulder send",
    "V9 boulder send", "V10 boulder send", "V11 boulder send",
    "V12 boulder send",
    "V5 flash bouldering", "V6 flash bouldering", "V7 flash bouldering",
    "V8 flash bouldering", "V9 flash bouldering", "V10 flash bouldering",
    "V7 first ascent", "V8 first ascent", "V9 first ascent",
    # Font scale (European outdoor)
    "Font 6c send", "Font 7a send", "Font 7b send",
    "Font 7c send", "Font 8a send", "Font 8b boulder",
    # Kilter Board grade sends (most directly useful for our model)
    "kilter board V7", "kilter board V8", "kilter board V9",
    "kilter board V10", "kilter board V6", "kilter board V5",
    "tension board V7", "tension board V8", "tension board V6",
    "moonboard V7", "moonboard V8", "moonboard V9",
]

# Keywords that strongly indicate a tutorial/explainer rather than an actual send
_TUTORIAL_TITLE_KW = [
    "how to", "tutorial", "tips", "guide", "lesson", "mistakes",
    "improve", "training for", "exercise", "drills", "beginner",
    "101", "explained", "learn", "secrets of", "technique for",
]

STYLE_QUERIES = [
    # Move types — append "problem" or "send" to get actual climbs not tutorials
    "heel hook bouldering problem",
    "toe hook bouldering problem",
    "drop knee boulder send",
    "dyno boulder send",
    "compression bouldering send",
    "slab boulder problem send",
    "overhang boulder problem",
    "pinch hold bouldering",
    "sloper bouldering problem",
    "undercling boulder send",
    "high step boulder problem",
    "knee bar climbing problem",
    "mantle problem send",
    "tension board problem",
    "system board send",
]

OUTDOOR_QUERIES = [
    # Classic outdoor bouldering areas
    "Bishop bouldering problem",
    "Fontainebleau bouldering",
    "Magic Wood bouldering",
    "Rocklands bouldering",
    "Hueco Tanks bouldering",
    "Joe's Valley bouldering",
    "Buttermilks bouldering",
    "Rifle climbing",
    "Red River Gorge climbing",
    "Yosemite bouldering",
    "Squamish bouldering",
    "Ticino bouldering",
    "Cresciano bouldering",
    "Chironico bouldering",
    # General outdoor
    "outdoor bouldering problem",
    "outdoor climbing send",
    "sport climbing redpoint",
    "trad climbing crack",
]

COMPETITION_QUERIES = [
    "IFSC World Cup bouldering",
    "IFSC bouldering final",
    "USA Climbing bouldering nationals",
    "climbing world championships bouldering",
    "Olympic climbing bouldering",
    "bouldering competition finals",
    "IFSC lead climbing",
    "sport climbing competition",
]

# Curated channels with high climbing content density
CLIMBING_CHANNELS = [
    "https://www.youtube.com/@JörgVerhoeven",
    "https://www.youtube.com/@nalle.hukkataival",
    "https://www.youtube.com/@EpicTV",
    "https://www.youtube.com/@climbingdailytv",
    "https://www.youtube.com/@LauraKoglerClimbing",
    "https://www.youtube.com/@adambondclimbing",
    "https://www.youtube.com/@tensionclimbing",
    "https://www.youtube.com/@benchmarkclimbing",
    "https://www.youtube.com/@powercompanyClimbing",
    "https://www.youtube.com/@TomRandall",
    "https://www.youtube.com/@NickMullerClimbing",
]

# All queries combined for --mode all
ALL_QUERIES = GRADE_QUERIES + STYLE_QUERIES + OUTDOOR_QUERIES + COMPETITION_QUERIES

QUERIES_BY_MODE = {
    "grade":       GRADE_QUERIES,
    "style":       STYLE_QUERIES,
    "outdoor":     OUTDOOR_QUERIES,
    "competition": COMPETITION_QUERIES,
    "channel":     [],   # handled separately
    "all":         ALL_QUERIES,
}


# ── Metadata extraction ───────────────────────────────────────────────────────

_VGRADE_RE   = re.compile(r'\bV(\d{1,2})\b', re.IGNORECASE)
_FONT_RE     = re.compile(r'\b(6[abc][+]?|7[abc][+]?|8[abc][+]?|9[ab][+]?)\b', re.IGNORECASE)
_OUTDOOR_KW  = {"outdoor","fontainebleau","bishop","magic wood","rocklands","yosemite",
                "hueco","squamish","ticino","cresciano","chironico","joe's valley",
                "rifle","red river","font","crag","rockface","sandstone","granite","limestone"}
_COMP_KW     = {"ifsc","world cup","championship","competition","finals","nationals","olympic"}
_STYLE_KW    = {
    "heel hook","toe hook","drop knee","dyno","campus","compression","slab","overhang",
    "pinch","crimp","sloper","undercling","sidepull","high step","flag","knee bar",
    "mantle","dead point","coordination","tension board",
}


def parse_video_meta(title: str, description: str = "") -> dict:
    """
    Extract structured metadata from a YouTube video title + description.
    Returns dict with: vgrade, font_grade, setting, styles, is_competition.
    """
    combined = (title + " " + description).lower()

    vgrade = None
    m = _VGRADE_RE.search(title)
    if m:
        vgrade = "V" + m.group(1)

    font_grade = None
    m2 = _FONT_RE.search(title)
    if m2:
        font_grade = m2.group(0)

    is_outdoor    = any(kw in combined for kw in _OUTDOOR_KW)
    is_comp       = any(kw in combined for kw in _COMP_KW)
    setting       = "outdoor" if is_outdoor else "competition" if is_comp else "indoor"

    styles = [kw for kw in _STYLE_KW if kw in combined]

    return {
        "vgrade":       vgrade,
        "font_grade":   font_grade,
        "setting":      setting,
        "styles":       styles,
        "is_competition": is_comp,
    }


def _is_climbing_video(title: str, strict: bool = True) -> bool:
    """Filter out non-climbing content."""
    tl = title.lower()
    # Hard exclusions
    if any(w in tl for w in ["minecraft","gaming","vlog","cooking","music","podcast",
                               "reaction","news","politics","review","unboxing"]):
        return False
    # Must contain a climbing indicator
    climbing_signals = ["climb","boulder","V0","V1","V2","V3","V4","V5","V6","V7","V8",
                        "V9","V10","V11","V12","font","ifsc","dyno","crimp","sloper",
                        "heel hook","send","flash","redpoint","on-sight","crux","beta",
                        "overhang","slab","crack","gym","crag","outdoor problem"]
    return any(s.lower() in tl for s in climbing_signals)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}, "video_meta": {}}


def save_checkpoint(cp: dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)


# ── attempt_id ───────────────────────────────────────────────────────────────

def make_attempt_id(video_id: str) -> str:
    raw = hashlib.sha256(f"scan:{video_id}".encode()).hexdigest()[:32]
    return str(uuid.UUID(raw))


# ── yt-dlp helpers ────────────────────────────────────────────────────────────

def search_youtube(query: str, max_results: int = 8) -> list[dict]:
    """Search YouTube, return list of {id, title, url, duration_sec}."""
    r = subprocess.run(
        ["yt-dlp", "--dump-json", "--flat-playlist", "--no-warnings", "--quiet",
         f"ytsearch{max_results}:{query}"],
        capture_output=True, text=True, timeout=60,
    )
    results = []
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
            vid_id = d.get("id", "")
            title  = d.get("title", "")
            dur    = d.get("duration")
            if vid_id and title:
                results.append({
                    "id":           vid_id,
                    "title":        title,
                    "url":          f"https://www.youtube.com/watch?v={vid_id}",
                    "duration_sec": dur,
                })
        except json.JSONDecodeError:
            pass
    return results


def get_channel_videos(channel_url: str, max_videos: int = 60) -> list[dict]:
    """Fetch video list from a YouTube channel."""
    r = subprocess.run(
        ["yt-dlp", "--dump-json", "--flat-playlist", "--no-warnings", "--quiet",
         "--playlist-end", str(max_videos), channel_url],
        capture_output=True, text=True, timeout=120,
    )
    results = []
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
            vid_id = d.get("id", "")
            title  = d.get("title", "")
            dur    = d.get("duration")
            if vid_id and title not in ("[Private video]", "[Deleted video]"):
                results.append({
                    "id":           vid_id,
                    "title":        title,
                    "url":          f"https://www.youtube.com/watch?v={vid_id}",
                    "duration_sec": dur,
                })
        except json.JSONDecodeError:
            pass
    return results


def download_video(url: str, out_path: Path) -> bool:
    """Download a YouTube video (max 720p mp4). Returns True on success."""
    try:
        r = subprocess.run(
            ["yt-dlp", "--quiet", "--no-playlist", "--socket-timeout", "60",
             "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
             "--merge-output-format", "mp4",
             "--output", str(out_path), url],
            capture_output=True, text=True, timeout=180,
        )
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 50_000:
            return True
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown"
        print(f"    yt-dlp: {last}")
        return False
    except subprocess.TimeoutExpired:
        print("    yt-dlp timed out")
        return False
    except FileNotFoundError:
        print("yt-dlp not found — pip install yt-dlp")
        sys.exit(1)


# ── Core processing ───────────────────────────────────────────────────────────

def _should_skip_video(
    vid: dict,
    min_dur: int,
    max_dur: int,
    require_grade: bool = False,
    skip_tutorials: bool = True,
) -> str | None:
    """Returns a skip reason string, or None if the video should be processed."""
    dur = vid.get("duration_sec")
    if dur is not None:
        if dur < min_dur:
            return f"too short ({dur}s)"
        if dur > max_dur:
            return f"too long ({dur}s)"
    if not _is_climbing_video(vid["title"]):
        return "not climbing"
    title_lower = vid["title"].lower()
    if skip_tutorials and any(kw in title_lower for kw in _TUTORIAL_TITLE_KW):
        return "tutorial (low training value)"
    if require_grade and not _VGRADE_RE.search(vid["title"]) and not _FONT_RE.search(vid["title"]):
        return "no grade in title"
    return None


def process_video_entry(
    vid: dict,
    cp: dict,
    dry_run: bool,
    sample_fps: float,
) -> tuple[bool, int]:
    """
    Download, extract pose, update checkpoint for one video.
    Returns (success, frame_count).
    """
    vid_id     = vid["id"]
    title      = vid["title"]
    url        = vid["url"]
    attempt_id = make_attempt_id(vid_id)

    if attempt_id in cp["completed"]:
        return None, 0   # None = already done (skipped)

    tmp_mp4 = TMP_VIDEO_DIR / f"scan_{vid_id}.mp4"

    if not download_video(url, tmp_mp4):
        cp["failed"][url] = "download_failed"
        save_checkpoint(cp)
        return False, 0

    size_mb = tmp_mp4.stat().st_size / 1_048_576
    print(f"    {size_mb:.1f} MB  |  {title[:55]}")

    meta = parse_video_meta(title)

    try:
        n_frames = process_video(
            str(tmp_mp4),
            attempt_id=attempt_id,
            sample_fps=sample_fps,
            min_confidence=0.4,
            dry_run=dry_run,
            verbose=False,
            climb_uuid=None,     # no route match — general climbing library
            filter_climbing=True,
        )
    except Exception as e:
        import traceback
        print(f"    pose_extractor error: {e or type(e).__name__}")
        print(f"    {traceback.format_exc().strip().splitlines()[-1]}")
        cp["failed"][url] = str(e) or type(e).__name__
        save_checkpoint(cp)
        return False, 0
    else:
        grade = meta.get("vgrade") or meta.get("font_grade")
        # Estimate n_sampled from video duration × sample_fps (rough)
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(str(tmp_mp4)) if tmp_mp4.exists() else None
            n_sampled = n_frames  # fallback — file may already be deleted
            if _cap and _cap.isOpened():
                src_fps = _cap.get(_cv2.CAP_PROP_FPS) or 10.0
                total   = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
                n_sampled = max(n_frames, total // max(1, int(round(src_fps / sample_fps))))
                _cap.release()
        except Exception:
            n_sampled = n_frames

        quality = score_attempt(
            attempt_id = attempt_id,
            n_detected = n_frames,
            n_sampled  = n_sampled,
            rows       = [],          # no in-memory rows (already written to DB)
            grade      = grade,
        )

        grade_str = grade or "?"
        print(f"    Pose frames: {n_frames}  |  {meta['setting']}  grade={grade_str}"
              f"  styles={meta['styles'][:2]}  quality={quality.score:.0f}/100 [{quality.label}]")
        if quality.flags:
            print(f"    quality flags: {', '.join(quality.flags)}")

        if not dry_run:
            cp["completed"][attempt_id] = {
                "video_id":     vid_id,
                "title":        title,
                "url":          url,
                "quality_score": quality.score,
                "quality_label": quality.label,
                "quality_flags": quality.flags,
                **meta,
            }
            cp["video_meta"][vid_id] = {**meta, "quality_score": quality.score}
            save_checkpoint(cp)
        return True, n_frames
    finally:
        if tmp_mp4.exists():
            tmp_mp4.unlink()


# ── Scan modes ────────────────────────────────────────────────────────────────

def run_query_scan(
    queries: list[str],
    limit: int | None,
    dry_run: bool,
    sample_fps: float,
    results_per_query: int,
    delay_sec: float,
    max_consecutive_failures: int,
    min_dur: int = 20,
    max_dur: int = 600,
    require_grade: bool = False,
    skip_tutorials: bool = True,
):
    cp = load_checkpoint()
    TMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    processed = skipped = errors = consecutive_failures = 0
    total_frames = 0
    total_attempted = 0

    random.shuffle(queries)   # vary order each run so no query monopolises

    for q in queries:
        if limit and total_attempted >= limit:
            break

        print(f"\nSearch: {q!r}")
        videos = search_youtube(q, max_results=results_per_query)

        for vid in videos:
            if limit and total_attempted >= limit:
                break

            skip = _should_skip_video(vid, min_dur, max_dur,
                                      require_grade=require_grade,
                                      skip_tutorials=skip_tutorials)
            if skip:
                continue

            total_attempted += 1
            print(f"  [{total_attempted}] {vid['title'][:70]}")

            ok, n = process_video_entry(vid, cp, dry_run, sample_fps)
            if ok is None:
                skipped += 1
            elif ok:
                consecutive_failures = 0
                processed += 1
                total_frames += n
            else:
                errors += 1
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n✗ {consecutive_failures} consecutive failures — aborting.")
                    _print_summary(processed, skipped, errors, total_frames)
                    return

            time.sleep(delay_sec + random.uniform(0, 1.0))

    _print_summary(processed, skipped, errors, total_frames)


def run_channel_scan(
    channels: list[str],
    limit: int | None,
    dry_run: bool,
    sample_fps: float,
    delay_sec: float,
    max_consecutive_failures: int,
    min_dur: int = 20,
    max_dur: int = 600,
):
    cp = load_checkpoint()
    TMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    processed = skipped = errors = consecutive_failures = 0
    total_frames = 0
    total_attempted = 0

    for ch in channels:
        if limit and total_attempted >= limit:
            break

        per_ch = (limit - total_attempted) if limit else 80
        print(f"\nChannel: {ch}")
        videos = get_channel_videos(ch, max_videos=per_ch)
        print(f"  {len(videos)} videos found")

        for vid in videos:
            if limit and total_attempted >= limit:
                break

            skip = _should_skip_video(vid, min_dur, max_dur)
            if skip:
                print(f"  skip ({skip}): {vid['title'][:50]}")
                continue

            total_attempted += 1
            print(f"  [{total_attempted}] {vid['title'][:70]}")

            ok, n = process_video_entry(vid, cp, dry_run, sample_fps)
            if ok is None:
                skipped += 1
            elif ok:
                consecutive_failures = 0
                processed += 1
                total_frames += n
            else:
                errors += 1
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n✗ {consecutive_failures} consecutive failures — aborting.")
                    _print_summary(processed, skipped, errors, total_frames)
                    return

            time.sleep(delay_sec + random.uniform(0, 0.5))

    _print_summary(processed, skipped, errors, total_frames)


def _print_summary(processed, skipped, errors, total_frames):
    print(f"\n── Scan complete ────────────────────────────────")
    print(f"  Processed  : {processed}")
    print(f"  Skipped    : {skipped}  (already done)")
    print(f"  Errors     : {errors}")
    print(f"  Pose rows  : {total_frames:,}")


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats():
    cp = load_checkpoint()

    completed = cp.get("completed", {})
    failed    = cp.get("failed", {})
    meta      = cp.get("video_meta", {})

    print(f"\nClimbing video scan progress")
    print(f"  Processed  : {len(completed):,}")
    print(f"  Failed     : {len(failed):,}")

    # Break down by setting and grade
    settings = {}
    grades   = {}
    for v in completed.values():
        if isinstance(v, dict):
            s = v.get("setting", "unknown")
            settings[s] = settings.get(s, 0) + 1
            g = v.get("vgrade") or v.get("font_grade") or "unknown"
            grades[g] = grades.get(g, 0) + 1

    if settings:
        print(f"\n  By setting:")
        for s, n in sorted(settings.items(), key=lambda x: -x[1]):
            print(f"    {s:<14} {n}")
    if grades:
        top_grades = sorted(grades.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top grades detected:")
        for g, n in top_grades:
            print(f"    {g:<10} {n}")

    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME","climbing_platform"),
            user=os.getenv("DB_USER","shaincarrasco"),
            password=os.getenv("DB_PASSWORD","") or None,
            host=os.getenv("DB_HOST","localhost"),
            port=os.getenv("DB_PORT","5432"),
        )
        cur = conn.cursor()
        scan_ids = list(completed.keys())
        if scan_ids:
            cur.execute(
                "SELECT COUNT(*), COUNT(DISTINCT attempt_id) FROM pose_frames WHERE attempt_id = ANY(%s)",
                (scan_ids,)
            )
            frames, attempts = cur.fetchone()
            print(f"\n  pose_frames (scan):")
            print(f"    Pose rows : {frames:,}")
            print(f"    Videos    : {attempts:,}")
        cur.close(); conn.close()
    except Exception as e:
        print(f"\n  (DB unavailable: {e})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan YouTube for real climbing videos, extract pose data"
    )
    parser.add_argument("--mode", choices=["grade","style","outdoor","competition","channel","all"],
                        default="channel",
                        help="Search strategy (default: channel). 'all' runs every query bank.")
    parser.add_argument("--limit",      type=int,
                        help="Max videos to process")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Download + extract but no DB writes")
    parser.add_argument("--stats",      action="store_true",
                        help="Show progress stats and exit")
    parser.add_argument("--sample-fps", type=float, default=5.0,
                        help="Pose sampling rate (default: 5 fps)")
    parser.add_argument("--delay",      type=float, default=1.0,
                        help="Base delay between downloads in seconds (default: 1)")
    parser.add_argument("--results-per-query", type=int, default=6,
                        help="YouTube search results per query (default: 6)")
    parser.add_argument("--max-consecutive-failures", type=int, default=10)
    parser.add_argument("--min-duration", type=int, default=20,
                        help="Skip videos shorter than this many seconds (default: 20)")
    parser.add_argument("--max-duration", type=int, default=600,
                        help="Skip videos longer than this many seconds (default: 600 = 10 min)")
    parser.add_argument("--channels",   default=None,
                        help="Comma-separated channel URLs to override the built-in list")
    parser.add_argument("--require-grade", action="store_true",
                        help="Only process videos with a V-grade or font grade in the title")
    parser.add_argument("--allow-tutorials", action="store_true",
                        help="Don't filter out tutorial/technique/how-to videos")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    channels = CLIMBING_CHANNELS
    if args.channels:
        channels = [
            c.strip() if c.strip().startswith("http")
            else f"https://www.youtube.com/{c.strip()}"
            for c in args.channels.split(",")
        ]

    common = dict(
        limit=args.limit,
        dry_run=args.dry_run,
        sample_fps=args.sample_fps,
        delay_sec=args.delay,
        max_consecutive_failures=args.max_consecutive_failures,
        min_dur=args.min_duration,
        max_dur=args.max_duration,
    )

    if args.mode == "channel":
        run_channel_scan(channels=channels, **common)
    else:
        queries = QUERIES_BY_MODE.get(args.mode, ALL_QUERIES)
        run_query_scan(
            queries=queries,
            results_per_query=args.results_per_query,
            require_grade=args.require_grade,
            skip_tutorials=not args.allow_tutorials,
            **common,
        )


if __name__ == "__main__":
    main()

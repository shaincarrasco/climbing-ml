"""
pipeline/beta_scraper.py

Downloads Instagram/TikTok beta videos from kilter.db beta_links table,
runs each through pose_extractor, saves pose_frames to Supabase, deletes video.

Usage:
    python3 pipeline/beta_scraper.py --limit 10
    python3 pipeline/beta_scraper.py --dry-run --limit 5
    python3 pipeline/beta_scraper.py --climb-uuid 00045F0E6F0340ACAE7CF216E1055B8B
    python3 pipeline/beta_scraper.py --stats
"""

import os, sys, json, time, random, hashlib, sqlite3, argparse, tempfile, subprocess, uuid as uuid_mod
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SQLITE_PATH  = os.path.expanduser(os.getenv("KILTER_DB_PATH", "~/Desktop/ClimbingML/kilter.db"))
CHECKPOINT   = Path(__file__).parent.parent / "data" / "beta_scrape_progress.json"
VIDEO_TMPDIR = Path(tempfile.gettempdir()) / "beta_videos"

def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            cp = json.load(f)
        # Migrate old format: completed was {attempt_id: climb_uuid} dict
        # New format: completed is a list of URLs for easy dedup
        if isinstance(cp.get("completed"), dict):
            cp["completed"] = []  # old attempt_id keys are useless for URL dedup
        return cp
    return {"completed": [], "failed": {}, "last_run": None}

def save_checkpoint(cp):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2)

def make_attempt_id(climb_uuid: str, link: str) -> str:
    digest = hashlib.sha256((climb_uuid + link).encode()).hexdigest()[:32]
    return str(uuid_mod.UUID(digest))

def get_beta_links(climb_uuid=None, limit=None):
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()
    if climb_uuid:
        cur.execute("SELECT climb_uuid, link, angle FROM beta_links WHERE is_listed=1 AND climb_uuid=? ORDER BY created_at", (climb_uuid,))
    else:
        cur.execute("SELECT climb_uuid, link, angle FROM beta_links WHERE is_listed=1 ORDER BY created_at")
    rows = cur.fetchall()
    con.close()
    return rows[:limit] if limit else rows

COOKIES_FILE = Path(__file__).parent / "instagram_cookies.txt"

def download_video(url: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--merge-output-format", "mp4",
        "--no-playlist", "--socket-timeout", "60", "--retries", "2",
        "--output", str(out_dir / "%(id)s.%(ext)s"),
        "--print", "after_move:filepath", "--quiet", url,
    ]
    if COOKIES_FILE.exists() and "instagram.com" in url:
        cmd = cmd[:1] + ["--cookies", str(COOKIES_FILE)] + cmd[1:]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            path = result.stdout.strip().splitlines()[-1]
            if path and os.path.exists(path):
                return path
        print(f"    yt-dlp error: {result.stderr.strip()[:200]}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    yt-dlp failed: {e}")
    return None

def run_pose_extractor(video_path: str, attempt_id: str, climb_uuid: str) -> bool:
    script = Path(__file__).parent / "pose_extractor.py"
    cmd = [sys.executable, str(script), "--video", video_path, "--attempt-id", attempt_id, "--climb-uuid", climb_uuid]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return True
        print(f"    pose_extractor error: {result.stderr.strip()[-300:]}")
    except subprocess.TimeoutExpired:
        print("    pose_extractor timed out")
    return False

def run(limit=None, climb_uuid_filter=None, dry_run=False):
    cp = load_checkpoint()
    completed_set = set(cp["completed"])
    rows = get_beta_links(climb_uuid=climb_uuid_filter)
    pending = [(cu, lk, ang) for cu, lk, ang in rows if lk not in completed_set]
    if limit:
        pending = pending[:limit]

    print(f"Beta scraper — {len(pending)} links to process (dry_run={dry_run})")
    print(f"Checkpoint: {len(completed_set)} completed, {len(cp['failed'])} failed\n")

    done = failed = 0
    for i, (climb_uuid, link, angle) in enumerate(pending, 1):
        attempt_id = make_attempt_id(climb_uuid, link)
        print(f"[{i}/{len(pending)}] {link[:70]}")
        print(f"         climb={climb_uuid[:12]}… attempt={attempt_id[:8]}…")

        if dry_run:
            print("         [DRY RUN]\n")
            done += 1
            continue

        video_path = download_video(link, VIDEO_TMPDIR)
        if not video_path:
            cp["failed"][link] = {"reason": "download_failed", "ts": datetime.utcnow().isoformat()}
            save_checkpoint(cp)
            failed += 1
            print()
            continue

        print(f"         downloaded {os.path.getsize(video_path)/1024/1024:.1f} MB")
        success = run_pose_extractor(video_path, attempt_id, climb_uuid)
        try: os.remove(video_path)
        except OSError: pass

        if success:
            cp["completed"].append(link)
            completed_set.add(link)
            done += 1
            print("         ✓ pose extracted")
        else:
            cp["failed"][link] = {"reason": "pose_failed", "ts": datetime.utcnow().isoformat()}
            failed += 1
            print("         ✗ pose extraction failed")

        save_checkpoint(cp)
        delay = 2.0 + random.uniform(0, 1.0)
        time.sleep(delay)

    cp["last_run"] = datetime.utcnow().isoformat()
    save_checkpoint(cp)
    print(f"\nDone — {done} succeeded, {failed} failed")

def show_stats():
    cp = load_checkpoint()
    total = len(get_beta_links())
    print(f"Total links : {total:,}")
    print(f"Completed   : {len(cp['completed']):,}")
    print(f"Failed      : {len(cp['failed']):,}")
    print(f"Remaining   : {total - len(cp['completed']) - len(cp['failed']):,}")
    print(f"Last run    : {cp.get('last_run', 'never')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--climb-uuid", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    if args.stats:
        show_stats()
    else:
        run(limit=args.limit, climb_uuid_filter=args.climb_uuid, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

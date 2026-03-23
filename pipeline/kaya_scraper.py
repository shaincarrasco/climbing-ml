"""
pipeline/kaya_scraper.py
------------------------
Scrapes beta videos from the Kaya climbing app (kayaclimb.com).

Kaya does not publish a public API, so this scraper uses your browser session
token obtained from DevTools. Videos are downloaded directly from Kaya's CDN.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO GET YOUR KAYA AUTH TOKEN (one-time setup):
  1. Open https://kaya-app.kayaclimb.com in Chrome/Firefox
  2. Log in to your Kaya account
  3. Open DevTools (F12 → Network tab)
  4. Refresh the page and click any request to api.kayaclimb.com
  5. In Request Headers, find the "Authorization: Bearer <token>" value
  6. Copy the token and paste it below, OR set KAYA_TOKEN in your .env

The token typically lasts 7-30 days. When it expires, repeat the steps above.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
    # Discover Kaya API endpoints from your browser (step 1)
    python pipeline/kaya_scraper.py --discover

    # Search Kilter Board routes with videos, dry run
    python pipeline/kaya_scraper.py --dry-run --limit 5

    # Full run, 50 videos matched to your kilter.db climb UUIDs
    python pipeline/kaya_scraper.py --limit 50

    # Stats
    python pipeline/kaya_scraper.py --stats
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import uuid
import random
from pathlib import Path

import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR        = Path(__file__).resolve().parent.parent
KILTER_DB        = _BASE_DIR / "kilter.db"
CHECKPOINT_FILE  = _BASE_DIR / "data" / "kaya_scrape_progress.json"
TMP_VIDEO_DIR    = Path("/tmp/kaya_videos")

# ── Kaya API constants (discovered via DevTools) ───────────────────────────────
# These endpoints were identified by intercepting Kaya web app traffic.
# If they change, re-discover using --discover mode.
KAYA_API_BASE    = "https://api.kayaclimb.com/v1"
KAYA_APP_BASE    = "https://kaya-app.kayaclimb.com"

# Kilter Board gym IDs on Kaya (look up via --discover or gym search)
# Primary Kilter Board area slug used in Kaya search
KILTER_BOARD_KEYWORDS = ["kilter board", "kilterboard", "kilter"]

sys.path.insert(0, str(_BASE_DIR / "pipeline"))
from pose_extractor import process_video


# ── Auth ──────────────────────────────────────────────────────────────────────
def get_auth_token() -> str:
    """
    Returns Kaya auth token from env var or .env file.
    Set KAYA_TOKEN in your .env file or environment.
    """
    # Try env var first
    token = os.environ.get("KAYA_TOKEN", "").strip()
    if token:
        return token

    # Try .env file
    env_path = _BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("KAYA_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                if token:
                    return token

    return ""


def make_session(token: str) -> requests.Session:
    """Create a requests session with Kaya auth headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Origin":  KAYA_APP_BASE,
        "Referer": KAYA_APP_BASE + "/",
    })
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


# ── Kaya API helpers ──────────────────────────────────────────────────────────
def search_kilter_routes(session: requests.Session, page: int = 1, per_page: int = 50) -> list[dict]:
    """
    Search Kaya for Kilter Board routes that have beta videos.
    Returns list of route dicts.
    """
    # Kaya's public route search endpoint (discovered via DevTools)
    # Format: /climbs/search?q=kilter+board&has_beta=true&page=1
    params = {
        "q":        "kilter board",
        "has_beta": "true",
        "page":     page,
        "per_page": per_page,
    }
    try:
        resp = session.get(f"{KAYA_API_BASE}/climbs/search", params=params, timeout=30)
        if resp.status_code == 401:
            print("⚠  Kaya returned 401 — your token is expired or missing.")
            print("   Re-run `--discover` and update KAYA_TOKEN in your .env")
            return []
        resp.raise_for_status()
        data = resp.json()
        # Handle both {data: [...]} and plain list responses
        if isinstance(data, list):
            return data
        return data.get("data", data.get("climbs", data.get("results", [])))
    except requests.RequestException as e:
        print(f"  Kaya API error: {e}")
        return []


def get_route_betas(session: requests.Session, route_id: str) -> list[dict]:
    """
    Fetch beta video list for a specific Kaya route.
    Returns list of beta dicts with 'video_url' or 'url' keys.
    """
    try:
        resp = session.get(f"{KAYA_API_BASE}/climbs/{route_id}/betas", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("data", data.get("betas", []))
    except requests.RequestException:
        return []


def extract_video_url(beta: dict) -> str | None:
    """Pull the actual mp4 URL from a beta dict (format varies by Kaya version)."""
    for key in ("video_url", "url", "media_url", "file_url", "hls_url"):
        val = beta.get(key, "")
        if val and (val.endswith(".mp4") or "video" in val.lower() or ".m3u8" in val):
            return val
    return None


def download_kaya_video(session: requests.Session, url: str, out_path: Path) -> bool:
    """Download a Kaya video directly (they're typically plain mp4 CDN links)."""
    # Try yt-dlp first (handles HLS, auth cookies, etc.)
    import subprocess
    token = get_auth_token()
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-playlist",
        "--socket-timeout", "60",
        "--format", "best[height<=720]/best",
        "--output", str(out_path),
    ]
    if token:
        cmd += ["--add-header", f"Authorization:Bearer {token}"]
    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and out_path.exists():
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: direct HTTP download
    try:
        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return out_path.exists() and out_path.stat().st_size > 10_000
    except Exception as e:
        print(f"    Direct download failed: {e}")
        return False


# ── kilter.db helpers ─────────────────────────────────────────────────────────
def load_kilter_climbs() -> dict[str, str]:
    """
    Returns {normalized_name: climb_uuid} from kilter.db.
    Used to match Kaya routes back to our climb UUIDs.
    """
    conn = sqlite3.connect(KILTER_DB)
    cur  = conn.cursor()
    cur.execute("SELECT uuid, name FROM climbs WHERE name IS NOT NULL")
    result = {}
    for row in cur.fetchall():
        norm = _normalize(row[1])
        result[norm] = row[0]
    conn.close()
    return result


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation for fuzzy name matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_climb_uuid(kaya_name: str, kilter_index: dict[str, str]) -> str | None:
    """Find a kilter.db climb UUID for a Kaya route name."""
    return kilter_index.get(_normalize(kaya_name))


# ── Deterministic attempt_id ──────────────────────────────────────────────────
def make_attempt_id(climb_uuid: str, url: str) -> str:
    raw = hashlib.sha256(f"kaya:{climb_uuid}{url}".encode()).hexdigest()[:32]
    return str(uuid.UUID(raw))


# ── Checkpoint ────────────────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_checkpoint(checkpoint: dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ── Discover mode ─────────────────────────────────────────────────────────────
def run_discover(token: str):
    """
    Probe Kaya API endpoints to discover the correct structure.
    Useful when the API changes or to validate your token.
    """
    session = make_session(token)
    print("\n── Kaya API Discovery ────────────────────────────────────────────")
    print(f"Token: {'✓ present' if token else '✗ missing (set KAYA_TOKEN in .env)'}\n")

    endpoints_to_probe = [
        f"{KAYA_API_BASE}/climbs/search?q=kilter+board&per_page=3",
        f"{KAYA_API_BASE}/gyms/search?q=kilter&per_page=5",
        f"{KAYA_APP_BASE}/api/v1/climbs/search?q=kilter&per_page=3",
    ]

    for url in endpoints_to_probe:
        print(f"GET {url}")
        try:
            r = session.get(url, timeout=15)
            print(f"  → {r.status_code}  ({len(r.content)} bytes)")
            if r.status_code == 200:
                data = r.json()
                print(f"  → Keys: {list(data.keys()) if isinstance(data, dict) else 'list[' + str(len(data)) + ']'}")
                print(f"  ✓ This endpoint works — update KAYA_API_BASE in kaya_scraper.py")
                break
        except Exception as e:
            print(f"  → Error: {e}")

    print("\nNext steps:")
    print("  1. Set KAYA_TOKEN=<your_bearer_token> in .env")
    print("  2. Open DevTools → Network while browsing kaya-app.kayaclimb.com")
    print("  3. Find XHR calls and note the API base URL + endpoints")
    print("  4. Update KAYA_API_BASE at the top of this file")
    print("  5. Re-run --discover to validate")


# ── Main scrape loop ──────────────────────────────────────────────────────────
def run_scrape(
    limit: int | None = None,
    dry_run: bool = False,
    sample_fps: float = 5.0,
    delay_sec: float = 3.0,
    max_consecutive_failures: int = 10,
):
    token = get_auth_token()
    if not token:
        print("⚠  No KAYA_TOKEN found. Set it in .env:")
        print("   KAYA_TOKEN=eyJhbGci...")
        print("   (Get it from DevTools → Network → Authorization header)")
        print("\n   Run --discover to test your token once set.")
        sys.exit(1)

    session        = make_session(token)
    kilter_index   = load_kilter_climbs()
    checkpoint     = load_checkpoint()
    completed      = checkpoint["completed"]
    failed         = checkpoint["failed"]
    TMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nKaya beta scrape")
    print(f"  Kilter climbs in DB : {len(kilter_index):,}")
    print(f"  Already completed   : {len(completed):,}")
    print(f"  Previously failed   : {len(failed):,}")
    if dry_run:
        print("  [dry-run] No DB writes.\n")

    processed = skipped = errors = matched = 0
    total_frames = 0
    page = 1
    processed_urls = 0
    consecutive_failures = 0

    while True:
        routes = search_kilter_routes(session, page=page, per_page=50)
        if not routes:
            break

        for route in routes:
            if limit and processed_urls >= limit:
                break

            route_id   = str(route.get("id", route.get("uuid", "")))
            route_name = route.get("name", route.get("title", ""))
            if not route_id or not route_name:
                continue

            # Match to kilter.db climb UUID
            climb_uuid = match_climb_uuid(route_name, kilter_index)
            if not climb_uuid:
                continue   # not a known Kilter Board climb

            matched += 1
            betas = get_route_betas(session, route_id)
            if not betas:
                continue

            for beta in betas:
                if limit and processed_urls >= limit:
                    break

                video_url = extract_video_url(beta)
                if not video_url:
                    continue

                attempt_id = make_attempt_id(climb_uuid, video_url)
                if attempt_id in completed:
                    skipped += 1
                    continue

                processed_urls += 1
                print(f"[{processed_urls}] {route_name[:40]!r} | {video_url[:60]}")

                slug    = hashlib.sha256(video_url.encode()).hexdigest()[:12]
                tmp_mp4 = TMP_VIDEO_DIR / f"{climb_uuid[:8]}_{slug}.mp4"

                ok = download_kaya_video(session, video_url, tmp_mp4)
                if not ok:
                    print("    → skip (download failed)")
                    failed[video_url] = "download_failed"
                    save_checkpoint({"completed": completed, "failed": failed})
                    errors += 1
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\n✗ {consecutive_failures} consecutive download failures — aborting.")
                        print(f"  Token may be expired. Re-run --discover to check.")
                        return
                    continue

                consecutive_failures = 0
                size_mb = tmp_mp4.stat().st_size / 1_048_576
                print(f"    Downloaded  {size_mb:.1f} MB")

                try:
                    n_frames = process_video(
                        str(tmp_mp4),
                        attempt_id=attempt_id,
                        sample_fps=sample_fps,
                        min_confidence=0.4,
                        dry_run=dry_run,
                        verbose=False,
                        climb_uuid=climb_uuid,
                        filter_climbing=True,
                    )
                    total_frames += n_frames
                    print(f"    Pose frames : {n_frames}")
                except Exception as e:
                    print(f"    pose_extractor error: {e}")
                    failed[video_url] = str(e)
                    save_checkpoint({"completed": completed, "failed": failed})
                    errors += 1
                else:
                    if not dry_run:
                        completed[attempt_id] = climb_uuid
                        save_checkpoint({"completed": completed, "failed": failed})
                    processed += 1
                finally:
                    if tmp_mp4.exists():
                        tmp_mp4.unlink()

                jitter = random.uniform(0, 2.0)
                time.sleep(delay_sec + jitter)

        if limit and processed_urls >= limit:
            break
        page += 1
        time.sleep(1.0)

    print(f"\n── Run complete ──────────────────────────────────")
    print(f"  Matched routes : {matched}")
    print(f"  Processed      : {processed}")
    print(f"  Skipped        : {skipped}  (already done)")
    print(f"  Errors         : {errors}")
    print(f"  Pose rows      : {total_frames:,}")


# ── Stats ─────────────────────────────────────────────────────────────────────
def print_stats():
    checkpoint = load_checkpoint()
    completed  = checkpoint["completed"]
    failed     = checkpoint["failed"]

    print(f"\nKaya scrape progress")
    print(f"  Completed    : {len(completed):,}")
    print(f"  Failed       : {len(failed):,}")
    if CHECKPOINT_FILE.exists():
        print(f"  Checkpoint   : {CHECKPOINT_FILE}")

    try:
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
        # Kaya attempt_ids have 'kaya:' prefix baked in via make_attempt_id
        kaya_ids = list(completed.keys())
        if kaya_ids:
            cur.execute(
                "SELECT COUNT(*), COUNT(DISTINCT climb_uuid) FROM pose_frames "
                "WHERE attempt_id = ANY(%s)",
                (kaya_ids,),
            )
            frames, climbs = cur.fetchone()
            print(f"\npose_frames (kaya):")
            print(f"  Total rows   : {frames:,}")
            print(f"  Unique routes: {climbs:,}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"\n  (DB unavailable: {e})")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Kaya beta videos for Kilter Board routes, extract pose"
    )
    parser.add_argument("--discover",    action="store_true",
                        help="Probe Kaya API endpoints to validate token + discover structure")
    parser.add_argument("--limit",       type=int,   help="Max videos to process")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Download + extract but no DB writes")
    parser.add_argument("--stats",       action="store_true",
                        help="Show progress stats and exit")
    parser.add_argument("--sample-fps",  type=float, default=5.0)
    parser.add_argument("--delay",       type=float, default=3.0,
                        help="Base delay between downloads in seconds (default: 3)")
    parser.add_argument("--max-consecutive-failures", type=int, default=10,
                        help="Abort after this many consecutive download failures (default: 10)")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    token = get_auth_token()

    if args.discover:
        run_discover(token)
        return

    run_scrape(
        limit=args.limit,
        dry_run=args.dry_run,
        sample_fps=args.sample_fps,
        delay_sec=args.delay,
        max_consecutive_failures=args.max_consecutive_failures,
    )


if __name__ == "__main__":
    main()

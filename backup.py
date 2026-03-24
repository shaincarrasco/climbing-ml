"""
backup.py
---------
Archives critical model + data files to data/backups/<timestamp>.zip.

These files are NOT in git (data/ is gitignored) so manual backups are
the only recovery path if a retrain corrupts or overwrites a good model.

What gets backed up:
  ml/difficulty_model.xgb
  ml/pose_difficulty_model.xgb (if exists)
  ml/grade_boundaries.json
  ml/evaluation.json
  ml/feature_importance.csv
  data/pose_features.csv (if exists)
  data/pose_correlations.json (if exists)
  data/routes_features.csv (if exists)

Usage:
    python3 backup.py            # backup now
    python3 backup.py --list     # list existing backups
    python3 backup.py --restore  # restore from most recent backup
    python3 backup.py --restore data/backups/2026-03-23_120000.zip
"""

import argparse
import datetime
import zipfile
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
_BACKUP_DIR = _BASE / "data" / "backups"

_TARGETS = [
    "ml/difficulty_model.xgb",
    "ml/pose_difficulty_model.xgb",
    "ml/grade_boundaries.json",
    "ml/evaluation.json",
    "ml/feature_importance.csv",
    "data/pose_features.csv",
    "data/pose_correlations.json",
    "data/routes_features.csv",
]


def do_backup() -> Path:
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = _BACKUP_DIR / f"{ts}.zip"

    included = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in _TARGETS:
            p = _BASE / rel
            if p.exists():
                zf.write(p, rel)
                included.append(rel)
            else:
                print(f"  skip (not found): {rel}")

    print(f"\n  Backed up {len(included)} files -> {out.relative_to(_BASE)}")
    for f in included:
        print(f"    + {f}")
    return out


def do_list():
    if not _BACKUP_DIR.exists():
        print("  No backups found.")
        return
    zips = sorted(_BACKUP_DIR.glob("*.zip"))
    if not zips:
        print("  No backups found.")
        return
    print(f"  {len(zips)} backup(s) in {_BACKUP_DIR.relative_to(_BASE)}:")
    for z in zips:
        size = z.stat().st_size / 1024
        print(f"    {z.name}  ({size:.0f} KB)")


def do_restore(path: str = None):
    if path:
        src = Path(path)
        if not src.is_absolute():
            src = _BASE / path
    else:
        if not _BACKUP_DIR.exists():
            print("  No backups to restore from.")
            sys.exit(1)
        zips = sorted(_BACKUP_DIR.glob("*.zip"))
        if not zips:
            print("  No backups found.")
            sys.exit(1)
        src = zips[-1]
        print(f"  Restoring from most recent: {src.name}")

    if not src.exists():
        print(f"  Error: {src} not found")
        sys.exit(1)

    with zipfile.ZipFile(src) as zf:
        names = zf.namelist()
        for name in names:
            dest = _BASE / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
            print(f"  + Restored {name}")

    print(f"\n  {len(names)} files restored. Restart the API to load new models.")


def main():
    ap = argparse.ArgumentParser(description="Backup/restore Climbing ML model files")
    ap.add_argument("--list",    action="store_true", help="List existing backups")
    ap.add_argument("--restore", nargs="?", const="", metavar="PATH",
                    help="Restore from PATH (or most recent if omitted)")
    args = ap.parse_args()

    print("\n== Climbing ML -- Backup ============================================")
    if args.list:
        do_list()
    elif args.restore is not None:
        do_restore(args.restore if args.restore else None)
    else:
        do_backup()
    print()


if __name__ == "__main__":
    main()

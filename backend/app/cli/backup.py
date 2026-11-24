"""Consistent backup / restore of the BowlIQ database and analysis files.

  backup:  python scripts/backup.py backup  E:\\BowlIQBackups
  restore: python scripts/backup.py restore E:\\BowlIQBackups\\bowliq-20261121-0930   (stop the server first)

The database is copied with SQLite's online backup API (safe while the server runs); analysis folders are copied
incrementally. Database and files must be restored together because files are keyed by analysis id.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from app.core.config import settings


def backup(dest_root: Path, keep: int = 14) -> Path:
    dest = dest_root / f"bowliq-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=False)
    src = sqlite3.connect(settings.db_path)
    dst = sqlite3.connect(dest / "bowliq.db")
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    analyses = settings.storage_dir / "analyses"
    if analyses.exists():
        shutil.copytree(analyses, dest / "analyses")
    for name in (".api_token", ".signing_key"):
        p = settings.storage_dir / name
        if p.exists():
            shutil.copy2(p, dest / name)
    backups = sorted(p for p in dest_root.glob("bowliq-*") if p.is_dir())
    for old in backups[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def restore(src: Path) -> None:
    if not (src / "bowliq.db").exists():
        sys.exit(f"{src} is not a BowlIQ backup (bowliq.db missing)")
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(settings.db_path) + suffix).unlink(missing_ok=True)
    shutil.copy2(src / "bowliq.db", settings.db_path)
    target = settings.storage_dir / "analyses"
    if target.exists():
        shutil.rmtree(target)
    if (src / "analyses").exists():
        shutil.copytree(src / "analyses", target)
    for name in (".api_token", ".signing_key"):
        if (src / name).exists():
            shutil.copy2(src / name, settings.storage_dir / name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["backup", "restore"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--keep", type=int, default=14, help="number of backups to keep")
    a = ap.parse_args(argv)
    if a.action == "backup":
        print(f"Backup written to {backup(a.path, a.keep)}")
    else:
        restore(a.path)
        print(f"Restored from {a.path}. Start the server again.")
    return 0

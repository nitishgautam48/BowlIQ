"""Thin entry point: see app/cli/reencode.py.  Usage: python scripts/reencode_videos.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli.reencode import main

if __name__ == "__main__":
    raise SystemExit(main())

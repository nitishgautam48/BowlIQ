"""Thin entry point: see app/cli/run_folder.py.  Usage: python scripts/run_folder.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli.run_folder import main

if __name__ == "__main__":
    raise SystemExit(main())

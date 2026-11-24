"""Thin entry point: see app/cli/validate.py.  Usage: python scripts/validate.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli.validate import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Analyse every .mp4/.mov in a folder (no web server needed) and save the results to the BowlIQ database.

Writes <folder>/bowliq_runs.csv mapping each clip to its analysis id, for use with validate.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import uuid
from pathlib import Path

from app.core.config import api_tokens
from app.core.errors import BowlIQError
from app.core.jobs import history, save_analysis_and_finish
from app.core.storage import AnalysisFiles
from app.models.db import get_engine
from app.pipeline.orchestrator import run_analysis


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--bowler", default="Me")
    ap.add_argument("--height", type=float, default=None, help="bowler height in cm (enables metres and ball speed)")
    ap.add_argument("--owner", default=None, help="owner id to save under (default: the first configured owner)")
    args = ap.parse_args(argv)
    get_engine()
    owner = args.owner or next(iter(api_tokens().values()))
    clips = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in (".mp4", ".mov"))
    if not clips:
        sys.exit(f"No .mp4/.mov files in {args.folder}")
    rows = []
    for clip in clips:
        aid = uuid.uuid4().hex[:16]
        files = AnalysisFiles(aid).ensure()
        shutil.copy(clip, files.input("side", clip.suffix.lower()))
        params = {"bowler_name": args.bowler, "bowler_height_cm": args.height, "trim_start_s": None, "trim_end_s": None}
        print(f"-> {clip.name} ... ", end="", flush=True)
        try:
            res = run_analysis(aid, params, history(owner, args.bowler, aid), lambda s, p: None)
            save_analysis_and_finish(aid, owner, res)
            ev = res["events"]
            print(f"ok  BFC {ev['bfc']['frame']}  FFC {ev['ffc']['frame']}  release {ev['release']['frame']}  ({aid})")
            rows.append({"clip": clip.name, "analysis_id": aid, "status": "done", "error": ""})
        except BowlIQError as exc:
            files.delete()
            print(f"failed  {exc.code}: {exc.message}")
            rows.append({"clip": clip.name, "analysis_id": "", "status": "failed", "error": f"{exc.code}: {exc.message}"})
    out_csv = args.folder / "bowliq_runs.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "analysis_id", "status", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")

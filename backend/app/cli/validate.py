"""Compare BowlIQ against your manual (e.g. Kinovea) measurements.

Input CSV columns: clip, metric, manual_value, event, manual_frame
  - a row with `metric` + `manual_value` compares a metric value (e.g. front_knee_angle_ffc, 158)
  - a row with `event` (BFC | FFC | release) + `manual_frame` compares an event frame
`clip` is an analysis id, or a clip filename listed in a bowliq_runs.csv produced by run_folder.

Prints the mean absolute error (MAE) per metric and the frame error per event.

--sources: instead of stored analyses, `clip` is a video file (absolute, relative to the CSV, or in --clips-dir).
Each clip is analysed three ways from the same video and every metric is compared with your manual value:
  mediapipe_2d   MediaPipe pose, 2D angles (the old pipeline)
  rtmpose_2d     RTMPose (+ MediaPipe cross-check, identity correction, roles), 2D angles
  lifted_3d      the same skeleton lifted to 3D with MotionBERT
The table shows the MAE per metric and source; --write-config stores the most accurate of 2D/3D per metric as
angles.source in config/pipeline.yaml (the pose engine itself is chosen with BOWLIQ_POSE_BACKEND).
Skeletons are cached in tests/clips/.cache/, so re-runs only redo the analysis.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import pandas as pd

EVENT_KEYS = {"bfc": "bfc", "ffc": "ffc", "release": "release", "rel": "release"}


def db_loader(analysis_id: str) -> dict | None:
    from app.models.db import Analysis, session

    with session() as s:
        row = s.get(Analysis, analysis_id)
        return json.loads(row.result_json) if row else None


def validate(manual: pd.DataFrame, mapping: dict[str, str], load: Callable[[str], dict | None]):
    metric_rows, event_rows, problems = [], [], []
    for _, r in manual.iterrows():
        clip = str(r["clip"]).strip()
        aid = mapping.get(clip, clip)
        res = load(aid)
        if res is None:
            problems.append(f"{clip}: no analysis found (id {aid})")
            continue
        metric = str(r.get("metric") or "").strip()
        if metric and metric.lower() != "nan" and pd.notna(r.get("manual_value")):
            m = next((m for m in res["metrics"] if m["key"] == metric), None)
            if m is None:
                problems.append(f"{clip}: unknown metric {metric}")
            elif m["value"] is None:
                problems.append(f"{clip}: {metric} not measured ({m.get('reason')})")
            else:
                metric_rows.append({"clip": clip, "metric": metric, "manual": float(r["manual_value"]),
                                    "bowliq": m["value"], "abs_error": abs(m["value"] - float(r["manual_value"]))})
        event = str(r.get("event") or "").strip().lower()
        if event and event != "nan" and pd.notna(r.get("manual_frame")):
            key = EVENT_KEYS.get(event)
            if key is None:
                problems.append(f"{clip}: unknown event {event}")
            else:
                f = res["events"][key]["frame"]
                fps = res["video"]["fps"]
                err = f - int(r["manual_frame"])
                event_rows.append({"clip": clip, "event": key.upper() if key != "release" else "release",
                                   "manual": int(r["manual_frame"]), "bowliq": f, "frame_error": err,
                                   "abs_frame_error": abs(err), "abs_ms": abs(err) / fps * 1000})
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows), problems


ANGLE_KEYS = ("front_knee_angle_ffc", "front_knee_angle_release", "front_knee_flexion_ffc_to_release",
              "back_knee_angle_bfc", "trunk_forward_flexion_release", "trunk_lateral_flexion_release",
              "hip_shoulder_separation_bfc", "shoulder_counter_rotation")
SOURCES = ("mediapipe_2d", "rtmpose_2d", "lifted_3d")


def _analyse(video, engine: str, lifting: bool) -> dict:
    """Full analysis of one clip with a given pose engine, the skeleton cached per engine (like regression.py)."""
    import hashlib
    import os
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("BOWLIQ_STORAGE_DIR", str(root / "tests" / "clips" / ".cache" / "storage"))
    from app.core.storage import AnalysisFiles
    from app.pipeline import lift3d
    from app.pipeline import orchestrator as O
    from app.pipeline.fusion import MultiModelEstimator
    from app.pipeline.params import P
    from app.pipeline.pose import FallbackPoseEstimator, MediaPipePoseEstimator
    from app.pipeline.rtm import RtmPoseEstimator

    video = Path(video)
    key = "va" + hashlib.sha1(f"{video.name}{video.stat().st_size}{engine}".encode()).hexdigest()[:16]  # noqa: S324  # nosec B324 (cache key)
    files = AnalysisFiles(key)
    st = O.load_cache(files)
    if st is None:
        shutil.rmtree(files.dir, ignore_errors=True)
        files.ensure()
        shutil.copyfile(video, files.input("side", ".mp4"))
        est = MediaPipePoseEstimator() if engine == "mediapipe" else MultiModelEstimator(
            FallbackPoseEstimator(RtmPoseEstimator(), MediaPipePoseEstimator()), P.perception.secondary)
        st = O.prepare(files, {"bowler_name": "Validate"}, lambda s, p: None, est)
        O.save_cache(files, st)
        st = O.load_cache(files)
    real = lift3d.available
    lift3d.available = (lambda: real()) if lifting else (lambda: False)
    try:
        return O.analyse(files, st, {"bowler_name": "Validate"}, [], lambda s, p: None, render=False)
    finally:
        lift3d.available = real


def source_values(video) -> dict[str, dict[str, float | None]]:
    """{source: {metric: value}} for the three sources, plus events per engine under '_events'."""
    mp = _analyse(video, "mediapipe", lifting=False)
    rt = _analyse(video, "rtm", lifting=True)
    srcs = {"rtm": rt.get("context", {}).get("angle_sources", {})}
    out: dict[str, dict] = {s: {} for s in SOURCES}
    for m in mp["metrics"]:
        out["mediapipe_2d"][m["key"]] = m["value"]
    for m in rt["metrics"]:
        rep = srcs["rtm"].get(m["key"])
        if rep is not None:
            out["rtmpose_2d"][m["key"]] = rep["2d"]
            out["lifted_3d"][m["key"]] = rep["3d"]
        else:
            out["rtmpose_2d"][m["key"]] = m["value"]
    out["_events"] = {"mediapipe": {k: v["frame"] for k, v in mp["events"].items()},
                      "rtm": {k: v["frame"] for k, v in rt["events"].items()}, "fps": rt["video"]["fps"]}
    return out


def validate_sources(manual: pd.DataFrame, find_video: Callable[[str], object]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rows, ev_rows, problems, cache = [], [], [], {}
    for _, r in manual.iterrows():
        clip = str(r["clip"]).strip()
        video = find_video(clip)
        if video is None:
            problems.append(f"{clip}: video not found")
            continue
        if clip not in cache:
            try:
                cache[clip] = source_values(video)
            except Exception as exc:  # report and carry on with the other clips
                problems.append(f"{clip}: analysis failed ({type(exc).__name__}: {exc})")
                cache[clip] = None
        vals = cache[clip]
        if vals is None:
            continue
        metric = str(r.get("metric") or "").strip()
        if metric and metric.lower() != "nan" and pd.notna(r.get("manual_value")):
            for src in SOURCES:
                v = vals[src].get(metric)
                if v is not None:
                    rows.append({"clip": clip, "metric": metric, "source": src, "manual": float(r["manual_value"]),
                                 "value": float(v), "abs_error": abs(float(v) - float(r["manual_value"]))})
        event = str(r.get("event") or "").strip().lower()
        if event and event != "nan" and pd.notna(r.get("manual_frame")):
            key = EVENT_KEYS.get(event)
            for eng in ("mediapipe", "rtm"):
                f = vals["_events"][eng][key]
                ev_rows.append({"clip": clip, "event": key, "engine": eng, "abs_frame_error": abs(f - int(r["manual_frame"])),
                                "abs_ms": abs(f - int(r["manual_frame"])) / vals["_events"]["fps"] * 1000})
    return pd.DataFrame(rows), pd.DataFrame(ev_rows), problems


def best_sources(mdf: pd.DataFrame) -> dict[str, str]:
    """Per angle metric: '2d' or '3d', whichever of rtmpose_2d / lifted_3d has the lower MAE (needs both)."""
    out = {}
    if mdf.empty:
        return out
    mae = mdf.groupby(["metric", "source"])["abs_error"].mean().unstack()
    for metric in mae.index:
        if metric not in ANGLE_KEYS:
            continue
        a, b = mae.loc[metric].get("rtmpose_2d"), mae.loc[metric].get("lifted_3d")
        if pd.notna(a) and pd.notna(b):
            out[metric] = "2d" if a < b else "3d"
        elif pd.notna(b):
            out[metric] = "3d"
    return out


def write_config(best: dict[str, str]) -> None:
    import re
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"
    s = p.read_text(encoding="utf-8")
    for metric, src in best.items():
        s = re.sub(rf"(\n    {metric}: )\"?\w+\"?", rf"\g<1>{src}", s)
    p.write_text(s, encoding="utf-8")


def main_sources(a) -> int:
    from pathlib import Path
    manual = pd.read_csv(a.csv)
    base = Path(a.csv).resolve().parent
    clips_dir = Path(a.clips_dir) if a.clips_dir else None

    def find(clip: str):
        for c in (Path(clip), base / clip, *( [clips_dir / clip] if clips_dir else [])):
            if c.exists():
                return c
        return None

    mdf, edf, problems = validate_sources(manual, find)
    pd.set_option("display.width", 140)
    print("\n== Angle accuracy: mean absolute error (degrees / metric units) per source ==")
    if mdf.empty:
        print("no metric rows")
    else:
        t = mdf.groupby(["metric", "source"])["abs_error"].mean().unstack().reindex(columns=list(SOURCES))
        n = mdf.groupby("metric")["clip"].nunique().rename("clips")
        print(pd.concat([t.round(2), n], axis=1).to_string())
        best = best_sources(mdf)
        print("\nMost accurate 2D/3D source per metric (RTMPose skeleton):",
              ", ".join(f"{k} → {v}" for k, v in best.items()) or "need both sources for at least one clip")
        if a.write_config and best:
            write_config(best)
            print("Written to config/pipeline.yaml (angles.source). Re-run scripts/regression.py before keeping it.")
    print("\n== Event timing per pose engine (mean absolute error) ==")
    if edf.empty:
        print("no event rows")
    else:
        print(edf.groupby(["event", "engine"]).agg(n=("abs_frame_error", "count"), frames=("abs_frame_error", "mean"),
                                                   ms=("abs_ms", "mean")).round(2).to_string())
    for p in problems:
        print(" -", p)
    return 0


def main(argv: list[str] | None = None, load: Callable[[str], dict | None] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--runs", help="bowliq_runs.csv from run_folder.py (maps clip filename → analysis id)")
    ap.add_argument("--sources", action="store_true", help="compare MediaPipe-2D, RTMPose-2D and 3D-lifted per metric")
    ap.add_argument("--clips-dir", help="with --sources: folder holding the videos named in the CSV")
    ap.add_argument("--write-config", action="store_true", help="with --sources: store the best 2D/3D source per metric")
    a = ap.parse_args(argv)
    if a.sources:
        return main_sources(a)
    manual = pd.read_csv(a.csv)
    missing = {"clip", "metric", "manual_value", "event", "manual_frame"} - set(manual.columns)
    if missing:
        print(f"CSV is missing columns: {sorted(missing)}")
        return 2
    mapping = {}
    if a.runs:
        runs = pd.read_csv(a.runs, dtype=str, keep_default_na=False)
        mapping = {str(r["clip"]): str(r["analysis_id"]) for _, r in runs.iterrows() if r["analysis_id"]}
    mdf, edf, problems = validate(manual, mapping, load or db_loader)
    pd.set_option("display.width", 120)
    print("\n== Metric accuracy (MAE per metric) ==")
    print("no metric rows" if mdf.empty else
          mdf.groupby("metric")["abs_error"].agg(n="count", MAE="mean", max="max").round(2).to_string())
    print("\n== Event timing (frame error per event) ==")
    if edf.empty:
        print("no event rows")
    else:
        print(edf.groupby("event").agg(n=("abs_frame_error", "count"), mean_abs_frames=("abs_frame_error", "mean"),
                                       max_abs_frames=("abs_frame_error", "max"), mean_signed=("frame_error", "mean"),
                                       mean_abs_ms=("abs_ms", "mean")).round(2).to_string())
    if problems:
        print("\n== Skipped rows ==")
        for p in problems:
            print(" -", p)
    return 0

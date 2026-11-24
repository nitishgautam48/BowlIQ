"""Regression suite over every labelled clip in tests/clips/ (a video + a sidecar JSON with the ground truth).

    python scripts/regression.py                 # run all clips, print the table, compare with tests/clips/best.json
    python scripts/regression.py --fresh         # re-run the pose stage too (after changing pose/zoom/camera settings)
    python scripts/regression.py --only side_c1_30fps behind_c3_30fps
    python scripts/regression.py --jobs 2        # clips in parallel
    python scripts/regression.py --update-best   # accept the current results as the new best (only if none got worse)

A clip PASSes when every labelled item is right (key moments within the sidecar's tolerance_frames — foot contacts
never stricter than ±0.05 s, the heel-strike/foot-flat ambiguity — arm, direction, view). It counts as REVIEW (accepted) when something is off but the result is flagged needs_review with a reason that
names that item. Anything else is a FAIL.

Exit code 1 if any clip FAILs or gets worse than its stored best (status drops, or total timing error grows by more
than one frame). Rule: a change is accepted only if this passes on ALL clips.

The pose stage (slow) is cached per clip in tests/clips/.cache/ (keyed by the video's hash and the pose engine), so
re-running after a change to the event/context/metrics logic takes seconds per clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "tests" / "clips"
CACHE = CLIPS / ".cache"
BEST = CLIPS / "best.json"
CONTACT_TOL_S = 0.05  # one rule for every clip (in seconds): BFC/FFC within ±0.05 s or the sidecar's tolerance, if wider
RANK = {"PASS": 2, "REVIEW": 1, "FAIL": 0, "ERROR": 0}
KEYWORDS = {"bfc": ("BFC", "key moments", "back foot"), "ffc": ("FFC", "key moments", "front foot"),
            "release": ("release", "key moments"), "bowling_arm": ("arm",), "direction": ("direction",),
            "view": ("view", "camera angle", "front/behind")}


def sidecars(only: list[str] | None) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(CLIPS.glob("*.json")):
        if p.name == "best.json":
            continue
        if only and p.stem not in only:
            continue
        out.append((p.stem, json.loads(p.read_text())))
    return out


def _key(name: str, video: Path, truth: dict) -> str:
    h = hashlib.sha1(video.read_bytes()).hexdigest()[:12]  # noqa: S324  # nosec B324 (cache key, not security)
    extra = json.dumps({k: truth.get(k) for k in ("trim_start_s", "trim_end_s", "side_tap")}, sort_keys=True)
    engine = os.getenv("BOWLIQ_POSE_BACKEND") or os.getenv("POSE_BACKEND", "auto")
    return "rg" + hashlib.sha1(f"{name}{h}{extra}{engine}".encode()).hexdigest()[:16]  # noqa: S324  # nosec B324


def run_clip(name: str, truth: dict, fresh: bool, gemini: bool = False) -> dict:
    os.environ["BOWLIQ_STORAGE_DIR"] = str(CACHE / "storage")
    if gemini:
        os.environ["BOWLIQ_GEMINI_VIDEO"] = "1"
    sys.path.insert(0, str(ROOT))
    from app.core.storage import AnalysisFiles
    from app.pipeline import orchestrator as O

    video = CLIPS / truth["file"]
    key = _key(name, video, truth)
    files = AnalysisFiles(key)
    params = {"bowler_name": "Regression", "bowler_height_cm": truth.get("height_cm"),
              "slow_motion": truth.get("slow_motion"), "side_tap": truth.get("side_tap"),
              "trim_start_s": truth.get("trim_start_s"), "trim_end_s": truth.get("trim_end_s")}
    t0 = time.time()
    try:
        st = None if fresh else O.load_cache(files)
        if st is None:
            shutil.rmtree(files.dir, ignore_errors=True)
            files.ensure()
            shutil.copyfile(video, files.input("side", video.suffix.lower() if video.suffix.lower() in (".mp4", ".mov") else ".mp4"))
            from app.pipeline.pose import get_estimator
            st = O.prepare(files, params, lambda s, p: None, get_estimator())
            O.save_cache(files, st)
            st = O.load_cache(files)
        if gemini and getattr(st, "gemini", None) is None:  # Gemini brief once per clip, cached with the skeleton
            O.attach_gemini(files, st)
            O.save_cache(files, st)
            st = O.load_cache(files)
        if not gemini:
            st.gemini = None
        res = O.analyse(files, st, params, [], lambda s, p: None, render=False)
    except Exception as exc:  # the job itself failed
        return {"name": name, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "secs": round(time.time() - t0, 1)}
    return {"name": name, "result": {k: res[k] for k in ("events", "bowling_arm", "direction", "needs_review",
                                                          "review_reasons", "quality", "context", "video", "gemini")},
            "secs": round(time.time() - t0, 1)}


def judge(name: str, truth: dict, out: dict) -> dict:
    if out.get("status") == "ERROR":
        return {"name": name, "status": "ERROR", "detail": out["error"], "err_s": None, "errors": {}}
    r = out["result"]
    fps = r["video"]["fps"]
    tol = truth.get("tolerance_frames") or {}
    errs, wrong, got = {}, [], {}
    for k in ("bfc", "ffc", "release"):
        if truth.get(f"{k}_frame") is None:
            continue
        f = r["events"][k]["frame"]
        got[k] = f
        errs[k] = f - truth[f"{k}_frame"]
        # foot contacts: heel strike vs foot flat is ambiguous by ~50 ms, so never stricter than CONTACT_TOL_S
        limit = int(tol.get(k, 1))
        if k in ("bfc", "ffc"):
            limit = max(limit, int(round(CONTACT_TOL_S * fps)))
        if abs(errs[k]) > limit:
            wrong.append(k)
    view = r["context"]["view"]["value"] if r.get("context") else r["quality"]["view_detected"]
    view = {"diagonal": "oblique"}.get(view, view)
    got["view"], got["bowling_arm"], got["direction"] = view, r["bowling_arm"], r["direction"]
    ok_views = truth["view"] if isinstance(truth.get("view"), list) else [truth.get("view")]
    if truth.get("view") is not None and view not in ok_views:
        wrong.append("view")
    wrong.extend(k for k in ("bowling_arm", "direction") if truth.get(k) is not None and r[k] != truth[k])
    reasons = " ".join(r.get("review_reasons") or [])
    if not wrong:
        status = "PASS"
    elif r.get("needs_review") and all(any(w.lower() in reasons.lower() for w in KEYWORDS[k]) for k in wrong):
        status = "REVIEW"
    else:
        status = "FAIL"
    err_s = round(sum(abs(e) for e in errs.values()) / fps, 4)
    return {"name": name, "status": status, "wrong": wrong, "errors": errs, "got": got, "err_s": err_s, "fps": fps,
            "needs_review": r.get("needs_review"), "reasons": r.get("review_reasons"), "secs": out.get("secs")}


def worse(now: dict, best: dict | None) -> str | None:
    if best is None:
        return None
    if RANK[now["status"]] < RANK[best["status"]]:
        return f"status {best['status']} → {now['status']}"
    if now["err_s"] is not None and best.get("err_s") is not None:
        frame = 1.0 / (now.get("fps") or 30.0)
        if now["err_s"] > best["err_s"] + frame + 1e-9:
            return f"timing error {best['err_s'] * 1000:.0f} ms → {now['err_s'] * 1000:.0f} ms"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="re-run the pose stage (ignore the cache)")
    ap.add_argument("--only", nargs="*", help="clip names (sidecar file names without .json)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--update-best", action="store_true", help="store the current results as the best (if none got worse)")
    ap.add_argument("--json", help="also write the full results to this file")
    ap.add_argument("--gemini", action="store_true", help="use the Gemini video check (needs GEMINI_API_KEY; sends the "
                    "clips to Google). Results are compared with tests/clips/best_gemini.json")
    a = ap.parse_args()
    clips = sidecars(a.only)
    if not clips:
        print("No labelled clips found in tests/clips/ (add <name>.mp4 + <name>.json; see scripts/label_clip.py).")
        return 1
    global BEST
    if a.gemini:
        BEST = CLIPS / "best_gemini.json"
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            outs = list(ex.map(run_clip, [n for n, _ in clips], [t for _, t in clips], [a.fresh] * len(clips),
                               [a.gemini] * len(clips)))
    else:
        outs = [run_clip(n, t, a.fresh, a.gemini) for n, t in clips]
    best = json.loads(BEST.read_text()) if BEST.exists() else {}
    rows = [judge(n, t, o) for (n, t), o in zip(clips, outs, strict=True)]
    hdr = f"{'clip':28s} {'status':7s} {'BFC':>9s} {'FFC':>9s} {'release':>9s}  {'view':9s} {'arm':5s} {'dir':4s} {'err ms':>6s}  note"
    print(hdr)
    print("-" * len(hdr))
    regressions = []
    for (n, _t), row in zip(clips, rows, strict=True):
        if row["status"] == "ERROR":
            print(f"{n:28s} ERROR   {row['detail'][:90]}")
            regressions.append(n)
            continue

        def ev(k: str, row: dict = row) -> str:
            if k not in row["got"]:
                return "–"
            return f"{row['got'][k]}({row['errors'][k]:+d})" + ("!" if k in row["wrong"] else "")
        def mark(k: str, s: str, row: dict = row) -> str:
            return s + ("!" if k in row["wrong"] else "")
        w = worse(row, best.get(n))
        note = (f"WORSE: {w}" if w else "") + (f" review: {'; '.join(row['reasons'])[:80]}" if row["needs_review"] else "")
        d = {"left_to_right": "→", "right_to_left": "←"}[row["got"]["direction"]]
        print(f"{n:28s} {row['status']:7s} {ev('bfc'):>9s} {ev('ffc'):>9s} {ev('release'):>9s}  "
              f"{mark('view', row['got']['view']):9s} {mark('bowling_arm', row['got']['bowling_arm']):5s} "
              f"{mark('direction', d):4s} {row['err_s'] * 1000:6.0f}  {note}")
        if row["status"] == "FAIL" or w:
            regressions.append(n)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "REVIEW", "FAIL", "ERROR")}
    print("-" * len(hdr))
    print(f"{len(rows)} clips: {counts['PASS']} pass, {counts['REVIEW']} flagged for review (accepted), "
          f"{counts['FAIL']} fail, {counts['ERROR']} error.  '!' = outside tolerance / wrong.")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
    improved = {r["name"]: {k: r[k] for k in ("status", "err_s", "errors", "fps")} for r in rows if r["status"] != "ERROR"
                and (r["name"] not in best or (not worse(r, best[r["name"]]) and (RANK[r["status"]] > RANK[best[r["name"]]["status"]]
                     or r["err_s"] < best[r["name"]]["err_s"])))}
    if a.update_best and not regressions:
        best.update({r["name"]: {k: r[k] for k in ("status", "err_s", "errors", "fps")} for r in rows})
        BEST.write_text(json.dumps(best, indent=2, sort_keys=True))
        print(f"Stored as best: {BEST}")
    elif improved and not regressions:
        best.update(improved)
        BEST.write_text(json.dumps(best, indent=2, sort_keys=True))
        print(f"New best for: {', '.join(sorted(improved))}")
    if regressions:
        print(f"REGRESSION / FAIL: {', '.join(regressions)}")
        return 1
    print("OK: no clip failed or got worse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""HTTP API (v1). Every route except /health requires a signed-in owner and only sees that owner's data.
All errors are JSON {code, message, fix}."""
from __future__ import annotations

import contextlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlencode

import anyio
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlmodel import func, select
from starlette.concurrency import run_in_threadpool

from app.analysis.benchmarks import rule
from app.api.deps import load_result
from app.core import jobs
from app.core.config import settings
from app.core.errors import BowlIQError
from app.core.ffmpeg import ffmpeg_available
from app.core.security import limiter, media_user, require_user, sign_media
from app.core.storage import AnalysisFiles, ensure_free_space
from app.models.db import Analysis, Feedback, Job, db_ok, iso_utc, session
from app.models.schemas import (AnalysisResult, ApiError, BowlerHistory, BowlerSummary, CompareResponse, DeliverySummary,
                                FeedbackIn, HealthResponse, HistoryPoint, JobCreated, JobState, KeypointsResponse, Me,
                                MediaLinks, MetricDiff, MetricPoint, SampleClip, SyncIn, SyncInfo, BowlerSummaryOut, EventsIn,
                                ClipInfo, RecomputeIn)

public = APIRouter(prefix="/api/v1")
router = APIRouter(prefix="/api/v1")
ALLOWED_EXT = {".mp4", ".mov"}
CHUNK = 1 << 20
MEDIA_CACHE = "private, max-age=31536000, immutable"


def _read_limit(owner: str) -> None:
    limiter.check(f"read:{owner}", settings.rate_read_per_min)


# ───────────────────────────── auth helpers ─────────────────────────────

@router.get("/me", response_model=Me)
def me(owner: str = Depends(require_user)) -> Me:
    return Me(owner=owner)


# ───────────────────────────── upload / jobs ─────────────────────────────

async def _save_upload(up: UploadFile, dest_dir: Path, view: str) -> Path:
    name = up.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise BowlIQError("UNSUPPORTED_FORMAT", f"'{name[:80]}' is not an MP4 or MOV file.",
                          "Export the clip as MP4 (H.264) or MOV and upload it again.", 415)
    dest = dest_dir / f"{view}_input{ext}"
    size = 0
    async with await anyio.open_file(dest, "wb") as f:
        while chunk := await up.read(CHUNK):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                raise BowlIQError("FILE_TOO_LARGE", f"'{name[:80]}' is larger than 200 MB.",
                                  "Trim the clip to one delivery or export at a lower bitrate, then upload again.", 413)
            await f.write(chunk)
    if size == 0:
        raise BowlIQError("EMPTY_FILE", f"'{name[:80]}' is empty.", "Choose the video file again.", 400)
    return dest


def parse_float(v: Optional[str], field: str) -> Optional[float]:
    """Optional finite float from a form field (rejects NaN/Infinity)."""
    if v is None or str(v).strip() == "":
        return None
    try:
        x = float(v)
    except ValueError as exc:
        raise BowlIQError("INVALID_REQUEST", f"{field} must be a number.", f"Enter a number for {field}.", 400) from exc
    if not math.isfinite(x):
        raise BowlIQError("INVALID_REQUEST", f"{field} must be a finite number.", f"Enter a number for {field}.", 400)
    return x


def _nums(text: str, n: int, field: str) -> list[float]:
    parts = [p.strip() for p in text.split(",")]
    hint = f"Pick the {field.lower()} again on the upload page."
    if len(parts) != n:
        raise BowlIQError("INVALID_REQUEST", f"{field} must have {n} comma-separated numbers.", hint, 400)
    out = []
    for p in parts:
        try:
            x = float(p)
        except ValueError as exc:
            raise BowlIQError("INVALID_REQUEST", f"{field} must contain numbers only.", hint, 400) from exc
        if not math.isfinite(x):
            raise BowlIQError("INVALID_REQUEST", f"{field} must contain finite numbers.", hint, 400)
        out.append(x)
    return out


def parse_tap(v: Optional[str], field: str) -> Optional[list[float]]:
    """'t,x,y': t = seconds into the uploaded clip, x/y = position as a fraction (0–1) of the frame width/height."""
    if v is None or not v.strip():
        return None
    t, x, y = _nums(v, 3, field)
    if t < 0 or not (0 <= x <= 1 and 0 <= y <= 1):
        raise BowlIQError("INVALID_REQUEST", f"{field} is outside the video.", "Tap the bowler inside the video frame.", 400)
    return [t, x, y]


MAX_IGNORE_ZONES = 8


def parse_zones(v: Optional[str], field: str) -> list[list[float]]:
    """'x0,y0,x1,y1;…' rectangles (fractions 0–1 of the frame) whose people are ignored."""
    if v is None or not v.strip():
        return []
    rects = [r for r in v.split(";") if r.strip()]
    if len(rects) > MAX_IGNORE_ZONES:
        raise BowlIQError("INVALID_REQUEST", f"At most {MAX_IGNORE_ZONES} ignore zones are allowed.", "Remove some zones.", 400)
    out = []
    for r in rects:
        x0, y0, x1, y1 = _nums(r, 4, field)
        if not all(0 <= c <= 1 for c in (x0, y0, x1, y1)):
            raise BowlIQError("INVALID_REQUEST", f"{field} is outside the video.", "Draw the zone inside the video frame.", 400)
        if abs(x1 - x0) < 0.01 or abs(y1 - y0) < 0.01:
            raise BowlIQError("INVALID_REQUEST", f"{field} is too small.", "Drag a larger rectangle.", 400)
        out.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
    if len(out) == 1 and (out[0][2] - out[0][0]) * (out[0][3] - out[0][1]) > 0.95:
        raise BowlIQError("INVALID_REQUEST", f"{field} covers the whole video, so nobody could be analysed.",
                          "Draw the zone only around the people to ignore.", 400)
    return out


@router.post("/analyze", response_model=JobCreated)
async def analyze(request: Request, side_video: UploadFile = File(...), front_video: Optional[UploadFile] = File(None),
                  bowler_name: Optional[str] = Form(None), bowler_height_cm: Optional[str] = Form(None),
                  trim_start_s: Optional[str] = Form(None), trim_end_s: Optional[str] = Form(None),
                  side_tap: Optional[str] = Form(None), side_ignore: Optional[str] = Form(None),
                  front_tap: Optional[str] = Form(None), front_ignore: Optional[str] = Form(None),
                  slow_motion: Optional[str] = Form(None), camera_view: Optional[str] = Form(None),
                  bowling_arm: Optional[str] = Form(None),
                  owner: str = Depends(require_user)) -> JobCreated:
    from app.pipeline.quality import probe_video, resolve_trim

    limiter.check(f"analyze:{owner}", settings.rate_analyze_per_min)
    if await run_in_threadpool(jobs.active_jobs, owner) >= settings.max_queued_jobs:
        raise BowlIQError("TOO_MANY_JOBS", f"You already have {settings.max_queued_jobs} analyses waiting or running.",
                          "Wait for them to finish, then upload again.", 429)
    height = parse_float(bowler_height_cm, "Height")
    if height is not None and not (120 <= height <= 230):
        raise BowlIQError("INVALID_HEIGHT", f"Height {height:g} cm is outside 120–230 cm.", "Enter your height in centimetres, e.g. 178.", 400)
    ts, te = parse_float(trim_start_s, "Trim start"), parse_float(trim_end_s, "Trim end")
    if camera_view not in (None, "", "auto", "side-on", "front-on", "behind"):
        raise BowlIQError("INVALID_REQUEST", "Camera view must be auto, side-on, front-on or behind.", "Choose it again.", 400)
    if bowling_arm not in (None, "", "auto", "right", "left"):
        raise BowlIQError("INVALID_REQUEST", "Bowling arm must be auto, right or left.", "Choose it again.", 400)
    slow = parse_float(slow_motion, "Slow motion")
    if slow is not None and not (1 <= slow <= 60):
        raise BowlIQError("INVALID_REQUEST", "Slow motion must be between ×1 (real time) and ×60.",
                          "Choose the clip's slow-motion speed again.", 400)
    guide = {"side_tap": parse_tap(side_tap, "Bowler tap"), "side_ignore": parse_zones(side_ignore, "Ignore zone"),
             "front_tap": parse_tap(front_tap, "Front-clip bowler tap"),
             "front_ignore": parse_zones(front_ignore, "Front-clip ignore zone")}
    name = re.sub(r"[\x00-\x1f\x7f]", "", re.sub(r"\s+", " ", (bowler_name or "").strip()))[:60] or "Me"
    await run_in_threadpool(ensure_free_space, int(request.headers.get("content-length") or 0))
    job_id = uuid.uuid4().hex[:16]
    files = AnalysisFiles(job_id).ensure()
    try:
        side = await _save_upload(side_video, files.dir, "side")
        if front_video is not None and front_video.filename:
            await _save_upload(front_video, files.dir, "front")
        meta = await run_in_threadpool(probe_video, side)  # blocking ffprobe/OpenCV off the event loop
        start, end = resolve_trim(meta.duration_s, ts, te)
        if guide["side_tap"] is not None and not (start - 0.05 <= guide["side_tap"][0] <= end + 0.05):
            raise BowlIQError("INVALID_REQUEST", "The bowler tap is outside the trimmed part of the clip.",
                              "Tap the bowler again after setting the trim.", 400)
        params = {"bowler_name": name, "bowler_height_cm": height, "trim_start_s": ts, "trim_end_s": te,
                  **{k: v for k, v in guide.items() if v}, **({"slow_motion": slow} if slow is not None else {}),
                  **({"camera_view": camera_view} if camera_view in ("side-on", "front-on", "behind") else {}),
                  **({"bowling_arm": bowling_arm} if bowling_arm in ("right", "left") else {})}
        await run_in_threadpool(jobs.create_job, job_id, owner, params)
    except BaseException:
        files.delete()
        raise
    return JobCreated(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobState)
def job_state(job_id: str, owner: str = Depends(require_user)) -> JobState:
    _read_limit(owner)
    job = jobs.get_job(job_id, owner)
    if job is None:
        raise BowlIQError("NOT_FOUND", "That job doesn't exist.", "Start a new analysis.", 404)
    out = JobState(job_id=job.id, status=job.status, stage=job.stage, progress=job.progress)
    if job.status == "failed":
        out.error = ApiError(code=job.error_code or "FAILED", message=job.error_message or "The analysis failed.",
                             fix=job.error_fix or "Try again.")
        clip = _markable_clip(job)
        if clip is not None:
            out.can_mark, out.clip = True, clip
    elif job.status == "done":
        out.result = AnalysisResult.model_validate(load_result(job.id, owner))
    return out


def _markable_clip(job) -> Optional[ClipInfo]:
    """A failed job whose processed side-on clip is still there and can be finished by marking key moments."""
    if job.status != "failed" or job.error_code not in jobs.MARKABLE:
        return None
    f = AnalysisFiles(job.id)
    if not f.side_prep.exists():
        return None
    import cv2

    cap = cv2.VideoCapture(str(f.side_prep))
    info = ClipInfo(fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0), frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info if info.frame_count > 0 else None


def _job_media_user(request: Request, job_id: str, exp: Optional[int] = Query(default=None),
                    sig: Optional[str] = Query(default=None), u: Optional[str] = Query(default=None)) -> str:
    from app.core.security import verify_media
    auth = request.headers.get("authorization")
    if auth:
        return require_user(auth)
    if exp is not None and sig and u and verify_media(f"job{job_id}", u, int(exp), sig):
        return u
    raise BowlIQError("UNAUTHORIZED", "This media link is missing or has expired.", "Reload the page.", 401)


@router.get("/jobs/{job_id}/links", response_model=MediaLinks)
def job_links(job_id: str, owner: str = Depends(require_user)) -> MediaLinks:
    job = jobs.get_job(job_id, owner)
    if job is None or _markable_clip(job) is None:
        raise BowlIQError("NOT_FOUND", "There's no clip to mark for this job.", "Start a new analysis.", 404)
    sgn = sign_media(f"job{job_id}", owner)
    return MediaLinks(query=urlencode({"exp": sgn["exp"], "sig": sgn["sig"], "u": owner}), expires_at=sgn["exp"])


@router.get("/jobs/{job_id}/video")
def job_video(job_id: str, request: Request, owner: str = Depends(_job_media_user)) -> Response:
    """The processed side-on clip of a failed job, for marking the key moments by hand."""
    _read_limit(owner)
    job = jobs.get_job(job_id, owner)
    if job is None or _markable_clip(job) is None:
        raise BowlIQError("NOT_FOUND", "There's no clip to mark for this job.", "Start a new analysis.", 404)
    return _range_response(AnalysisFiles(job_id).side_prep, request, "video/mp4")


@router.post("/jobs/{job_id}/mark", response_model=JobCreated)
def mark_job(job_id: str, body: EventsIn, owner: str = Depends(require_user)) -> JobCreated:
    """Finish a failed analysis with key moments marked by hand (and the slow-motion factor, if it's slow motion)."""
    limiter.check(f"analyze:{owner}", settings.rate_analyze_per_min)
    job = jobs.get_job(job_id, owner)
    clip = _markable_clip(job) if job is not None else None
    if clip is None:
        raise BowlIQError("NOT_FOUND", "There's no clip to mark for this job (it may have expired after 24 hours).",
                          "Upload the clip again.", 404)
    _check_marks(body, clip.frame_count, clip.fps * (body.slow_motion or 1.0))
    old = json.loads(job.params_json)
    params = {"bowler_name": old.get("bowler_name") or "Me", "bowler_height_cm": old.get("bowler_height_cm"),
              "trim_start_s": None, "trim_end_s": None, "side_tap": None,
              "forced_events": {"bfc": body.bfc, "ffc": body.ffc, "release": body.release},
              "slow_motion": body.slow_motion or old.get("slow_motion"), "marked_from_job": job_id}
    src = AnalysisFiles(job_id)
    return JobCreated(job_id=_start_copy_job(src, params, owner))


def _check_marks(body: EventsIn, n: int, eff_fps: float) -> None:
    if not (body.bfc < body.ffc < body.release < n):
        raise BowlIQError("INVALID_EVENTS", "Key moments must be in order: back foot, then front foot, then release, "
                          "inside the clip.", "Step to each moment again and set it.", 400)
    if not (0.03 <= (body.ffc - body.bfc) / eff_fps <= 0.6 and 0.02 <= (body.release - body.ffc) / eff_fps <= 0.4):
        raise BowlIQError("INVALID_EVENTS", "Those key moments are too far apart (or too close) to be one delivery in "
                          "real time: back foot to front foot should take 0.03–0.6 s and front foot to release "
                          "0.02–0.4 s.", "Check the frames, and if the clip is slow motion choose its speed.", 400)


def _start_copy_job(src: AnalysisFiles, params: dict, owner: str) -> str:
    import shutil

    if jobs.active_jobs(owner) >= settings.max_queued_jobs:
        raise BowlIQError("TOO_MANY_JOBS", f"You already have {settings.max_queued_jobs} analyses waiting or running.",
                          "Wait for them to finish, then try again.", 429)
    ensure_free_space(src.side_prep.stat().st_size * 2)
    new_id = uuid.uuid4().hex[:16]
    dst = AnalysisFiles(new_id).ensure()
    try:
        shutil.copyfile(src.side_prep, dst.input("side", ".mp4"))
        if src.front_prep.exists():
            shutil.copyfile(src.front_prep, dst.input("front", ".mp4"))
        jobs.create_job(new_id, owner, {k: v for k, v in params.items() if v is not None})
    except BaseException:
        dst.delete()
        raise
    return new_id


# ───────────────────────────── analysis outputs ─────────────────────────────

@router.get("/analysis/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str, owner: str = Depends(require_user)) -> dict:
    _read_limit(owner)
    return load_result(analysis_id, owner)


@router.get("/analysis/{analysis_id}/links", response_model=MediaLinks)
def media_links(analysis_id: str, owner: str = Depends(require_user)) -> MediaLinks:
    """Short-lived signed query string for <video>/<img>/PDF URLs (they can't send an Authorization header)."""
    load_result(analysis_id, owner)
    s = sign_media(analysis_id, owner)
    return MediaLinks(query=urlencode({"exp": s["exp"], "sig": s["sig"], "u": owner}), expires_at=s["exp"])


def _range_response(path: Path, request: Request, media_type: str) -> Response:
    if not path.exists():
        raise BowlIQError("NOT_FOUND", "The video file is missing.", "Run the analysis again.", 404)
    size = path.stat().st_size
    rng = request.headers.get("range")
    headers = {"Accept-Ranges": "bytes", "Cache-Control": MEDIA_CACHE}
    if not rng:
        return FileResponse(path, media_type=media_type, headers=headers)
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    if m.group(1) == "":
        length = int(m.group(2))
        if length == 0:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        start, end = max(0, size - length), size - 1
    else:
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def body():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = f.read(min(CHUNK, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)})
    return StreamingResponse(body(), status_code=206, media_type=media_type, headers=headers)


@router.get("/analysis/{analysis_id}/video")
def get_video(analysis_id: str, request: Request, clean: bool = False, view: Literal["side", "front"] = "side",
              owner: str = Depends(media_user)) -> Response:
    _read_limit(owner)
    load_result(analysis_id, owner)
    f = AnalysisFiles(analysis_id)
    if view == "front":  # the front-on clip is only served overlay-free (the skeleton is drawn in the browser)
        return _range_response(f.front_clean, request, "video/mp4")
    return _range_response(f.clean if clean else f.annotated, request, "video/mp4")


@router.get("/analysis/{analysis_id}/keypoints", response_model=KeypointsResponse)
def get_keypoints(analysis_id: str, view: Literal["side", "front"] = "side",
                  owner: str = Depends(require_user)) -> Response:
    _read_limit(owner)
    load_result(analysis_id, owner)
    files = AnalysisFiles(analysis_id)
    p = files.front_keypoints if view == "front" else files.keypoints
    if not p.exists():
        raise BowlIQError("NOT_FOUND", "Keypoints are missing for this analysis.", "Run the analysis again.", 404)
    return Response(p.read_bytes(), media_type="application/json", headers={"Cache-Control": "private, max-age=3600"})


JOINT_RE = re.compile(r"^[a-z_]{3,20}$")


@router.get("/analysis/{analysis_id}/frame/{n}")
def get_frame(analysis_id: str, n: int, joint: Optional[str] = None, view: Literal["side", "front"] = "side",
              owner: str = Depends(media_user)) -> Response:
    from app.report.render_video import OverlayData, frame_jpeg

    _read_limit(owner)
    load_result(analysis_id, owner)
    if joint is not None and not JOINT_RE.match(joint):
        joint = None
    f = AnalysisFiles(analysis_id)
    cached = f.frame_cache(n, joint, view)
    if cached.exists():
        return Response(cached.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": MEDIA_CACHE})
    overlay, src = (f.front_overlay, f.front_prep) if view == "front" else (f.overlay, f.side_prep)
    if not overlay.exists() or not src.exists():
        raise BowlIQError("NOT_FOUND", "Frames are missing for this analysis.", "Run the analysis again.", 404)
    data = frame_jpeg(src, n, OverlayData.load(overlay), highlight=joint)
    if data is None:
        raise BowlIQError("FRAME_OUT_OF_RANGE", f"Frame {n} is outside this clip.", "Choose a frame inside the clip.", 404)
    cached.parent.mkdir(exist_ok=True)
    cached.write_bytes(data)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": MEDIA_CACHE})


@router.get("/analysis/{analysis_id}/pdf")
def get_pdf(analysis_id: str, owner: str = Depends(media_user)) -> Response:
    _read_limit(owner)
    r = load_result(analysis_id, owner)
    p = AnalysisFiles(analysis_id).pdf
    if not p.exists():
        raise BowlIQError("NOT_FOUND", "The PDF report is missing.", "Run the analysis again.", 404)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", r["bowler_name"])[:40]
    return FileResponse(p, media_type="application/pdf", filename=f"BowlIQ_{safe}_{analysis_id}.pdf",
                        headers={"Cache-Control": "private, no-store"})


@router.delete("/analysis/{analysis_id}")
def delete_analysis(analysis_id: str, owner: str = Depends(require_user)) -> dict:
    with session() as s:
        row = s.exec(select(Analysis).where(Analysis.id == analysis_id, Analysis.owner_id == owner)).first()
        if row is None:
            raise BowlIQError("NOT_FOUND", "That analysis doesn't exist.", "Refresh the page.", 404)
        job = s.get(Job, analysis_id)
        if job:
            s.delete(job)
        s.delete(row)  # feedback rows cascade (FK ON DELETE CASCADE)
        s.commit()
    AnalysisFiles(analysis_id).delete()
    import logging
    logging.getLogger("bowliq.audit").info("deleted analysis %s owner=%s", analysis_id, owner)
    return {"deleted": analysis_id}


@router.put("/analysis/{analysis_id}/sync", response_model=SyncInfo)
def set_sync(analysis_id: str, body: SyncIn, owner: str = Depends(require_user)) -> dict:
    """Save a hand-adjusted alignment of the two clips (offset_s: front_time = side_time + offset), or null to go back
    to the automatic one."""
    limiter.check(f"sync:{owner}", 60)
    with session() as s:
        row = s.exec(select(Analysis).where(Analysis.id == analysis_id, Analysis.owner_id == owner)).first()
        if row is None:
            raise BowlIQError("NOT_FOUND", "That analysis doesn't exist (it may have been deleted).",
                              "Open My deliveries and choose another delivery.", 404)
        result = json.loads(row.result_json)
        if not result.get("sync"):
            raise BowlIQError("NO_FRONT_CLIP", "This delivery has no second clip to line up.",
                              "Upload a front-on clip together with the side-on clip.", 400)
        result["sync"]["manual_offset_s"] = None if body.offset_s is None else round(body.offset_s, 4)
        row.result_json = json.dumps(result)
        s.add(row)
        s.commit()
    return result["sync"]


# ───────────────────────────── manual assist: corrections in seconds ─────────────────────────────

CTX_FIELDS = {"view": "view", "bowling_arm": "bowling_arm", "direction": "direction", "bowler_track_id": "bowler"}


def _auto_value(old: dict, field: str):
    """What the pipeline detected on its own for `field` (before any correction)."""
    ctx = old.get("context") or {}
    if field in CTX_FIELDS:
        d = ctx.get(CTX_FIELDS[field]) or {}
        return (d.get("details") or {}).get("auto_value", d.get("value")) if d.get("source") == "user" else d.get("value")
    if field.endswith("_frame"):
        ev = (old.get("events") or {}).get(field[:-6]) or {}
        methods = ev.get("methods") or {}
        return None if "user" in methods else ev.get("frame")
    if field == "height_cm":
        return old.get("bowler_height_cm")
    return None


def log_corrections(analysis_id: str, old: dict, new: dict) -> None:
    """Append each correction to storage/corrections.jsonl: clip id, field, auto value, corrected value. No personal
    data: these records are for checking where the automatic detection goes wrong."""
    from datetime import datetime, timezone

    path = settings.storage_dir / "corrections.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as fh:
        for field, value in new.items():
            fh.write(json.dumps({"ts": ts, "clip_id": analysis_id, "field": field, "auto": _auto_value(old, field),
                                 "corrected": value, "fps": (old.get("video") or {}).get("fps")}) + "\n")


@router.post("/analysis/{analysis_id}/recompute", response_model=AnalysisResult)
def recompute_analysis(analysis_id: str, body: RecomputeIn, owner: str = Depends(require_user)) -> dict:
    """Apply corrections (view, bowling arm, direction, which person is the bowler, key-moment frames, height) and
    recompute metrics, analysis, report, video and PDF from the cached keypoints — no pose re-run, a few seconds.
    The same delivery is updated in place; corrections accumulate until `clear` is sent."""
    from datetime import datetime, timezone

    from app.pipeline.orchestrator import recompute

    limiter.check(f"recompute:{owner}", 30)
    old = load_result(analysis_id, owner)
    n = old["video"]["frame_count"]
    for k in ("bfc_frame", "ffc_frame", "release_frame"):
        v = getattr(body, k)
        if v is not None and v >= n:
            raise BowlIQError("INVALID_EVENTS", f"Frame {v} is outside this clip (it has {n} frames).",
                              "Step to the moment on the video and set it again.", 400)
    job = jobs.get_job(analysis_id, owner)
    params = json.loads(job.params_json) if job is not None else {}
    params.setdefault("bowler_name", old["bowler_name"])
    params.setdefault("bowler_height_cm", old.get("bowler_height_cm"))
    new = {k: v for k, v in body.model_dump(exclude={"clear"}).items() if v is not None}
    if new.get("view") == "diagonal":
        new["view"] = "oblique"
    prev = {} if body.clear else dict(old.get("overrides") or {})
    if "bowler_track_id" in new and new.get("bowler_track_id") != prev.get("bowler_track_id"):
        for k in ("bfc_frame", "ffc_frame", "release_frame"):  # another person: their own key moments
            if k not in new:
                prev.pop(k, None)
    params.update({"overrides": {**prev, **new}, "forced_events": None, "camera_view": None, "bowling_arm": None,
                   "_created_at": old["created_at"],
                   "_recomputed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    hist = jobs.history(owner, old["bowler_name"], analysis_id)
    result = recompute(analysis_id, params, hist)
    metrics = {m["key"]: {"value": m["value"], "unit": m["unit"]} for m in result["metrics"] if m["value"] is not None}
    with session() as s:
        row = s.exec(select(Analysis).where(Analysis.id == analysis_id, Analysis.owner_id == owner)).first()
        if row is None:
            raise BowlIQError("NOT_FOUND", "That analysis doesn't exist (it may have been deleted).",
                              "Open My deliveries.", 404)
        row.result_json = json.dumps(result, allow_nan=False)
        row.metrics_json = json.dumps(metrics, allow_nan=False)
        row.ball_speed_kmh = result["ball_speed"]["value_kmh"]
        row.quality = result["quality"]["level"]
        row.top_finding = result["findings"][0]["title"] if result["findings"] else None
        s.add(row)
        s.commit()
    if new:
        with contextlib.suppress(OSError):  # the log is a help, never a reason to fail the correction
            log_corrections(analysis_id, old, new)
    return load_result(analysis_id, owner)


DEBUG_NAMES = {"signals.png": "image/png", "context.json": "application/json", "bfc.jpg": "image/jpeg",
               "ffc.jpg": "image/jpeg", "release.jpg": "image/jpeg", "roles.mp4": "video/mp4",
               "pose3d.json": "application/json"}


@router.get("/analysis/{analysis_id}/debug/{name}")
def get_debug(analysis_id: str, name: str, owner: str = Depends(media_user)) -> Response:
    """Diagnostics written for this analysis (storage/debug/{id}/): signals.png, context.json and the key frames."""
    _read_limit(owner)
    load_result(analysis_id, owner)
    if name not in DEBUG_NAMES:
        raise BowlIQError("NOT_FOUND", "Unknown diagnostics file.", "Choose signals.png, context.json, pose3d.json, roles.mp4 or a key frame.", 404)
    p = AnalysisFiles(analysis_id).debug_dir / name
    if not p.exists():
        raise BowlIQError("NOT_FOUND", "Diagnostics are missing for this analysis.", "Recompute it to write them.", 404)
    if name.endswith(".mp4"):  # FileResponse answers Range requests, so the video can be scrubbed
        return FileResponse(p, media_type=DEBUG_NAMES[name], headers={"Cache-Control": "private, no-store"})
    return Response(p.read_bytes(), media_type=DEBUG_NAMES[name], headers={"Cache-Control": "private, no-store"})


@router.get("/analysis/{analysis_id}/person/{cid}")
def get_person(analysis_id: str, cid: int, owner: str = Depends(media_user)) -> Response:
    """Thumbnail of person `cid` (from context.candidates) for 'Choose bowler'."""
    import cv2

    _read_limit(owner)
    r = load_result(analysis_id, owner)
    cands = ((r.get("context") or {}).get("candidates")) or []
    c = next((c for c in cands if c.get("id") == cid), None)
    f = AnalysisFiles(analysis_id)
    if c is None or not f.side_prep.exists():
        raise BowlIQError("NOT_FOUND", "That person isn't in this clip's list.", "Recompute the analysis.", 404)
    cap = cv2.VideoCapture(str(f.side_prep))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(c["frame"]))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise BowlIQError("FRAME_OUT_OF_RANGE", "That frame couldn't be read.", "Recompute the analysis.", 404)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = c["box"]
    x0, x1 = int(max(0, x0 * w)), int(min(w, x1 * w))
    y0, y1 = int(max(0, y0 * h)), int(min(h, y1 * h))
    crop = img[y0:max(y0 + 2, y1), x0:max(x0 + 2, x1)]
    scale = 220 / max(1, crop.shape[0])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


@router.post("/analysis/{analysis_id}/reanalyze", response_model=JobCreated)
def reanalyze(analysis_id: str, body: EventsIn, owner: str = Depends(require_user)) -> JobCreated:
    """Re-run the analysis with key moments set by hand (frames of this analysis's side-on video). The processed
    video is reused, the same bowler is kept (tapped at the release frame), and a new delivery is created."""
    import shutil

    limiter.check(f"analyze:{owner}", settings.rate_analyze_per_min)
    old = load_result(analysis_id, owner)
    n = old["video"]["frame_count"]
    fps = old["video"]["fps"]
    _check_marks(body, n, fps * (body.slow_motion or 1.0))
    src = AnalysisFiles(analysis_id)
    if not src.side_prep.exists():
        raise BowlIQError("NOT_FOUND", "The processed video for this delivery is missing.", "Upload the clip again.", 404)
    if jobs.active_jobs(owner) >= settings.max_queued_jobs:
        raise BowlIQError("TOO_MANY_JOBS", f"You already have {settings.max_queued_jobs} analyses waiting or running.",
                          "Wait for them to finish, then try again.", 429)
    ensure_free_space(src.side_prep.stat().st_size * 2)
    job_id = uuid.uuid4().hex[:16]
    dst = AnalysisFiles(job_id).ensure()
    try:
        shutil.copyfile(src.side_prep, dst.input("side", ".mp4"))
        if src.front_prep.exists():
            shutil.copyfile(src.front_prep, dst.input("front", ".mp4"))
        params = {"bowler_name": old["bowler_name"], "bowler_height_cm": old.get("bowler_height_cm"),
                  "trim_start_s": None, "trim_end_s": None,
                  "forced_events": {"bfc": body.bfc, "ffc": body.ffc, "release": body.release},
                  "reanalysis_of": analysis_id}
        if body.slow_motion and body.slow_motion > 1:
            params["slow_motion"] = body.slow_motion
        tap = _bowler_tap(src, body.release, n, fps)
        if tap:
            params["side_tap"] = tap
        jobs.create_job(job_id, owner, params)
    except BaseException:
        dst.delete()
        raise
    return JobCreated(job_id=job_id)


def _bowler_tap(files: AnalysisFiles, frame: int, n: int, fps: float) -> Optional[list[float]]:
    """Keep the same person: a tap on the analysed bowler's hips at `frame` (or the nearest frame he's visible)."""
    try:
        kp = json.loads(files.keypoints.read_text())
    except (OSError, ValueError):
        return None
    joints = {name: i for i, name in enumerate(kp["joints"])}
    for d in sorted(range(-15, 16), key=abs):
        f = frame + d
        if not 0 <= f < len(kp["frames"]):
            continue
        pts = [kp["frames"][f][joints[j]] for j in ("left_hip", "right_hip", "left_shoulder", "right_shoulder")]
        pts = [p for p in pts if p[2] > 0]
        if len(pts) >= 2:
            return [round(f / fps, 4), round(sum(p[0] for p in pts) / len(pts), 4), round(sum(p[1] for p in pts) / len(pts), 4)]
    return None


@router.post("/analysis/{analysis_id}/feedback")
def feedback(analysis_id: str, body: FeedbackIn, owner: str = Depends(require_user)) -> dict:
    limiter.check(f"feedback:{owner}", 60)
    r = load_result(analysis_id, owner)
    if body.finding_rank > len(r["findings"]):
        raise BowlIQError("INVALID_REQUEST", f"This report has no finding {body.finding_rank}.", "Refresh the page.", 400)
    with session() as s:  # one vote per finding: update in place
        fb = s.exec(select(Feedback).where(Feedback.analysis_id == analysis_id,
                                           Feedback.finding_rank == body.finding_rank)).first()
        if fb is None:
            fb = Feedback(analysis_id=analysis_id, finding_rank=body.finding_rank, helpful=body.helpful)
        else:
            fb.helpful = body.helpful
        s.add(fb)
        s.commit()
    return {"ok": True}


# ───────────────────────────── bowlers / history / compare ─────────────────────────────

@router.get("/deliveries", response_model=list[DeliverySummary])
def deliveries(bowler: Optional[str] = None, limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
               owner: str = Depends(require_user)) -> list[DeliverySummary]:
    _read_limit(owner)
    with session() as s:
        q = select(Analysis.id, Analysis.created_at, Analysis.bowler_name, Analysis.ball_speed_kmh, Analysis.quality,
                   Analysis.top_finding, Analysis.has_front).where(Analysis.owner_id == owner)
        if bowler:
            q = q.where(Analysis.bowler_name == bowler)
        rows = s.exec(q.order_by(Analysis.created_at.desc()).offset(offset).limit(limit)).all()
    return [DeliverySummary(id=i, created_at=iso_utc(c), bowler_name=b, ball_speed_kmh=sp, quality=qu, top_finding=t,
                            has_front=hf) for i, c, b, sp, qu, t, hf in rows]


@router.get("/bowlers", response_model=list[BowlerSummary])
def bowlers(owner: str = Depends(require_user)) -> list[BowlerSummary]:
    _read_limit(owner)
    with session() as s:
        rows = s.exec(select(Analysis.bowler_name, func.count(Analysis.id), func.max(Analysis.created_at))
                      .where(Analysis.owner_id == owner).group_by(Analysis.bowler_name)
                      .order_by(func.max(Analysis.created_at).desc())).all()
    return [BowlerSummary(name=n, deliveries=c, last_delivery_at=iso_utc(m)) for n, c, m in rows]


@router.get("/bowlers/{name}/history", response_model=BowlerHistory)
def history(name: str, limit: int = Query(200, ge=1, le=1000), owner: str = Depends(require_user)) -> BowlerHistory:
    _read_limit(owner)
    with session() as s:
        total = s.exec(select(func.count()).select_from(Analysis)
                       .where(Analysis.owner_id == owner, Analysis.bowler_name == name)).one()
        rows = s.exec(select(Analysis.id, Analysis.created_at, Analysis.ball_speed_kmh, Analysis.quality,
                             Analysis.top_finding, Analysis.metrics_json)
                      .where(Analysis.owner_id == owner, Analysis.bowler_name == name)
                      .order_by(Analysis.created_at.desc()).limit(limit)).all()
    pts = [HistoryPoint(id=i, created_at=iso_utc(c), ball_speed_kmh=sp, quality=q, top_finding=t,
                        metrics={k: MetricPoint(**v) for k, v in json.loads(mj).items()})
           for i, c, sp, q, t, mj in reversed(rows)]
    need = settings.baseline_min_deliveries
    sample = name == settings.sample_bowler
    return BowlerHistory(name=name, deliveries=pts, baseline_active=(total >= need and not sample),
                         deliveries_until_baseline=0 if sample else max(0, need - total))


@router.get("/bowlers/{name}/summary", response_model=BowlerSummaryOut)
def bowler_summary(name: str, last: int = Query(10, ge=1, le=30), owner: str = Depends(require_user)) -> dict:
    """Phase 5: consistent vs one-off issues across the bowler's last `last` deliveries."""
    from app.analysis.consistency import summarise
    from app.report.generator import coaching_notes

    _read_limit(owner)
    with session() as s:
        rows = s.exec(select(Analysis.result_json).where(Analysis.owner_id == owner, Analysis.bowler_name == name)
                      .order_by(Analysis.created_at.desc()).limit(last)).all()
    results = [json.loads(r) for r in rows]
    return {"name": name, **summarise(results, coaching_notes())}


def verdict(key: str, a: Optional[float], b: Optional[float], ua: str, ub: str) -> tuple[str, str, Optional[float]]:
    if a is None or b is None:
        return "not_comparable", "Not measured in both deliveries", None
    if ua != ub:
        return "not_comparable", "Measured in different units (height given for only one)", None
    delta = round(a - b, 2)
    r = rule(key) if key != "ball_speed" else {"type": "higher_better"}
    if abs(delta) < 1e-9:
        return "same", "No change", delta
    typ = r["type"]
    if typ == "higher_better":
        better = a > b
    elif typ == "max":
        better = a < b
    elif typ == "band":
        lo, hi = r["band"]

        def dist(v: float) -> float:
            return 0.0 if lo <= v <= hi else (lo - v if v < lo else v - hi)

        da, db = dist(a), dist(b)
        if da == db:
            return "same", "Both in range" if da == 0 else "Equally far from the target", delta
        better = da < db
    else:
        return "neutral", "Monitored, no better or worse", delta
    return ("better", "Better", delta) if better else ("worse", "Worse", delta)


@router.get("/compare", response_model=CompareResponse)
def compare(a: str, b: str, owner: str = Depends(require_user)) -> dict:
    _read_limit(owner)
    ra, rb = load_result(a, owner), load_result(b, owner)
    diffs: list[MetricDiff] = []
    va, vb = ra["ball_speed"]["value_kmh"], rb["ball_speed"]["value_kmh"]
    v, note, d = verdict("ball_speed", va, vb, "km/h", "km/h")
    diffs.append(MetricDiff(key="ball_speed", name="Ball speed", unit="km/h", a=va, b=vb, delta=d, verdict=v, note=note))
    mb = {m["key"]: m for m in rb["metrics"]}
    for m in ra["metrics"]:
        o = mb.get(m["key"])
        if o is None:
            continue
        v, note, d = verdict(m["key"], m["value"], o["value"], m["unit"], o["unit"])
        diffs.append(MetricDiff(key=m["key"], name=m["name"], unit=m["unit"], a=m["value"], b=o["value"], delta=d,
                                verdict=v, note=note))
    return {"a": ra, "b": rb, "diffs": diffs}


# ───────────────────────────── reference data / samples ─────────────────────────────

@router.get("/benchmarks")
def get_benchmarks(owner: str = Depends(require_user)) -> dict:
    from app.analysis.benchmarks import benchmarks
    from app.pipeline.metrics import METHODS

    return {k: {**v, "method": METHODS[k]["method"], "event": METHODS[k]["event"]} for k, v in benchmarks().items()}


@router.get("/samples", response_model=list[SampleClip])
def samples(owner: str = Depends(require_user)) -> list[SampleClip]:
    d = settings.samples_dir
    return [SampleClip(name=p.name, size_bytes=p.stat().st_size) for p in sorted(d.glob("*")) if p.suffix.lower() in ALLOWED_EXT]


@router.get("/samples/{name}")
def sample_file(name: str, owner: str = Depends(require_user)) -> Response:
    p = (settings.samples_dir / name).resolve()
    if p.parent != settings.samples_dir.resolve() or not p.exists() or p.suffix.lower() not in ALLOWED_EXT:
        raise BowlIQError("NOT_FOUND", "That sample clip doesn't exist.",
                          "Run `python scripts/make_sample.py` in backend/ to generate the sample clip.", 404)
    return FileResponse(p, media_type="video/mp4" if p.suffix.lower() == ".mp4" else "video/quicktime", filename=p.name)


# ───────────────────────────── health / readiness / metrics ─────────────────────────────

_state: dict[str, str] = {"pose_device": "unknown", "pose_engine": "unknown", "video_encoder": "unknown"}


def set_runtime_info(engine: str, device: str, encoder: str) -> None:
    _state.update(pose_engine=engine, pose_device=device, video_encoder=encoder)


@public.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + dependency status. No side effects and no filesystem paths."""
    from app.pipeline.pose import model_file_ok

    ff, model, dbok = ffmpeg_available(), model_file_ok(), db_ok()
    dev = _state["pose_device"]
    details = {
        "ffmpeg": "found" if ff else "missing: install with `winget install Gyan.FFmpeg` and reopen the terminal",
        "model_file": "present and verified" if model else "missing: it downloads when the server starts",
        "database": "ok" if dbok else "unavailable",
        "pose_device": {"gpu": f"GPU ({_state['pose_engine']})", "cpu": f"CPU ({_state['pose_engine']})",
                        "unknown": "not probed yet"}[dev],
        "video_encoder": {"h264_nvenc": "NVIDIA NVENC (GPU)", "libx264": "libx264 (CPU)"}.get(_state["video_encoder"], "not probed yet"),
        "gemini": "key set" if settings.gemini_api_key else "no key: reports use the built-in template",
        "gemini_video": ("on: each clip is checked by Gemini first (sent to Google)" if settings.gemini_video and
                         settings.gemini_api_key else "off" if not settings.gemini_video else
                         "switched on, but there is no GEMINI_API_KEY"),
    }
    return HealthResponse(ok=ff and model and dbok, ffmpeg=ff, model_file=model, database=dbok,
                          pose_device=dev, pose_engine=_state["pose_engine"], video_encoder=_state["video_encoder"],
                          gemini_key=bool(settings.gemini_api_key),
                          gemini_video=bool(settings.gemini_video and settings.gemini_api_key), details=details)


@router.get("/metrics")
def metrics(owner: str = Depends(require_user)) -> Response:
    """Prometheus text format."""
    from app.core.observability import render_metrics

    return Response(render_metrics(jobs.queue_depth()), media_type="text/plain; version=0.0.4")

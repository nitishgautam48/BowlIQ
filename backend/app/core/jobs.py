"""Job queue persisted in SQLite with atomic, lease-based claiming.

Any number of worker processes (or the embedded worker thread inside the API process) can run safely: a worker
claims a queued job with one atomic UPDATE, renews its lease while it works, and only jobs whose lease has expired
(their worker died) are treated as interrupted.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import traceback
import uuid
from datetime import timedelta
from typing import Optional

from sqlmodel import func, select

from app.core.config import settings
from app.core.errors import BowlIQError
from app.core.observability import add_stage_time, count_job
from app.core.storage import AnalysisFiles
from app.models.db import Analysis, Job, get_engine, session, utcnow

log = logging.getLogger("bowliq.jobs")
LEASE_S = 120
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
_last_progress: dict[str, tuple[str, int, float]] = {}
_stage_start: dict[str, float] = {}
_lock = threading.Lock()
_stop = threading.Event()
_wake = threading.Event()


def create_job(job_id: str, owner: str, params: dict) -> Job:
    with session() as s:
        job = Job(id=job_id, owner_id=owner, params_json=json.dumps(params, allow_nan=False))
        s.add(job)
        s.commit()
        s.refresh(job)
    _wake.set()
    return job


def get_job(job_id: str, owner: Optional[str] = None) -> Optional[Job]:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None or (owner is not None and job.owner_id != owner):
            return None
        return job


def active_jobs(owner: str) -> int:
    with session() as s:
        return s.exec(select(func.count()).select_from(Job)
                      .where(Job.owner_id == owner, Job.status.in_(["queued", "processing"]))).one()


def update_job(job_id: str, **fields) -> None:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        job.updated_at = utcnow()
        s.add(job)
        s.commit()


def progress(job_id: str, stage: str, pct: float) -> None:
    """Persist stage/progress (throttled to integer changes) and renew the lease at least every 10 s."""
    p = int(max(0, min(99, pct)))
    now = time.monotonic()
    with _lock:
        prev = _last_progress.get(job_id)
        if prev and (prev[0], prev[1]) == (stage, p) and now - prev[2] < 10:
            return
        if prev and prev[0] == stage and prev[1] > p:
            p = prev[1]
        if prev and prev[0] != stage:
            add_stage_time(prev[0], now - _stage_start.get(job_id, now))
            _stage_start[job_id] = now
        elif not prev:
            _stage_start[job_id] = now
        _last_progress[job_id] = (stage, p, now)
    update_job(job_id, stage=stage, progress=p, status="processing", lease_until=utcnow() + timedelta(seconds=LEASE_S))


def recover_interrupted() -> int:
    """Jobs still 'processing' whose lease has expired (or that have none) were killed with their worker:
    mark them failed with code INTERRUPTED. Queued jobs are kept and will be picked up."""
    now = utcnow()
    with session() as s:
        jobs = s.exec(select(Job).where(Job.status == "processing")).all()
        n = 0
        for j in jobs:
            lease = j.lease_until
            if lease is not None and lease.tzinfo is None:
                lease = lease.replace(tzinfo=now.tzinfo)
            if lease is None or lease < now:
                j.status = "failed"
                j.error_code = "INTERRUPTED"
                j.error_message = "The analysis stopped because the server or worker was restarted."
                j.error_fix = "Upload the clip again."
                j.updated_at = now
                s.add(j)
                n += 1
        s.commit()
        return n


def claim_next(worker_id: str = WORKER_ID) -> Optional[str]:
    """Atomically claim the oldest queued job."""
    with get_engine().begin() as c:
        row = c.exec_driver_sql(
            "UPDATE job SET status='processing', worker=?, lease_until=?, updated_at=? "
            "WHERE id = (SELECT id FROM job WHERE status='queued' ORDER BY created_at LIMIT 1) "
            "AND status='queued' RETURNING id",
            (worker_id, utcnow() + timedelta(seconds=LEASE_S), utcnow())).first()
        return row[0] if row else None


def history(owner: str, bowler: str, exclude: str) -> list[dict]:
    """Previous deliveries for the baseline. The synthetic sample bowler never gets a baseline."""
    if bowler == settings.sample_bowler:
        return []
    with session() as s:
        rows = s.exec(select(Analysis.id, Analysis.ball_speed_kmh, Analysis.metrics_json)
                      .where(Analysis.owner_id == owner, Analysis.bowler_name == bowler, Analysis.id != exclude)
                      .order_by(Analysis.created_at.desc()).limit(500)).all()
    return [{"ball_speed_kmh": sp, "metrics": json.loads(mj)} for _, sp, mj in rows]


def save_analysis_and_finish(job_id: str, owner: str, result: dict) -> None:
    """Insert the Analysis row and mark the job done in ONE transaction."""
    metrics = {m["key"]: {"value": m["value"], "unit": m["unit"]} for m in result["metrics"] if m["value"] is not None}
    with session() as s:
        s.add(Analysis(
            id=result["id"], owner_id=owner, bowler_name=result["bowler_name"],
            ball_speed_kmh=result["ball_speed"]["value_kmh"], quality=result["quality"]["level"],
            top_finding=result["findings"][0]["title"] if result["findings"] else None, has_front=result["has_front"],
            metrics_json=json.dumps(metrics, allow_nan=False), result_json=json.dumps(result, allow_nan=False)))
        job = s.get(Job, job_id)
        if job is not None:
            job.status, job.stage, job.progress, job.updated_at, job.lease_until = "done", "Done", 100, utcnow(), None
            s.add(job)
        s.commit()


# Failures the user can get past by marking the key moments by hand; the processed video is kept for MARK_KEEP_H.
MARKABLE = {"EVENTS_NOT_FOUND", "WRONG_CAMERA_ANGLE", "SLOW_MOTION_CLIP"}
MARK_KEEP_H = 24


def run_job(job_id: str) -> None:
    from app.pipeline.orchestrator import run_analysis  # heavy imports (mediapipe) only in the worker

    job = get_job(job_id)
    if job is None:
        return
    params = json.loads(job.params_json)
    update_job(job_id, status="processing", stage="Checking quality", progress=0)
    try:
        hist = history(job.owner_id, params.get("bowler_name") or "Me", job_id)
        result = run_analysis(job_id, params, hist, lambda st, p: progress(job_id, st, p))
        save_analysis_and_finish(job_id, job.owner_id, result)
        count_job("done")
    except BowlIQError as exc:
        count_job(f"failed_{exc.code}")
        log.info("Job %s failed: %s %s", job_id, exc.code, exc.message)
        update_job(job_id, status="failed", error_code=exc.code, error_message=exc.message, error_fix=exc.fix,
                   lease_until=None)
        if exc.code not in MARKABLE:  # fixable by marking the key moments by hand: keep the processed video a while
            AnalysisFiles(job_id).delete()
    except Exception:  # unexpected bug: log details, return a generic message with an error id
        count_job("failed_INTERNAL_ERROR")
        err_id = uuid.uuid4().hex[:8]
        log.error("Job %s crashed (error %s):\n%s", job_id, err_id, traceback.format_exc())
        update_job(job_id, status="failed", error_code="INTERNAL_ERROR",
                   error_message=f"Something went wrong while analysing this clip (error {err_id}).",
                   error_fix="Try again. If it keeps happening, report this error id with the backend log.",
                   lease_until=None)
        AnalysisFiles(job_id).delete()
    finally:
        with _lock:
            _last_progress.pop(job_id, None)
            _stage_start.pop(job_id, None)


def worker_loop(poll_s: float = 1.0) -> None:
    from app.core.maintenance import cleanup

    log.info("Worker %s started", WORKER_ID)
    last_cleanup = time.monotonic()
    while not _stop.is_set():
        if time.monotonic() - last_cleanup > 3600:
            last_cleanup = time.monotonic()
            try:
                cleanup()
            except Exception:
                log.exception("Cleanup failed")
        try:
            job_id = claim_next()
        except Exception:
            log.exception("Claiming a job failed")
            job_id = None
        if job_id:
            run_job(job_id)
            continue
        _wake.wait(poll_s)
        _wake.clear()


_thread: Optional[threading.Thread] = None


def start_embedded_worker() -> None:
    global _thread
    if _thread is None or not _thread.is_alive():
        _stop.clear()
        _thread = threading.Thread(target=worker_loop, name="bowliq-worker", daemon=True)
        _thread.start()


def shutdown() -> None:
    _stop.set()
    _wake.set()


def queue_depth() -> dict[str, int]:
    with session() as s:
        rows = s.exec(select(Job.status, func.count()).group_by(Job.status)).all()
    return {st: n for st, n in rows}

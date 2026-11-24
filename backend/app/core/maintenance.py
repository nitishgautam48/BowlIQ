"""Retention and orphan cleanup."""
from __future__ import annotations

import logging
from datetime import timedelta

from sqlmodel import select

from app.core.config import settings
from app.core.storage import AnalysisFiles, purge_orphans
from app.models.db import Analysis, Job, session, utcnow

log = logging.getLogger("bowliq.maintenance")


def apply_retention() -> int:
    """Delete analyses older than BOWLIQ_RETENTION_DAYS (0 = keep forever)."""
    if settings.retention_days <= 0:
        return 0
    cutoff = utcnow() - timedelta(days=settings.retention_days)
    with session() as s:
        old = s.exec(select(Analysis).where(Analysis.created_at < cutoff)).all()
        ids = [a.id for a in old]
        for a in old:
            job = s.get(Job, a.id)
            if job:
                s.delete(job)
            s.delete(a)
        s.commit()
    for i in ids:
        AnalysisFiles(i).delete()
    if ids:
        log.info("Retention removed %d analyses older than %d days", len(ids), settings.retention_days)
    return len(ids)


def cleanup() -> None:
    apply_retention()
    with session() as s:
        from datetime import timedelta

        from app.core.jobs import MARK_KEEP_H, MARKABLE
        from app.models.db import utcnow
        known = set(s.exec(select(Analysis.id)).all()) | set(
            s.exec(select(Job.id).where(Job.status.in_(["queued", "processing"]))).all()) | set(
            s.exec(select(Job.id).where(Job.status == "failed", Job.error_code.in_(list(MARKABLE)),
                                        Job.updated_at > utcnow() - timedelta(hours=MARK_KEEP_H))).all())
    purge_orphans(known)

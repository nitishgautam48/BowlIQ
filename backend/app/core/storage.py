"""File layout for uploads and analysis outputs, disk checks and retention."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from app.core.config import settings
from app.core.errors import BowlIQError

log = logging.getLogger("bowliq.storage")


class AnalysisFiles:
    def __init__(self, analysis_id: str):
        if not analysis_id.isalnum():  # ids are server-generated hex; refuse anything path-like
            raise BowlIQError("NOT_FOUND", "That analysis doesn't exist.", "Open My deliveries.", 404)
        self.id = analysis_id
        self.dir = settings.analysis_dir(analysis_id)

    def ensure(self) -> "AnalysisFiles":
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        return self

    def input(self, view: str, ext: str) -> Path:
        return self.dir / f"{view}_input{ext}"

    def find_input(self, view: str) -> Path | None:
        for p in self.dir.glob(f"{view}_input.*"):
            return p
        return None

    @property
    def side_prep(self) -> Path:
        return self.dir / "side_prep.mp4"

    @property
    def front_prep(self) -> Path:
        return self.dir / "front_prep.mp4"

    @property
    def annotated(self) -> Path:
        return self.dir / "annotated.mp4"

    @property
    def clean(self) -> Path:
        return self.dir / "clean.mp4"

    @property
    def keypoints(self) -> Path:
        return self.dir / "keypoints.json"

    @property
    def front_clean(self) -> Path:
        return self.dir / "front_clean.mp4"

    @property
    def front_overlay(self) -> Path:
        return self.dir / "front_overlay.npz"

    @property
    def front_keypoints(self) -> Path:
        return self.dir / "front_keypoints.json"

    @property
    def overlay(self) -> Path:
        return self.dir / "overlay.npz"

    @property
    def pdf(self) -> Path:
        return self.dir / "report.pdf"

    @property
    def cache(self) -> Path:
        """Keypoints of every person + metadata after the pose stage: /recompute reuses them (no pose re-run)."""
        return self.dir / "cache.pkl"

    @property
    def debug_dir(self) -> Path:
        """storage/debug/{id}/: signals.png, context.json and key frames for checking the detection."""
        return settings.storage_dir / "debug" / self.id

    def key_frame(self, name: str) -> Path:
        return self.dir / "frames" / f"{name}.jpg"

    def frame_cache(self, n: int, joint: str | None, view: str = "side") -> Path:
        prefix = "f" if view == "side" else "ff"
        return self.dir / "frames" / f"{prefix}{n}_{joint or 'none'}.jpg"

    def delete(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.rmtree(self.debug_dir, ignore_errors=True)


def ensure_free_space(incoming_bytes: int) -> None:
    """Need room for the upload plus the derived videos (about 3× the input)."""
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(settings.storage_dir).free
    if free < 3 * incoming_bytes + 200 * 1024 * 1024:
        raise BowlIQError("DISK_FULL", "The server is low on disk space.",
                          "Delete old deliveries (My deliveries → Delete) or free up disk space, then try again.", 507)


def purge_orphans(known_ids: set[str]) -> int:
    """Remove analysis folders that have neither an analysis row nor an active job."""
    root = settings.storage_dir / "analyses"
    if not root.exists():
        return 0
    n = 0
    for d in root.iterdir():
        if d.is_dir() and d.name not in known_ids and time.time() - d.stat().st_mtime > 3600:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    dbg = settings.storage_dir / "debug"  # diagnostics of deleted analyses
    if dbg.exists():
        for d in dbg.iterdir():
            if d.is_dir() and d.name not in known_ids and time.time() - d.stat().st_mtime > 3600:
                shutil.rmtree(d, ignore_errors=True)
    if n:
        log.info("Removed %d orphaned analysis folder(s)", n)
    return n

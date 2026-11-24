"""Optional GPU pose engine: Ultralytics YOLO26x-pose on CUDA (e.g. an RTX 4060), batched, half precision.

Outputs the same 17 COCO keypoints as the MediaPipe path (heel/toe points left empty) (x, y in pixels, per-keypoint confidence used as visibility),
and goes through the same bowler selection, so everything downstream is unchanged.
Install the optional packages with requirements-gpu.txt (see README → GPU). Licence: AGPL-3.0 (fine for personal use).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import BowlIQError
from app.pipeline.types import N_JOINTS, BowlerSelection, PoseSequence, ProgressFn

log = logging.getLogger("bowliq.pose.yolo")
_lock = threading.Lock()


@lru_cache(maxsize=2)
def yolo_available(require_cuda: bool = False) -> bool:
    try:
        import torch
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return bool(torch.cuda.is_available()) if require_cuda else True


@lru_cache(maxsize=1)
def torch_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ensure_yolo_model() -> Path:
    """Download the pinned YOLO pose weights on first use and verify the SHA-256."""
    path = settings.yolo_model_path
    with _lock:
        if path.exists() and _sha256(path) == settings.yolo_model_sha256:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        log.info("Downloading GPU pose model to %s", path)
        try:
            with urllib.request.urlopen(settings.yolo_model_url, timeout=300) as r, open(tmp, "wb") as f:  # noqa: S310  # nosec B310 (fixed https URL)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise BowlIQError("MODEL_DOWNLOAD_FAILED", "Could not download the GPU pose model.",
                              f"Check your internet connection, or download {settings.yolo_model_url} into backend/models/.",
                              500) from exc
        if _sha256(tmp) != settings.yolo_model_sha256:
            tmp.unlink(missing_ok=True)
            raise BowlIQError("MODEL_INVALID", "The downloaded GPU pose model failed its checksum.",
                              "Delete backend/models/yolo26x-pose.pt and try again.", 500)
        tmp.replace(path)
        return path


@lru_cache(maxsize=1)
def _load_model(path: str, device: str):
    from ultralytics import YOLO

    model = YOLO(path)
    model.to(device)
    return model


class YoloPoseEstimator:
    name = "ultralytics_yolo26x_pose"

    def estimate(self, video_path: Path, fps: float, progress: Optional[ProgressFn] = None,
                 selection: Optional[BowlerSelection] = None) -> PoseSequence:
        from app.pipeline.tracking import Det, RawDetections, appearance, select

        if not yolo_available():
            raise BowlIQError("POSE_INIT_FAILED", "The GPU pose engine isn't installed.",
                              "Install it with `pip install -r requirements-gpu.txt`, or set BOWLIQ_POSE_BACKEND=mediapipe.", 500)
        weights = ensure_yolo_model()
        device = torch_device()
        half = device.startswith("cuda")
        try:
            model = _load_model(str(weights), device)
        except Exception as exc:
            log.error("YOLO pose model failed to load: %s", exc)
            raise BowlIQError("POSE_INIT_FAILED", "Could not start the GPU pose model.",
                              "Reinstall the GPU packages (requirements-gpu.txt) or set BOWLIQ_POSE_BACKEND=mediapipe.", 500) from exc
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise BowlIQError("UNREADABLE_VIDEO", "The processed video could not be opened.", "Try exporting the clip as MP4 (H.264).")
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        imgsz = min(settings.yolo_imgsz, int(np.ceil(max(w, h) / 32) * 32))
        cands: list[list[Det]] = []
        batch: list[np.ndarray] = []

        def flush() -> None:
            if not batch:
                return
            results = model.predict(batch, imgsz=imgsz, device=device, half=half, conf=0.25, verbose=False)
            for frame, r in zip(batch, results, strict=True):
                people = []
                if r.keypoints is not None and r.keypoints.data is not None and len(r.keypoints.data):
                    data = r.keypoints.data.float().cpu().numpy()  # (N, 17, 3): x, y, confidence (COCO order)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    for p in data:
                        kp = np.full((N_JOINTS, 3), np.nan)
                        kp[:17] = p.astype(np.float64)  # no heel/toe points from this engine
                        people.append(Det(kp, None, appearance(hsv, kp)))
                cands.append(people)
            batch.clear()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                batch.append(frame)
                if len(batch) >= settings.yolo_batch:
                    flush()
                    if progress:
                        progress(min(1.0, len(cands) / total))
            flush()
        finally:
            cap.release()
        if progress:
            progress(1.0)
        return select(RawDetections(cands, fps, w, h, "gpu" if half else "cpu"), selection)

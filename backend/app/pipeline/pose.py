"""Stage 3: pose estimation behind a PoseEstimator interface (MediaPipe Tasks PoseLandmarker, heavy).

Runs on every frame in VIDEO mode. When several people are visible, the bowler is the person with the largest
consistent horizontal travel in the first half of the clip, then tracked by nearest centroid.
"""
from __future__ import annotations

import hashlib
import logging
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import BowlIQError
from app.pipeline.params import P
from app.pipeline.tracking import Det, RawDetections, active_span, appearance, body_size, centroid, select  # noqa: F401
from app.pipeline.types import MP_INDEX, N_JOINTS, BowlerSelection, PoseSequence, ProgressFn

log = logging.getLogger("bowliq.pose")
_model_lock = threading.Lock()


class PoseEstimator(Protocol):
    name: str

    def estimate(self, video_path: Path, fps: float, progress: Optional[ProgressFn] = None,
                 selection: Optional[BowlerSelection] = None) -> PoseSequence: ...


# ───────────────────────────── model file ─────────────────────────────

_hash_cache: dict[tuple, str] = {}


def _sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def model_file_ok(path: Path | None = None) -> bool:
    """Present, a valid task bundle, and matching the pinned SHA-256."""
    p = path or settings.pose_model_path
    try:
        return (p.exists() and p.stat().st_size >= 20_000_000 and zipfile.is_zipfile(p)
                and _sha256(p) == settings.pose_model_sha256)
    except OSError:
        return False


def ensure_model() -> Path:
    """Download the pinned pose model on first run and verify its SHA-256."""
    path = settings.pose_model_path
    with _model_lock:
        if model_file_ok(path):
            return path
        if path.exists():
            log.warning("Pose model at %s failed verification; re-downloading.", path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        if not settings.pose_model_url.startswith("https://"):
            raise BowlIQError("MODEL_INVALID", "The pose model URL must use https.", "Fix pose_model_url.", 500)
        log.info("Downloading pose model to %s", path)
        try:
            with urllib.request.urlopen(settings.pose_model_url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310  # nosec B310 (https enforced above)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise BowlIQError(
                "MODEL_DOWNLOAD_FAILED", "Could not download the pose model.",
                f"Check your internet connection, or download {settings.pose_model_url} manually into backend/models/.",
                500) from exc
        if not model_file_ok(tmp):
            tmp.unlink(missing_ok=True)
            raise BowlIQError("MODEL_INVALID", "The downloaded pose model failed its checksum.",
                              "Delete backend/models/pose_landmarker_heavy.task and restart the server to re-download.", 500)
        tmp.replace(path)
        return path


# ───────────────────────────── MediaPipe ─────────────────────────────

_device_cache: dict[str, str] = {}
_gpu_failed = False


def _create_landmarker(model_path: Path, delegate: str, num_poses: int):
    from mediapipe.tasks.python import BaseOptions, vision

    d = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=d),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def open_landmarker(num_poses: int = 4):
    """Return (landmarker, device). GPU delegate when available (auto/gpu), else CPU."""
    model = ensure_model()
    global _gpu_failed
    pref = settings.pose_delegate
    # MediaPipe's Python GPU delegate is only built for Linux/macOS; on Windows "auto" goes straight to CPU.
    try_gpu = pref == "gpu" or (pref == "auto" and sys.platform != "win32" and not _gpu_failed)
    order = (["gpu"] if try_gpu else []) + ["cpu"]
    last_exc: Exception | None = None
    for dev in order:
        try:
            lm = _create_landmarker(model, dev, num_poses)
            _device_cache["device"] = dev
            return lm, dev
        except Exception as exc:  # GPU delegate unsupported on this OS/driver → fall back to CPU
            last_exc = exc
            if dev == "gpu":
                _gpu_failed = True
            log.info("Pose delegate %s unavailable: %s", dev, str(exc)[:200])
    log.error("Pose model failed to start: %s", last_exc)
    raise BowlIQError("POSE_INIT_FAILED", "Could not start the pose model.",
                      "Reinstall mediapipe (`pip install -U mediapipe`) and delete/re-download the model file.", 500)


def detected_device() -> str:
    """Device used by the pose model ('gpu'/'cpu'); probes once if no analysis has run yet."""
    if "device" not in _device_cache:
        try:
            lm, _ = open_landmarker(1)
            lm.close()
        except Exception:
            return "unknown"
    return _device_cache.get("device", "unknown")


class MediaPipePoseEstimator:
    name = "mediapipe_pose_landmarker_heavy"

    def __init__(self, num_poses: int = 4):
        self.num_poses = num_poses

    def estimate(self, video_path: Path, fps: float, progress: Optional[ProgressFn] = None,
                 selection: Optional[BowlerSelection] = None) -> PoseSequence:
        from app.core.config import settings
        from app.pipeline.adapt import TILES, zoom_refine

        dets, w, h, device = self._detect(video_path, fps, [None], progress)
        try:
            seq = select(RawDetections(dets, fps, w, h, device), selection)
        except BowlIQError as exc:
            if exc.code != "POSE_NOT_FOUND" or not settings.adapt_video:
                raise
            # Phase 1c: the bowler may be too small for the detector in the full frame: look again in four enlarged,
            # overlapping quarters of the picture and merge what each finds
            tiled, *_ = self._detect(video_path, fps, TILES, None)
            merged = [_merge(a, b, w) for a, b in zip(dets, tiled, strict=False)]
            seq = select(RawDetections(merged, fps, w, h, device), selection)
            seq.tiled = True
        return zoom_refine(seq, video_path, fps, progress)

    def _detect(self, video_path: Path, fps: float, regions: list, progress: Optional[ProgressFn]
                ) -> tuple[list[list[Det]], int, int, str]:
        """One pose pass. regions: None = the whole frame, or (x0, y0, x1, y1) fractions of it, each with its own
        landmarker (VIDEO mode keeps per-region tracking)."""
        import mediapipe as mp

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise BowlIQError("UNREADABLE_VIDEO", "The processed video could not be opened.", "Try exporting the clip as MP4 (H.264).")
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        lms, device = [], "cpu"
        for _ in regions:
            lm, device = open_landmarker(self.num_poses)
            lms.append(lm)
        dets: list[list[Det]] = []
        try:
            i = 0
            last_ts = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = int(round(i * 1000.0 / fps))
                ts = max(ts, last_ts + 1)
                last_ts = ts
                people: list[Det] = []
                hsv = None
                for lm, reg in zip(lms, regions, strict=True):
                    if reg is None:
                        x0 = y0 = 0
                        sub, sx, sy = frame, w, h
                    else:
                        x0, y0 = int(reg[0] * w), int(reg[1] * h)
                        x1, y1 = int(reg[2] * w), int(reg[3] * h)
                        sub = cv2.resize(frame[y0:y1, x0:x1], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        sx, sy = x1 - x0, y1 - y0
                    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                       data=cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)), ts)
                    if res.pose_landmarks and hsv is None:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    for lmk in res.pose_landmarks or []:
                        arr = np.empty((N_JOINTS, 3), dtype=np.float64)
                        z = np.empty(N_JOINTS, dtype=np.float64)
                        for k, mi in enumerate(MP_INDEX):
                            p = lmk[mi]
                            vis = p.visibility if p.visibility is not None else 0.0
                            pres = p.presence if getattr(p, "presence", None) is not None else 1.0
                            arr[k] = (x0 + p.x * sx, y0 + p.y * sy, min(vis, pres))
                            z[k] = p.z * sx  # MediaPipe depth is on the same scale as x
                        people.append(Det(arr, z, appearance(hsv, arr)))
                dets.append(people)
                i += 1
                if progress and (i % 8 == 0 or i == total):
                    progress(min(1.0, 0.85 * i / total))
        finally:
            for lm in lms:
                lm.close()
            cap.release()
        return dets, w, h, device


def _merge(full: list[Det], tiled: list[Det], w: int) -> list[Det]:
    """Detections from the full frame plus those from the tiles that aren't the same person (tiles overlap)."""
    from app.pipeline.tracking import centroid
    out = list(full)
    for d in tiled:
        c = centroid(d.kp)
        if not np.all(np.isfinite(c)):
            continue
        if all(np.linalg.norm(c - centroid(e.kp)) > 0.04 * w for e in out if np.all(np.isfinite(centroid(e.kp)))):
            out.append(d)
    return out


def finalize_candidates(cands: list[list[np.ndarray]], fps: float, w: int, h: int, device: str,
                        selection: Optional[BowlerSelection] = None) -> PoseSequence:
    """Compatibility helper: per-frame keypoint arrays (17 or 21 joints, no appearance) → selected bowler."""
    def pad(p: np.ndarray) -> np.ndarray:
        if p.shape[0] >= N_JOINTS:
            return p[:N_JOINTS, :3]
        return np.vstack([p[:, :3], np.full((N_JOINTS - p.shape[0], 3), np.nan)])

    dets = [[Det(pad(np.asarray(p, float))) for p in frame] for frame in cands]
    return select(RawDetections(dets, fps, w, h, device), selection)


# ───────────────────────────── engine selection ─────────────────────────────

def get_estimator() -> PoseEstimator:
    """BOWLIQ_POSE_BACKEND picks the PRIMARY 2D estimator:
      auto      perception.auto_primary in config/pipeline.yaml: "previous" (default, unchanged from v3: YOLO on a
                CUDA GPU when the optional GPU packages are installed, otherwise MediaPipe), "mediapipe", or "rtmpose"
                (RTMDet + ByteTrack + RTMPose-Halpe26 when ONNX Runtime is installed)
      rtm       RTMPose        yolo   Ultralytics YOLO pose        mediapipe   MediaPipe only
      vitpose   reserved for ViTPose (not installed yet: falls back with a clear error)
    The SECONDARY estimator (perception.secondary in config/pipeline.yaml, MediaPipe by default) cross-checks the
    primary on the bowler crop; if the primary can't find the bowler at all, MediaPipe takes over."""
    from app.pipeline.fusion import MultiModelEstimator
    from app.pipeline.pose_yolo import YoloPoseEstimator, yolo_available
    from app.pipeline.rtm import RtmPoseEstimator, onnx_available

    choice = settings.pose_backend
    if choice == "vitpose":
        return FallbackPoseEstimator(ViTPoseEstimator(), MediaPipePoseEstimator())
    if choice == "rtm" or (choice == "auto" and P.perception.auto_primary == "rtmpose" and onnx_available()):
        return MultiModelEstimator(FallbackPoseEstimator(RtmPoseEstimator(), MediaPipePoseEstimator()),
                                   P.perception.secondary)
    if choice == "yolo" or (choice == "auto" and P.perception.auto_primary == "previous" and yolo_available(require_cuda=True)):
        return MultiModelEstimator(FallbackPoseEstimator(YoloPoseEstimator(), MediaPipePoseEstimator()),
                                   P.perception.secondary)
    return MediaPipePoseEstimator()


class ViTPoseEstimator:
    """Placeholder behind the PoseEstimator interface so ViTPose can be added later (BOWLIQ_POSE_BACKEND=vitpose)."""
    name = "vitpose"

    def estimate(self, video_path: Path, fps: float, progress: Optional[ProgressFn] = None,
                 selection: Optional[BowlerSelection] = None) -> PoseSequence:
        raise BowlIQError("POSE_NOT_FOUND", "The ViTPose engine isn't installed yet.",
                          "Set BOWLIQ_POSE_BACKEND=auto (RTMPose) or mediapipe.")


class FallbackPoseEstimator:
    """Use the primary (GPU) engine; if it can't find the bowler (e.g. unusual footage), retry with the fallback."""

    def __init__(self, primary: PoseEstimator, fallback: PoseEstimator):
        self.primary, self.fallback = primary, fallback
        self.name = primary.name
        self.used_fallback = False

    def estimate(self, video_path: Path, fps: float, progress: Optional[ProgressFn] = None,
                 selection: Optional[BowlerSelection] = None) -> PoseSequence:
        try:
            return self.primary.estimate(video_path, fps, progress, selection)
        except BowlIQError as exc:
            if exc.code != "POSE_NOT_FOUND":
                raise
            log.warning("%s could not find the bowler (%s); retrying with %s", self.primary.name, exc.message,
                        self.fallback.name)
            self.used_fallback = True
            return self.fallback.estimate(video_path, fps, progress, selection)


def describe_engine() -> tuple[str, str]:
    """(engine name, device) for /health, probed once at startup."""
    est = get_estimator()
    if est.name.startswith("rtmpose"):
        from app.pipeline.rtm import device_name
        return est.name, device_name()
    if est.name.startswith("ultralytics"):  # FallbackPoseEstimator reports the primary engine's name
        from app.pipeline.pose_yolo import torch_device

        return est.name, "gpu" if torch_device().startswith("cuda") else "cpu"
    return est.name, detected_device()


# ───────────────────────────── bowler selection ─────────────────────────────


# compatibility aliases (older scripts/tests)
_centroid = centroid
_size = body_size


def select_bowler(cands: list[list[np.ndarray]], fps: float) -> np.ndarray:
    """Compatibility: keypoints (F, N_JOINTS, 3) of the automatically selected bowler."""
    return finalize_candidates(cands, fps, 0, 0, "cpu").kp

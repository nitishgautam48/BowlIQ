"""Primary pose engine: RTMDet person detection → ByteTrack → RTMPose (Halpe-26: body + heels + toes), top-down.

The models are OpenMMLab's own RTMDet-m (person) and RTMPose-m Halpe-26 networks in their official ONNX exports
(download.openmmlab.com, Apache-2.0), run with ONNX Runtime: CUDA on an NVIDIA GPU (TensorRT with FP16 when that
execution provider is installed), CPU otherwise. MMPose/MMDetection themselves are not used at run time because
their compiled `mmcv` dependency has no reliable Windows/CUDA wheels; pre- and post-processing below follow the
models' exported pipeline.json exactly (keep-ratio resize + pad 114 for RTMDet; 1.25× box padding, fixed-aspect
affine crop, RGB normalisation and SimCC decoding for RTMPose).

Every person found is tracked with ByteTrack and given a pose; the bowler is then chosen from the tracks by the
run-up pattern (tracking.score_track), never by size.
"""
from __future__ import annotations

import contextlib
import io
import hashlib
import logging
import threading
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import BowlIQError
from app.pipeline.params import P
from app.pipeline.types import N_JOINTS, BowlerSelection, PoseSequence, ProgressFn

log = logging.getLogger("bowliq.rtm")
_lock = threading.Lock()
R = P.perception

BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
MODELS = {
    "rtmdet_m_person": ("rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip",
                        "716262a99aec134232e2420ae2b1c5bd4a5c0c9a0bd5484067cd1920f881f32e"),
    "rtmpose_m_halpe26": ("rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
                          "55b81170e236040b59fc792ad0a8315301ac4c079a3bdb1095d838aad3088d18"),
}
# Halpe-26 → our 21 joints (COCO-17 + heel and big toe per foot); head/neck kept separately for 3D lifting
HALPE_TO_OURS = list(range(17)) + [24, 25, 20, 21]  # left_heel, right_heel, left_foot_index(big toe), right_...
HALPE_HEAD, HALPE_NECK = 17, 18


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ensure_onnx(key: str) -> Path:
    """Download (once), verify the pinned SHA-256 and unpack an OpenMMLab ONNX bundle → path to end2end.onnx."""
    name, sha = MODELS[key]
    d = settings.models_dir / "rtm"
    d.mkdir(parents=True, exist_ok=True)
    onnx = d / f"{key}.onnx"
    if onnx.exists() and onnx.stat().st_size > 1_000_000:
        return onnx
    with _lock:
        if onnx.exists():
            return onnx
        z = d / name
        if not z.exists() or _sha256(z) != sha:
            log.info("Downloading %s", name)
            tmp = z.with_suffix(".part")
            try:
                with urllib.request.urlopen(BASE + name, timeout=300) as r, open(tmp, "wb") as f:  # noqa: S310  # nosec B310 (fixed https URL)
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                raise BowlIQError("MODEL_DOWNLOAD_FAILED", f"Could not download the {key} model.",
                                  "Check the internet connection once, or set BOWLIQ_POSE_BACKEND=mediapipe.", 500) from exc
            if _sha256(tmp) != sha:
                tmp.unlink(missing_ok=True)
                raise BowlIQError("MODEL_INVALID", f"The downloaded {key} model failed its checksum.",
                                  "Delete backend/models/rtm and try again.", 500)
            tmp.replace(z)
        with zipfile.ZipFile(z) as zf:
            member = next(m for m in zf.namelist() if m.endswith("end2end.onnx"))
            with zf.open(member) as src, open(onnx, "wb") as dst:
                dst.write(src.read())
    return onnx


def onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def providers() -> list:
    import onnxruntime as ort
    av = ort.get_available_providers()
    want = []
    if "TensorrtExecutionProvider" in av and R.use_tensorrt:
        want.append(("TensorrtExecutionProvider", {"trt_fp16_enable": bool(R.fp16), "trt_engine_cache_enable": True,
                                                   "trt_engine_cache_path": str(settings.models_dir / "rtm" / "trt")}))
    if "CUDAExecutionProvider" in av:
        want.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
    want.append("CPUExecutionProvider")
    return want


def device_name() -> str:
    try:
        import onnxruntime as ort
        return "gpu" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    except ImportError:
        return "cpu"


_sessions: dict[str, object] = {}


def session(key: str):
    if key not in _sessions:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            # The CUDA/cuDNN DLLs come with PyTorch (torch\lib): importing torch loads them into the process, so
            # onnxruntime-gpu finds them. preload_dlls() only searches the separate nvidia-* pip packages, so it is
            # asked quietly, for setups that have those instead.
            with contextlib.suppress(Exception):
                import torch  # noqa: F401
            with contextlib.suppress(Exception), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                ort.preload_dlls()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _sessions[key] = ort.InferenceSession(str(ensure_onnx(key)), opts, providers=providers())
    return _sessions[key]


# ───────────────────────────── RTMDet ─────────────────────────────

def detect_people(frame: np.ndarray) -> np.ndarray:
    """(N, 5) x0, y0, x1, y1, score of the people in a BGR frame."""
    sess = session("rtmdet_m_person")
    h, w = frame.shape[:2]
    s = min(640 / h, 640 / w)
    img = np.full((640, 640, 3), 114, np.uint8)
    img[: int(h * s), : int(w * s)] = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) - np.array([103.53, 116.28, 123.675], np.float32)) / np.array([57.375, 57.12, 58.395], np.float32)
    dets, labels = sess.run(None, {"input": x.transpose(2, 0, 1)[None]})
    d = dets[0]
    keep = (d[:, 4] >= R.det_score) & (labels[0] == 0)
    d = d[keep].astype(np.float64)
    d[:, :4] /= s
    d[:, [0, 2]] = d[:, [0, 2]].clip(0, w - 1)
    d[:, [1, 3]] = d[:, [1, 3]].clip(0, h - 1)
    return d[(d[:, 2] - d[:, 0] > 4) & (d[:, 3] - d[:, 1] > 8)]


# ───────────────────────────── ByteTrack ─────────────────────────────

def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


@dataclass
class _Track:
    tid: int
    x: np.ndarray  # Kalman state: cx, cy, aspect, h, vx, vy, va, vh
    Pm: np.ndarray
    last: int
    hits: int = 1

    def box(self) -> np.ndarray:
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class ByteTracker:
    """ByteTrack (Zhang et al., 2022): Kalman-predicted boxes matched by IoU to high-score detections first, then the
    remaining tracks to low-score detections (people blurred mid-delivery keep their identity)."""
    high: float = 0.5
    low: float = 0.1
    match_iou: float = 0.3
    max_lost: int = 30
    tracks: list = field(default_factory=list)
    next_id: int = 0

    F = np.eye(8)
    for _i in range(4):
        F[_i, _i + 4] = 1.0
    H = np.eye(4, 8)

    @staticmethod
    def _xyah(b: np.ndarray) -> np.ndarray:
        w, h = b[2] - b[0], b[3] - b[1]
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w / max(h, 1e-6), h])

    def _noise(self, h: float) -> tuple[np.ndarray, np.ndarray]:
        sp, sv = 1 / 20, 1 / 160
        q = np.square([sp * h, sp * h, 1e-2, sp * h, sv * h, sv * h, 1e-5, sv * h])
        r = np.square([sp * h, sp * h, 1e-1, sp * h])
        return np.diag(q), np.diag(r)

    def _assign(self, tracks: list, boxes: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        from scipy.optimize import linear_sum_assignment
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        m = iou(np.array([t.box() for t in tracks]), boxes[:, :4])
        r, c = linear_sum_assignment(-m)
        pairs = [(i, j) for i, j in zip(r, c, strict=True) if m[i, j] >= self.match_iou]
        ti = {i for i, _ in pairs}
        dj = {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in ti], [j for j in range(len(boxes)) if j not in dj]

    def update(self, f: int, dets: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Detections (N, 5) of frame f → [(track id, detection row)]."""
        for t in self.tracks:  # predict
            Q, _ = self._noise(t.x[3])
            t.x = self.F @ t.x
            t.Pm = self.F @ t.Pm @ self.F.T + Q
        hi, lo = dets[dets[:, 4] >= self.high], dets[(dets[:, 4] < self.high) & (dets[:, 4] >= self.low)]
        out: list[tuple[int, np.ndarray]] = []

        def correct(t: _Track, d: np.ndarray) -> None:
            _, Rn = self._noise(t.x[3])
            S = self.H @ t.Pm @ self.H.T + Rn
            K = t.Pm @ self.H.T @ np.linalg.inv(S)
            t.x = t.x + K @ (self._xyah(d) - self.H @ t.x)
            t.Pm = (np.eye(8) - K @ self.H) @ t.Pm
            t.last, t.hits = f, t.hits + 1
            out.append((t.tid, d))

        pairs, free_t, free_d = self._assign(self.tracks, hi)
        for i, j in pairs:
            correct(self.tracks[i], hi[j])
        rest = [self.tracks[i] for i in free_t]
        pairs2, _, _ = self._assign(rest, lo)
        for i, j in pairs2:
            correct(rest[i], lo[j])
        for j in free_d:  # new tracks from unmatched confident detections
            d = hi[j]
            x = np.concatenate([self._xyah(d), np.zeros(4)])
            h = x[3]
            Pm = np.diag(np.square([2 * h / 20, 2 * h / 20, 1e-2, 2 * h / 20, 10 * h / 160, 10 * h / 160, 1e-5, 10 * h / 160]))
            t = _Track(self.next_id, x, Pm, f)
            self.next_id += 1
            self.tracks.append(t)
            out.append((t.tid, d))
        self.tracks = [t for t in self.tracks if f - t.last <= self.max_lost]
        return out

    def coasting(self, f: int, min_hits: int = 3) -> list[tuple[int, np.ndarray]]:
        """Confirmed tracks with no detection in frame f: their Kalman-predicted boxes (the detector can miss a
        blurred bowler at release; the pose model is then run on where the bowler should be)."""
        return [(t.tid, np.array([*t.box(), 0.0])) for t in self.tracks if t.last != f and t.hits >= min_hits]


# ───────────────────────────── RTMPose ─────────────────────────────

POSE_W, POSE_H = 192, 256
_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_STD = np.array([58.395, 57.12, 57.375], np.float32)


def _crop(frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = (box[2] - box[0]) * 1.25, (box[3] - box[1]) * 1.25
    ar = POSE_W / POSE_H
    if bw > bh * ar:
        bh = bw / ar
    else:
        bw = bh * ar
    src = np.float32([[cx, cy], [cx, cy - bw * 0.5], [cx - bw * 0.5, cy - bw * 0.5]])
    dst = np.float32([[POSE_W / 2, POSE_H / 2], [POSE_W / 2, POSE_H / 2 - POSE_W / 2], [0, POSE_H / 2 - POSE_W / 2]])
    M = cv2.getAffineTransform(src, dst)
    img = cv2.warpAffine(frame, M, (POSE_W, POSE_H), flags=cv2.INTER_LINEAR)
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) - _MEAN) / _STD
    return img.transpose(2, 0, 1), np.array([cx, cy]), np.array([bw, bh])


def estimate_poses(frame: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Halpe-26 keypoints (N, 26, 2) in frame pixels and scores (N, 26) for each box."""
    if len(boxes) == 0:
        return np.zeros((0, 26, 2)), np.zeros((0, 26))
    crops = [_crop(frame, b) for b in boxes]
    x = np.stack([c[0] for c in crops]).astype(np.float32)
    sx, sy = session("rtmpose_m_halpe26").run(None, {"input": x})
    lx, ly = sx.argmax(-1), sy.argmax(-1)
    score = 0.5 * (sx.max(-1) + sy.max(-1))
    kp = np.stack([lx, ly], -1).astype(np.float64) / 2.0  # SimCC split ratio 2
    for i, (_, c, s) in enumerate(crops):
        kp[i] = kp[i] / np.array([POSE_W, POSE_H]) * s + c - s / 2
    return kp, score


def calibrate(sc: np.ndarray) -> np.ndarray:
    """RTMPose SimCC scores → BowlIQ visibility. The two scales differ (OpenMMLab's keypoint threshold is 0.3; BowlIQ's
    visibility threshold is smoothing.vis_threshold), so scores are scaled to make the two thresholds coincide."""
    return np.clip(sc * (P.smoothing.vis_threshold / R.rtm_keypoint_threshold), 0, 1)


def halpe_to_ours(kp: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """(26, 2) + (26,) → (N_JOINTS, 3) x, y, visibility (calibrated score)."""
    out = np.full((N_JOINTS, 3), np.nan)
    out[:, :2] = kp[HALPE_TO_OURS]
    out[:, 2] = calibrate(sc[HALPE_TO_OURS])
    return out


# ───────────────────────────── estimator ─────────────────────────────

class RtmPoseEstimator:
    """PoseEstimator: RTMDet + ByteTrack + RTMPose-Halpe26 on every person, bowler chosen by run-up pattern."""
    name = "rtmpose_m_halpe26"

    def estimate(self, video_path: Path, fps: float, progress: Optional[ProgressFn] = None,
                 selection: Optional[BowlerSelection] = None) -> PoseSequence:
        from app.pipeline.tracking import Det, RawDetections, appearance, select

        if not onnx_available():
            raise BowlIQError("POSE_INIT_FAILED", "The RTMPose engine needs ONNX Runtime.",
                              "Install it with `pip install -r requirements-rtm.txt`, or set BOWLIQ_POSE_BACKEND=mediapipe.", 500)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise BowlIQError("UNREADABLE_VIDEO", "The processed video could not be opened.", "Try exporting the clip as MP4 (H.264).")
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracker = ByteTracker(high=R.track_high, low=R.track_low, match_iou=R.track_iou,
                              max_lost=max(1, int(round(R.track_lost_s * fps))))
        frames: list[list] = []
        extras: list[dict] = []
        f = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                dets = detect_people(frame)
                tracked = tracker.update(f, dets)
                coast = tracker.coasting(f)
                allt = tracked + coast
                boxes = np.array([d[:4] for _, d in allt]) if allt else np.zeros((0, 4))
                kps, scs = estimate_poses(frame, boxes)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                people, ex = [], {}
                for i, ((tid, _d), kp, sc) in enumerate(zip(allt, kps, scs, strict=True)):
                    if i >= len(tracked) and float(np.mean(sc)) < R.coast_min_score:
                        continue  # nobody convincing in the predicted box
                    ours = halpe_to_ours(kp, sc)
                    people.append(Det(ours, None, appearance(hsv, ours), tid=int(tid)))
                    ex[int(tid)] = np.array([[*kp[HALPE_HEAD], sc[HALPE_HEAD]], [*kp[HALPE_NECK], sc[HALPE_NECK]]])
                frames.append(people)
                extras.append(ex)
                f += 1
                if progress and f % 8 == 0:
                    progress(min(0.85, 0.85 * f / total))
        finally:
            cap.release()
        seq = select(RawDetections(frames, fps, w, h, device_name()), selection)
        # head + neck of the chosen person (for 3D lifting), by track id
        hn = np.full((len(frames), 2, 3), np.nan)
        tids = getattr(seq, "tids", None)
        if tids is not None:
            for i, tid in enumerate(tids):
                if tid is not None and tid in extras[i]:
                    hn[i] = extras[i][tid]
        seq.head_neck = hn
        if progress:
            progress(0.85)
        return seq

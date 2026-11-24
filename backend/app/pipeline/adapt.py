"""Phase 1c: adapt to imperfect video instead of rejecting it.

1. Camera movement (handheld shake, following the bowler): track the background between frames (people masked out),
   fit a rotation + scale + shift per frame, chain them into a camera path and express the skeleton in one fixed
   reference view. Measurements then behave as if the camera had been on a tripod. The picture itself (and the
   skeleton drawn on it) stays as filmed.
2. Small / far-away / dark bowler: if the full frame finds too little, search four enlarged quarters of the picture;
   then run the pose model again on a zoomed-in crop around the tracked bowler.
Tried and dropped (measured, made things no better or worse):
* local contrast enhancement (CLAHE) for dark footage: on a darkened copy of a real clip the zoomed pass alone
  recovered the correct key moments; adding CLAHE made them wrong (wrong bowling arm, ~20 frames late);
* key-moment times between frames (speed-threshold crossing, wrist-height parabola): dropping three clips from
  60 to 30 fps, the mean timing error was 16.7 ms with it vs 16.2 ms without, and contacts came out 13–20 ms early.

Each one switches itself on only when the clip needs it, and the report says what was adjusted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.pipeline.params import P

log = logging.getLogger("bowliq.adapt")
CAM, ZM = P.camera, P.zoom

# ───────────────────────────── 1. camera path ─────────────────────────────

WORK_W = CAM.work_width  # frames are analysed at this width (speed); transforms are scaled back
MIN_POINTS = CAM.min_points  # background points needed for a trustworthy frame-to-frame fit
MOVED_PX = CAM.moved_frac  # camera counts as moving if the view shifts by more than this fraction of the width ...
MOVED_DEG = CAM.moved_deg  # ... or turns by more than this


@dataclass
class CameraPath:
    """to_ref[f]: 3×3 matrix taking pixel coordinates of frame f into the reference frame's pixel coordinates."""
    to_ref: np.ndarray
    ref: int
    moved: bool
    ok: bool
    max_shift_px: float = 0.0
    max_turn_deg: float = 0.0
    failed_frames: int = 0
    notes: list[str] = field(default_factory=list)


def _mask(shape: tuple[int, int], boxes: list[tuple[float, float, float, float]], s: float) -> np.ndarray:
    m = np.full(shape, 255, np.uint8)
    for x0, y0, x1, y1 in boxes:
        if np.all(np.isfinite([x0, y0, x1, y1])):
            m[max(0, int(y0 * s)):max(0, int(y1 * s)), max(0, int(x0 * s)):max(0, int(x1 * s))] = 0
    return m


def people_boxes(raw, n: int, pad: float = 0.3) -> list[list[tuple[float, float, float, float]]]:
    """Boxes around every detected person in every frame (so moving people never count as 'background')."""
    out: list[list[tuple]] = [[] for _ in range(n)]
    if raw is None:
        return out
    for f, frame in enumerate(raw.dets[:n]):
        for d in frame:
            xy = d.kp[np.isfinite(d.kp[:, 0]), :2]
            if len(xy) < 3:
                continue
            x0, y0 = xy.min(0)
            x1, y1 = xy.max(0)
            w, h = x1 - x0, y1 - y0
            out[f].append((x0 - pad * w - 0.2 * h, y0 - pad * h, x1 + pad * w + 0.2 * h, y1 + pad * h))
    return out


MAX_STEP_SCALE = CAM.max_step_scale  # a phone can't zoom more than this between two frames
MAX_STEP_TURN = CAM.max_step_turn_deg  # … or turn more than this
MIN_INLIER_FRAC = CAM.min_inlier_frac


def estimate_camera_path(video_path: Path, boxes: list[list[tuple]], ref: int,
                         span: Optional[tuple[int, int]] = None) -> CameraPath:
    """Camera path over `span` (frames outside it keep the nearest estimate). Only the delivery matters, and every
    chained step adds a little error, so a short span keeps the drift small."""
    cap = cv2.VideoCapture(str(video_path))
    n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = span if span is not None else (0, n_all - 1)
    lo, hi = max(0, lo), min(n_all - 1, hi)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    s = min(1.0, WORK_W / max(1, w))
    S, Si = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
    to_first = [np.eye(3)]  # frame lo+k → frame lo
    prev = None
    failed = 0
    f = lo
    while f <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            M = None
            mask = _mask(prev.shape, (boxes[f - 1] if f - 1 < len(boxes) else []) + (boxes[f] if f < len(boxes) else []), s)
            p0 = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=7, mask=mask)
            if p0 is not None and len(p0) >= MIN_POINTS:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                good = st.ravel() == 1
                if good.sum() >= MIN_POINTS:
                    A, inl = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC,
                                                         ransacReprojThreshold=1.5, maxIters=2000, confidence=0.99)
                    if A is not None and inl is not None and inl.sum() >= max(MIN_POINTS * 0.6, MIN_INLIER_FRAC * good.sum()):
                        sc = float(np.hypot(A[0, 0], A[1, 0]))
                        turn = float(np.degrees(abs(np.arctan2(A[1, 0], A[0, 0]))))
                        if abs(sc - 1) <= MAX_STEP_SCALE and turn <= MAX_STEP_TURN:
                            M = np.vstack([A, [0, 0, 1]])  # prev → current (work scale)
            if M is None:
                failed += 1
                M = np.eye(3)  # assume no movement for this step
            Mfull = Si @ M @ S  # prev → current (full scale)
            to_first.append(to_first[-1] @ np.linalg.inv(Mfull))
        prev = g
        f += 1
    cap.release()
    k = len(to_first)
    r = int(np.clip(ref - lo, 0, k - 1))
    inv_ref = np.linalg.inv(to_first[r])
    part = np.array([inv_ref @ T for T in to_first])
    to_ref = np.empty((n_all, 3, 3))
    to_ref[lo:lo + k] = part
    to_ref[:lo] = part[0]
    to_ref[lo + k:] = part[-1]
    n = k
    w = int(w)
    shifts = np.hypot(part[:, 0, 2], part[:, 1, 2])
    turns = np.degrees(np.abs(np.arctan2(part[:, 1, 0], part[:, 0, 0])))
    max_shift, max_turn = float(shifts.max()), float(turns.max())
    moved = max_shift > MOVED_PX * w or max_turn > MOVED_DEG
    ok = failed <= 0.2 * max(1, n - 1)
    return CameraPath(to_ref, ref, moved, ok, max_shift, max_turn, failed)


def apply_path(kp_xy: np.ndarray, path: CameraPath) -> np.ndarray:
    """(F, J, 2) image coordinates → reference-view coordinates."""
    out = np.full_like(kp_xy, np.nan)
    n = min(len(kp_xy), len(path.to_ref))
    pts = np.concatenate([kp_xy[:n], np.ones(kp_xy[:n].shape[:-1] + (1,))], axis=-1)
    res = np.einsum("fij,fkj->fki", path.to_ref[:n], pts)
    out[:n] = res[..., :2] / res[..., 2:3]
    return out


def residual_motion(video_path: Path, boxes: np.ndarray, path: Optional[CameraPath], samples: int = 24
                    ) -> Optional[float]:
    """Like quality.camera_motion (background shift in frame widths per second, frames 2 apart), but after undoing
    the estimated camera movement: what is left is shake the correction couldn't remove."""
    from app.pipeline.quality import background_shift
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    if n < 4:
        cap.release()
        return None
    idx = np.unique(np.linspace(0, n - 3, min(samples, n - 2)).astype(int))
    vals = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok1, f1 = cap.read()
        cap.read()
        ok2, f2 = cap.read()
        if not (ok1 and ok2):
            continue
        g1 = cv2.GaussianBlur(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        g2 = cv2.GaussianBlur(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        h, w = g1.shape
        valid = np.full((h, w), 255, np.uint8)
        if path is not None and i + 2 < len(path.to_ref):
            T = np.linalg.inv(path.to_ref[i + 2]) @ path.to_ref[i]  # frame i → frame i+2
            g1 = cv2.warpAffine(g1, T[:2], (w, h), flags=cv2.INTER_LINEAR)
            valid = cv2.warpAffine(valid, T[:2], (w, h), flags=cv2.INTER_NEAREST)
        mask = valid > 0
        for j in (i, min(i + 2, len(boxes) - 1)):
            b = boxes[j] if j < len(boxes) else None
            if b is not None and np.all(np.isfinite(b)):
                x0, y0, x1, y1 = (int(max(0, v)) for v in b)
                mask[y0:y1, x0:x1] = False
        if mask.mean() < 0.2:
            continue
        vals.append(background_shift(g1, g2, mask, fps))
    cap.release()
    return float(np.median(vals)) if vals else None


# ───────────────────────────── 2. small bowler: enlarged quarters, then a zoomed crop ─────────────────────────────

TILES = [(0.0, 0.0, 0.6, 0.6), (0.4, 0.0, 1.0, 0.6), (0.0, 0.4, 0.6, 1.0), (0.4, 0.4, 1.0, 1.0)]

# ───────────────────────────── zoomed crop ─────────────────────────────

# The zoomed pass always runs when the bowler covers too few pixels for the pose model's input (zoom.always_model_frac);
# otherwise only as a rescue when the first pass finds no delivery.
ZOOM_OUT_H = ZM.out_px  # crop is enlarged to this model input size
MIN_ZOOM = ZM.min_zoom  # a smaller enlargement gains nothing over the first pass


def _body_px(kp: np.ndarray) -> np.ndarray:
    from app.pipeline.types import J
    top = np.fmin(kp[:, J["nose"], 1], np.fmin(kp[:, J["left_shoulder"], 1], kp[:, J["right_shoulder"], 1]))
    bot = np.fmax(kp[:, J["left_ankle"], 1], kp[:, J["right_ankle"], 1])
    return bot - top


def _align_labels(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    """Use the left/right labelling (as-is or mirrored) that agrees with the first pass, so the two passes never
    disagree about which leg or arm is which."""
    from app.pipeline.types import ALL_JOINTS, J
    if not np.isfinite(old[:, 0]).any():
        return new
    sw = new.copy()
    for name in ALL_JOINTS:
        if name.startswith("left_"):
            r = name.replace("left_", "right_")
            sw[[J[name], J[r]]] = new[[J[r], J[name]]]

    def cost(a: np.ndarray) -> float:
        d = np.linalg.norm(a[:, :2] - old[:, :2], axis=1)
        return float(np.nanmean(d)) if np.isfinite(d).any() else np.inf
    return sw if cost(sw) < cost(new) else new


def zoom_refine(seq, video_path: Path, fps: float, progress=None, force: bool = False):
    """Second pose pass on an enlarged crop around the tracked bowler when he is small in the picture.

    Each frame's crop is centred on the bowler (smoothed over time so the crop doesn't jump), the refined joints are
    mapped back to full-frame pixels and replace the first pass where the model is at least as sure of them."""
    from app.core.config import settings
    if not settings.adapt_video or seq.active_span is None:
        return seq
    a, b = seq.active_span
    kp = seq.kp
    hgt = _body_px(kp[a:b + 1])
    hgt = hgt[np.isfinite(hgt)]
    if len(hgt) < 5 or (not force and np.median(hgt) >= ZM.always_model_frac * P.pose_model.input_px):
        return seq
    import mediapipe as mp

    from app.pipeline.pose import open_landmarker
    from app.pipeline.types import MP_INDEX, N_JOINTS
    idx = np.arange(a, b + 1)
    cx = np.array([np.nanmedian(kp[f, :, 0]) if np.isfinite(kp[f, :, 0]).any() else np.nan for f in idx])
    cy = np.array([np.nanmedian(kp[f, :, 1]) if np.isfinite(kp[f, :, 1]).any() else np.nan for f in idx])
    ok = np.isfinite(cx) & np.isfinite(cy)
    if ok.sum() < 5:
        return seq
    cx, cy = np.interp(idx, idx[ok], cx[ok]), np.interp(idx, idx[ok], cy[ok])
    k = max(3, int(P.context.bowler.speed_smooth_s * fps) | 1)  # smooth the crop path
    cx = np.convolve(np.pad(cx, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    cy = np.convolve(np.pad(cy, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    side = float(np.clip(np.percentile(hgt, 90) * 1.9, ZM.min_crop_frac * min(seq.width, seq.height),
                         min(seq.width, seq.height) * 2))
    scale = ZOOM_OUT_H / side
    if scale < MIN_ZOOM:
        return seq
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    lm, _ = open_landmarker(2)
    refined = kp.copy()
    improved = 0
    try:
        last_ts = -1
        for i, f in enumerate(idx):
            okf, frame = cap.read()
            if not okf:
                break
            x0, y0 = cx[i] - side / 2, cy[i] - side / 2
            M = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
            crop = cv2.warpAffine(frame, M, (ZOOM_OUT_H, ZOOM_OUT_H), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
            ts = max(int(round(f * 1000.0 / fps)), last_ts + 1)
            last_ts = ts
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), ts)
            best, bd = None, np.inf
            for lms in res.pose_landmarks or []:
                arr = np.empty((N_JOINTS, 3))
                for j, mi in enumerate(MP_INDEX):
                    p = lms[mi]
                    vis = p.visibility if p.visibility is not None else 0.0
                    pres = p.presence if getattr(p, "presence", None) is not None else 1.0
                    arr[j] = (p.x * ZOOM_OUT_H / scale + x0, p.y * ZOOM_OUT_H / scale + y0, min(vis, pres))
                d = np.nanmedian(np.linalg.norm(arr[:, :2] - kp[f, :, :2], axis=1)) if np.isfinite(kp[f, :, 0]).any() \
                    else np.hypot(*(np.nanmedian(arr[:, :2], 0) - [cx[i], cy[i]]))
                if d < bd:
                    best, bd = arr, d
            if best is None or bd > 0.35 * side:  # nobody, or someone else in the crop
                continue
            old = kp[f, :, 2]
            if not np.isfinite(old).any() or np.nanmean(best[:, 2]) >= np.nanmean(np.nan_to_num(old)) - 0.02:
                refined[f] = best
                improved += 1
            if progress and i % 8 == 0:
                progress(min(1.0, 0.85 + 0.15 * i / max(1, len(idx))))
    finally:
        lm.close()
        cap.release()
    if improved < ZM.accept_frac * len(idx):  # mixing two passes frame by frame makes the motion jumpy: all or nothing
        return seq
    import dataclasses
    log.info("zoom refine: %d/%d frames refined (bowler %.0f px tall)", improved, len(idx), float(np.median(hgt)))
    out = dataclasses.replace(seq, kp=refined)
    out.zoomed = (int(np.median(hgt)), round(scale, 2))
    return out

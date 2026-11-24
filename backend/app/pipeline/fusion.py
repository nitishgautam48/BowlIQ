"""Two pose estimators on the bowler: the PRIMARY (RTMPose by default) and a SECONDARY (MediaPipe) run on a crop
that follows the primary's bowler. Per frame and joint their disagreement (distance in body heights) lowers that
joint's confidence; where the primary misses a joint and the secondary sees it, the secondary fills it in.

The result keeps the PoseSequence contract, so everything downstream is unchanged; the secondary keypoints and the
disagreement are kept on the sequence (and in the diagnostics).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.errors import BowlIQError
from app.pipeline.params import P
from app.pipeline.types import J, MP_INDEX, N_JOINTS, BowlerSelection, PoseSequence, ProgressFn

log = logging.getLogger("bowliq.fusion")
R = P.perception


def body_height_px(kp: np.ndarray) -> float:
    d = np.linalg.norm(kp[:, J["nose"], :2] - 0.5 * (kp[:, J["left_ankle"], :2] + kp[:, J["right_ankle"], :2]), axis=1)
    d = d[np.isfinite(d)]
    return float(np.percentile(d, 90) * P.scale.head_top_correction) if len(d) >= 3 else float("nan")


def mediapipe_on_crops(seq: PoseSequence, video: Path, progress: Optional[ProgressFn] = None) -> np.ndarray:
    """(F, N_JOINTS, 3) MediaPipe keypoints of the bowler, from a crop that follows the primary's bowler."""
    import mediapipe as mp

    from app.pipeline.pose import open_landmarker
    kp = seq.kp
    n = kp.shape[0]
    out = np.full((n, N_JOINTS, 3), np.nan)
    have = np.where(np.isfinite(kp[:, :, 0]).any(axis=1))[0]
    if len(have) < 3:
        return out
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    hh = np.full(n, np.nan)
    for f in have:
        p = kp[f][np.isfinite(kp[f, :, 0])]
        cx[f], cy[f] = np.median(p[:, 0]), np.median(p[:, 1])
        hh[f] = p[:, 1].max() - p[:, 1].min()
    idx = np.arange(n)
    ok = np.isfinite(cx)
    cx, cy = np.interp(idx, idx[ok], cx[ok]), np.interp(idx, idx[ok], cy[ok])
    side = float(np.clip(np.nanpercentile(hh, 90) * R.crop_scale, 64, 2 * max(seq.width, seq.height)))
    size = int(R.crop_px)
    s = size / side
    cap = cv2.VideoCapture(str(video))
    lm, _ = open_landmarker(2)
    try:
        last_ts = -1
        for f in range(n):
            okf, frame = cap.read()
            if not okf:
                break
            if f < have[0] or f > have[-1]:
                continue
            x0, y0 = cx[f] - side / 2, cy[f] - side / 2
            M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
            crop = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            ts = max(int(round(f * 1000.0 / seq.fps)), last_ts + 1)
            last_ts = ts
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), ts)
            best, bd = None, np.inf
            for lms in res.pose_landmarks or []:
                arr = np.empty((N_JOINTS, 3))
                for j, mi in enumerate(MP_INDEX):
                    p = lms[mi]
                    vis = p.visibility if p.visibility is not None else 0.0
                    pres = p.presence if getattr(p, "presence", None) is not None else 1.0
                    arr[j] = (p.x * size / s + x0, p.y * size / s + y0, min(vis, pres))
                ref = kp[f, :, :2]
                d = np.nanmedian(np.linalg.norm(arr[:, :2] - ref, axis=1)) if np.isfinite(ref).any() else 0.0
                if d < bd:
                    best, bd = arr, d
            if best is not None and bd < 0.35 * side:
                out[f] = best
            if progress and f % 16 == 0:
                progress(f / max(1, n))
    finally:
        lm.close()
        cap.release()
    return out


def fuse(primary: np.ndarray, secondary: np.ndarray, bh: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """→ (fused keypoints, disagreement (F, J) in body heights, summary). Positions stay the primary's; confidence
    falls where the estimators disagree; joints the primary misses are filled from the secondary."""
    fused = primary.copy()
    d = np.linalg.norm(primary[..., :2] - secondary[..., :2], axis=-1) / max(bh, 1.0)
    vis_thr = P.smoothing.vis_threshold
    p_ok = np.isfinite(primary[..., 0]) & (np.nan_to_num(primary[..., 2]) >= vis_thr)
    s_ok = np.isfinite(secondary[..., 0]) & (np.nan_to_num(secondary[..., 2]) >= vis_thr)
    both = p_ok & s_ok & np.isfinite(d)
    k = np.clip(1 - (d - R.agree_bh) / max(R.disagree_bh - R.agree_bh, 1e-6), R.conf_floor, 1.0)
    fused[..., 2] = np.where(both, fused[..., 2] * k, fused[..., 2])
    filled = 0
    if R.fill_from_secondary:
        take = ~p_ok & s_ok
        fused[take] = secondary[take]
        fused[take, 2] = secondary[take, 2] * 0.8  # from the secondary only: a little less sure
        filled = int(take.sum())
    d = np.where(both, d, np.nan)
    summ = {"median_disagreement_bh": float(np.nanmedian(d)) if np.isfinite(d).any() else None,
            "joints_downweighted": int((both & (k < 1)).sum()), "joints_filled_from_secondary": filled,
            "frames_compared": int(both.any(axis=1).sum())}
    return fused, d, summ


class MultiModelEstimator:
    """Primary estimator → secondary on the primary's bowler → fusion (see module docstring)."""

    def __init__(self, primary, secondary: str = "mediapipe"):
        self.primary, self.secondary = primary, secondary
        self.name = getattr(primary, "name", "primary")
        self.used_fallback = False

    def estimate(self, video_path: Path, fps: float, progress: Optional[ProgressFn] = None,
                 selection: Optional[BowlerSelection] = None) -> PoseSequence:
        seq = self.primary.estimate(video_path, fps, (lambda p: progress(0.8 * p)) if progress else None, selection)
        self.used_fallback = bool(getattr(self.primary, "used_fallback", False))
        seq.engine = getattr(self.primary, "name", "")
        if self.secondary != "mediapipe" or seq.engine.startswith("mediapipe"):
            return seq
        try:
            sec = mediapipe_on_crops(seq, video_path, (lambda p: progress(0.8 + 0.2 * p)) if progress else None)
        except (BowlIQError, Exception) as exc:  # the cross-check is a help, never a reason to fail
            log.warning("secondary estimator failed: %s", exc)
            return seq
        bh = body_height_px(seq.kp)
        if not np.isfinite(bh):
            return seq
        fused, d, summ = fuse(seq.kp, sec, bh)
        log.info("fusion: %s", summ)
        seq.kp, seq.secondary, seq.disagreement = fused, sec, d
        seq.fusion = summ
        return seq

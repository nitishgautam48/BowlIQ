"""Stage 6: body scale.

Two estimates of standing height in pixels, then averaged:
  1. upright frames: nose → mid-ankle, median over the frames above the 90th percentile of the clip, × 1.07
  2. segment sum:    (shank + thigh + trunk + neck→nose), each the 75th percentile of its length over the clip, × 1.07
Segment lengths barely depend on posture, so (2) corrects the bias of (1) when the bowler is never fully upright
(run-up crouch). Their disagreement is reported as the relative scale uncertainty, which lowers confidence of
scale-dependent metrics and widens the ball-speed ± range.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.errors import BowlIQError
from app.pipeline.params import P
from app.pipeline.types import Kinematics, Scale

HEAD_TOP_CORRECTION = P.scale.head_top_correction
MIN_REL_UNCERTAINTY = P.scale.min_rel_uncertainty


def body_height_series(kin: Kinematics) -> np.ndarray:
    nose = kin.joint("nose")
    ank = kin.mid("left_ankle", "right_ankle")
    return np.linalg.norm(nose - ank, axis=1)


def _p(x: np.ndarray, q: float | None = None) -> float:
    x = x[np.isfinite(x)]
    q = P.scale.segment_pct if q is None else q
    return float(np.percentile(x, q)) if len(x) >= max(5, P.general.min_samples) else float("nan")


def segment_height(kin: Kinematics) -> float:
    seg = lambda a, b: np.linalg.norm(kin.joint(a) - kin.joint(b), axis=1)  # noqa: E731
    shank = max(_p(seg("left_knee", "left_ankle")), _p(seg("right_knee", "right_ankle")))
    thigh = max(_p(seg("left_hip", "left_knee")), _p(seg("right_hip", "right_knee")))
    trunk = _p(np.linalg.norm(kin.mid("left_shoulder", "right_shoulder") - kin.mid("left_hip", "right_hip"), axis=1))
    head = _p(np.linalg.norm(kin.joint("nose") - kin.mid("left_shoulder", "right_shoulder"), axis=1))
    total = shank + thigh + trunk + head
    return float(total * HEAD_TOP_CORRECTION) if np.isfinite(total) else float("nan")


def estimate_scale(kin: Kinematics, height_cm: Optional[float]) -> Scale:
    d = body_height_series(kin)
    idx = np.where(np.isfinite(d))[0]
    if len(idx) < P.general.min_samples:
        # head and feet never seen together (tight broadcast framing, feet cut off): fall back to the segment sum,
        # and as a last resort to the trunk length — a scale with a wide uncertainty is better than no analysis
        h_seg = segment_height(kin)
        if not (np.isfinite(h_seg) and h_seg > 0):
            trunk = _p(np.linalg.norm(kin.mid("left_shoulder", "right_shoulder") - kin.mid("left_hip", "right_hip"), axis=1))
            h_seg = trunk / P.scale.trunk_fraction if np.isfinite(trunk) and trunk > 0 else float("nan")
        if not (np.isfinite(h_seg) and h_seg > 0):
            raise BowlIQError("POSE_NOT_FOUND", "The bowler's body couldn't be measured in this clip.",
                              "Keep the whole body, head to feet, in view throughout the clip.")
        mpp = (height_cm / 100.0) / h_seg if height_cm else None
        return Scale(body_height_px=h_seg, frame=0, metres_per_pixel=mpp, method="height" if height_cm else "body_relative",
                     rel_uncertainty=P.scale.fallback_rel_uncertainty * 2)
    # BH = nose-to-ankle × 1.07, median over the upright frames (the top of this clip's own distribution)
    upright = idx[d[idx] >= np.percentile(d[idx], P.scale.upright_pct)]
    med = float(np.median(d[upright]))
    f = int(upright[np.argmin(np.abs(d[upright] - med))])
    h_upright = med * HEAD_TOP_CORRECTION
    h_seg = segment_height(kin)
    if np.isfinite(h_seg) and h_seg > 0:
        h_px = 0.5 * (h_upright + h_seg)
        rel = max(MIN_REL_UNCERTAINTY, abs(h_upright - h_seg) / (h_upright + h_seg))
    else:
        h_px, rel = h_upright, P.scale.fallback_rel_uncertainty
    mpp = (height_cm / 100.0) / h_px if height_cm else None
    return Scale(body_height_px=h_px, frame=f, metres_per_pixel=mpp, method="height" if height_cm else "body_relative",
                 rel_uncertainty=round(rel, 4))

"""Phase 4: true trunk rotation from the two synced clips.

A single camera only sees rotation flattened into its picture. With two synced views at roughly right angles, the
side-on clip gives the forward/back (along the pitch, X) component of the shoulder and hip lines and the front-on or
behind clip gives the sideways (across the pitch, Z) component, frame by frame. Their angle in the horizontal plane
is the real transverse rotation:

    θ(t) = atan2(Z, X) of the line from the bowling-side joint to the front-side joint

    hip–shoulder separation at BFC = |θ_shoulders − θ_hips|
    shoulder counter-rotation      = largest |θ_shoulders(t) − θ_shoulders(BFC)| between BFC and FFC

Both are differences of angles, so a constant error in how square the second camera is to the pitch cancels out.
Each view is scaled by the bowler's body height in that frame (the second clip's bowler changes size as he runs to or
from the camera); the cameras are treated as far away compared with the body's depth (orthographic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.pipeline.types import DeliveryEvents, J, Kinematics

MAX_JUMP_DEG = 50.0  # θ can't change this much between two consecutive frames: a left/right label swap in a view
MIN_LINE = 0.08  # line length (body heights) in the horizontal plane below which its direction is meaningless


MAX_YAW_DEG = 25.0  # a second camera further off the line of the run than this is "diagonal", not corrected


@dataclass
class Rotation3D:
    shoulders: np.ndarray  # θ per side-clip frame (deg), NaN where unavailable
    hips: np.ndarray
    reason: Optional[str]  # why it can't be used, if so
    yaw_deg: Optional[float] = None  # how far the second camera is from square to the pitch (estimated, corrected)


def estimate_yaw(side: Kinematics, second: Kinematics, offset_s: float, bfc: int, bh_s: np.ndarray, bh_f: np.ndarray
                 ) -> Optional[float]:
    """How far the second camera is turned from the line of the run, from the run-up.

    The bowler runs straight along the pitch. A camera square to the pitch sees no sideways movement of the hips;
    one turned by `a` sees sideways movement = forward movement × sin(a). Forward movement comes from the side-on
    clip, both in body heights."""
    a = max(0, bfc - int(round(0.5 * side.fps)))
    X, L = [], []
    for f in range(a, bfc + 1):
        j = int(round((f / side.fps + offset_s) * second.fps))
        if not (0 <= j < second.n):
            continue
        xs = side.mid("left_hip", "right_hip")[f, 0]
        xf = second.mid("left_hip", "right_hip")[j, 0]
        if np.isfinite(xs) and np.isfinite(xf) and np.isfinite(bh_s[f]) and np.isfinite(bh_f[j]):
            X.append(xs / bh_s[f])
            L.append((xf - second.width / 2.0) / bh_f[j])
    if len(X) < max(6, int(0.2 * side.fps)) or np.ptp(X) < 0.5:  # need ≥ half a body height of forward travel
        return None
    slope = float(np.polyfit(X, L, 1)[0])
    if abs(slope) >= np.sin(np.radians(MAX_YAW_DEG)):
        return None
    return float(np.degrees(np.arcsin(slope)))


def body_height(kin: Kinematics) -> np.ndarray:
    def seg(a, b):
        return np.linalg.norm(kin.xy[:, J[a]] - kin.xy[:, J[b]], axis=1)
    leg = np.fmax(seg("left_hip", "left_knee") + seg("left_knee", "left_ankle"),
                  seg("right_hip", "right_knee") + seg("right_knee", "right_ankle"))
    h = leg + np.linalg.norm(kin.mid("left_shoulder", "right_shoulder") - kin.mid("left_hip", "right_hip"), axis=1)
    ok = np.isfinite(h)
    if ok.sum() < 3:
        return np.full(kin.n, np.nan)
    idx = np.arange(kin.n)
    h = np.interp(idx, idx[ok], h[ok])
    k = max(3, int(0.15 * kin.fps) | 1)
    pad = np.pad(h, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(kin.n)])


def lateral_sign(kin: Kinematics, frames: list[int]) -> float:
    """Image-x direction towards the bowler's front (non-bowling) side in the second clip, from the hip labels."""
    arm = kin.bowling_arm
    fr = "left" if arm == "right" else "right"
    dx = kin.xy[frames, J[f"{fr}_hip"], 0] - kin.xy[frames, J[f"{arm}_hip"], 0]
    dx = dx[np.isfinite(dx)]
    return float(np.sign(np.median(dx))) if len(dx) else 0.0


def reconstruct(side: Kinematics, second: Kinematics, offset_s: float, ev: DeliveryEvents) -> Rotation3D:
    """θ of the shoulder and hip lines in the horizontal plane, per side-clip frame, over BFC−0.1 s … release."""
    n = side.n
    sh_theta = np.full(n, np.nan)
    hip_theta = np.full(n, np.nan)
    arm = side.bowling_arm
    fr = "left" if arm == "right" else "right"
    a = max(0, ev.bfc.frame - int(round(0.1 * side.fps)))
    b = min(n - 1, ev.release.frame)
    jf = [(f / side.fps + offset_s) * second.fps for f in range(a, b + 1)]  # fractional second-clip frames
    js = [int(round(x)) for x in jf]
    if jf[0] < 0 or jf[-1] > second.n - 1:
        return Rotation3D(sh_theta, hip_theta, "The second clip doesn't cover back foot contact to release.")
    s_sign = lateral_sign(second, js)
    if s_sign == 0:
        return Rotation3D(sh_theta, hip_theta, "The bowler's left and right can't be told apart in the second clip.")
    bh_s, bh_f = body_height(side), body_height(second)
    # second-camera image x = lateral·cos(a) + forward·sin(a) (times the sign towards the front side); undo the turn
    yaw = estimate_yaw(side, second, offset_s, ev.bfc.frame, bh_s, bh_f)
    sin_a, cos_a = (np.sin(np.radians(yaw)), np.cos(np.radians(yaw))) if yaw is not None else (0.0, 1.0)

    def at(joint: int, x: float) -> Optional[float]:
        """Second-clip image x of a joint at fractional frame x (linear between frames: the second phone can film at
        a lower rate, and the trunk turns ~200°/s, so snapping to the nearest frame would cost several degrees)."""
        i0 = int(np.floor(x))
        i1 = min(i0 + 1, second.n - 1)
        w = x - i0
        if not second.valid[i0, joint] or (w > 0 and not second.valid[i1, joint]):
            return None
        return float((1 - w) * second.xy[i0, joint, 0] + w * second.xy[i1, joint, 0])

    for f, x in zip(range(a, b + 1), jf, strict=True):
        j = int(round(x))
        for p, out in (("shoulder", sh_theta), ("hip", hip_theta)):
            jb, jfr = J[f"{arm}_{p}"], J[f"{fr}_{p}"]
            if not (side.valid[f, jb] and side.valid[f, jfr]):
                continue
            zb, zf = at(jb, x), at(jfr, x)
            if zb is None or zf is None:
                continue
            X = (side.xy[f, jfr, 0] - side.xy[f, jb, 0]) / bh_s[f]
            Z = ((zf - zb) / bh_f[j] - X * sin_a) / cos_a * s_sign
            if np.hypot(X, Z) < MIN_LINE:
                continue
            out[f] = np.degrees(np.arctan2(Z, X))
    for th in (sh_theta, hip_theta):
        seg = th[a:b + 1]
        ok = np.isfinite(seg)
        if ok.sum() >= 2:
            d = np.abs(np.diff(np.unwrap(np.radians(seg[ok]))))
            if np.degrees(d).max() > MAX_JUMP_DEG:
                return Rotation3D(sh_theta, hip_theta, "The shoulder or hip labels flip between frames in one of the "
                                  "views, so the 3D rotation can't be followed.")
    return Rotation3D(sh_theta, hip_theta, None, yaw)


def fold(d: float) -> float:
    """Angle between two undirected lines, 0–90°."""
    d = abs((d + 180.0) % 360.0 - 180.0)
    return min(d, 180.0 - d)


def separation_at(rot: Rotation3D, f: int) -> Optional[float]:
    if not (np.isfinite(rot.shoulders[f]) and np.isfinite(rot.hips[f])):
        return None
    return fold(rot.shoulders[f] - rot.hips[f])


def counter_rotation(rot: Rotation3D, bfc: int, ffc: int) -> Optional[tuple[float, int]]:
    seg = rot.shoulders[bfc:ffc + 1]
    if not np.isfinite(seg[0]) or np.isfinite(seg).sum() < max(3, 0.7 * len(seg)):
        return None
    dev = [fold(v - seg[0]) if np.isfinite(v) else -1.0 for v in seg]
    i = int(np.argmax(dev))
    return float(dev[i]), bfc + i

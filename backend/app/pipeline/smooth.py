"""Stage 4: gap filling, Savitzky–Golay smoothing, bowling arm + direction detection, mirroring."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from app.pipeline.params import P, frames, odd
from app.pipeline.types import LEG_PARTS, J, Kinematics, PoseSequence

S = P.smoothing
VIS_THRESHOLD = S.vis_threshold


def sg_window(fps: float, ms: float) -> int:
    """Savitzky–Golay window (odd) covering `ms` milliseconds, at least smoothing.min_window samples."""
    return max(S.min_window, odd(frames(ms / 1000.0, fps)))


def fill_gaps(series: np.ndarray, ok: np.ndarray, max_gap: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation over runs of not-ok samples of length ≤ max_gap bounded by ok samples on both sides."""
    out = series.astype(float).copy()
    valid = ok.copy()
    out[~ok] = np.nan
    n = len(series)
    i = 0
    while i < n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j < n and not ok[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            t = np.arange(1, j - i + 1) / (j - i + 1)
            out[i:j] = out[i - 1] + (out[j] - out[i - 1]) * t
            valid[i:j] = True
        i = j
    return out, valid


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        segs.append((i, j))
        i = j
    return segs


def savgol_masked(x: np.ndarray, valid: np.ndarray, window: int, poly: int) -> np.ndarray:
    out = x.copy()
    for a, b in _segments(valid):
        seg_len = b - a
        if seg_len >= window:
            out[a:b] = savgol_filter(x[a:b], window, poly, mode="interp")
        elif seg_len >= 5:
            w = seg_len if seg_len % 2 == 1 else seg_len - 1
            out[a:b] = savgol_filter(x[a:b], w, min(poly, w - 1), mode="interp")
    return out


# (parts compared for continuity, parts swapped together)
SWAP_GROUPS = [(("knee", "ankle"), LEG_PARTS), (("elbow", "wrist"), ("elbow", "wrist"))]


def _swap(kp: np.ndarray, z: np.ndarray | None, f: int | slice, parts: tuple[str, ...]) -> None:
    L = [J[f"left_{p}"] for p in parts]
    R = [J[f"right_{p}"] for p in parts]
    tmp = kp[f, L].copy()
    kp[f, L] = kp[f, R]
    kp[f, R] = tmp
    if z is not None:
        tz = z[f, L].copy()
        z[f, L] = z[f, R]
        z[f, R] = tz


def fix_left_right_swaps(kp: np.ndarray, z: np.ndarray | None = None, fps: float = 30.0) -> tuple[np.ndarray, int]:
    """Undo frame-level left/right label swaps of the legs and arms (common for side-on views).

    Each limb is predicted from its last two corrected positions (constant velocity); when assigning the detections the
    other way round fits those predictions much better, the labels are swapped back (a whole leg = knee, ankle, heel and
    toe together). `z` (depth), if given, is swapped in place alongside.
    """
    kp = kp.copy()
    n = kp.shape[0]
    swaps = 0
    predict_n, hold_n = frames(S.swap_predict_s, fps), frames(S.swap_hold_s, fps)
    for (a, b), parts in SWAP_GROUPS:
        L = [J[f"left_{a}"], J[f"left_{b}"]]
        R = [J[f"right_{a}"], J[f"right_{b}"]]
        hist: list[tuple[int, np.ndarray, np.ndarray]] = []  # (frame, left xy (2,2), right xy (2,2))
        for f in range(n):
            lp, rp = kp[f, L, :2], kp[f, R, :2]
            ok = np.isfinite(lp).all() and np.isfinite(rp).all()
            if not ok:
                continue
            if len(hist) >= 1:
                f1, l1, r1 = hist[-1]
                if len(hist) >= 2 and f - f1 <= predict_n:
                    f0, l0, r0 = hist[-2]
                    k = (f - f1) / max(1, f1 - f0)
                    pl, pr = l1 + (l1 - l0) * k, r1 + (r1 - r0) * k
                else:
                    pl, pr = l1, r1
                if f - f1 <= hold_n:
                    keep = np.linalg.norm(lp - pl, axis=1).sum() + np.linalg.norm(rp - pr, axis=1).sum()
                    swap = np.linalg.norm(lp - pr, axis=1).sum() + np.linalg.norm(rp - pl, axis=1).sum()
                    if swap < S.swap_ratio * keep:
                        _swap(kp, z, f, parts)
                        lp, rp = rp, lp
                        swaps += 1
            hist.append((f, lp.copy(), rp.copy()))
    return kp, swaps


def detect_bowling_arm(kp: np.ndarray) -> str:
    """The bowling arm: the wrist that rises furthest above the nose at any point in the clip.

    Height is measured relative to the nose, so the bowling wrist at release (head dipping forward, arm vertical) beats
    the front arm reaching up in the coil. The second-highest frame is used so one mislabelled frame can't decide it.
    The whole clip is searched because release isn't always in the second half (long follow-throughs).
    """
    nose_y = kp[:, J["nose"], 1]
    score = {}
    for side in ("left", "right"):
        w = kp[:, J[f"{side}_wrist"]]
        ok = np.isfinite(w[:, 1]) & np.isfinite(nose_y) & (w[:, 2] >= VIS_THRESHOLD)
        above = np.sort(nose_y[ok] - w[ok, 1])[::-1]  # image y grows downwards
        score[side] = float(above[1]) if len(above) >= 2 else -np.inf
    return "left" if score["left"] > score["right"] else "right"


def detect_direction(kp: np.ndarray) -> str:
    hip_x = 0.5 * (kp[:, J["left_hip"], 0] + kp[:, J["right_hip"], 0])
    ok = np.isfinite(hip_x)
    xs = hip_x[ok]
    if len(xs) < 4:
        return "left_to_right"
    k = max(2, len(xs) // 5)
    return "left_to_right" if np.median(xs[-k:]) >= np.median(xs[:k]) else "right_to_left"


def smooth_pose(pose: PoseSequence, bowling_arm: str | None = None, direction: str | None = None) -> Kinematics:
    fps = pose.fps
    from app.pipeline.fusion import body_height_px
    from app.pipeline.identity import correct_swaps, suppress_duplicates
    bh0 = body_height_px(pose.kp)
    if np.isfinite(bh0):
        kp, dup_log = suppress_duplicates(pose.kp, bh0)
        kp, id_log = correct_swaps(kp, fps, bh0)
        id_log = id_log + [{**d, "limb": d["joint"], "corrected": True} for d in dup_log]
    else:
        kp, id_log = fix_left_right_swaps(pose.kp, fps=fps)[0], []
    n = kp.shape[0]
    max_gap = frames(S.max_gap_s, fps)
    arm = bowling_arm or detect_bowling_arm(kp)
    direction = direction or detect_direction(kp)
    body_w = sg_window(fps, S.body_window_s * 1000)
    arm_w = sg_window(fps, S.arm_window_s * 1000)
    arm_joints = {J[f"{arm}_elbow"], J[f"{arm}_wrist"]}
    nj = kp.shape[1]
    img = np.full((n, nj, 2), np.nan)
    vis = np.nan_to_num(kp[:, :, 2], nan=0.0)
    valid = np.zeros((n, nj), bool)
    interp_count = 0
    for j in range(nj):
        ok = np.isfinite(kp[:, j, 0]) & (vis[:, j] >= VIS_THRESHOLD)
        win, poly = (arm_w, 3) if j in arm_joints else (body_w, 2)
        for c in range(2):
            filled, v = fill_gaps(kp[:, j, c], ok, max_gap)
            img[:, j, c] = savgol_masked(filled, v, win, poly)
            valid[:, j] = v
        interp_count += int((valid[:, j] & ~ok).sum())
    img[~valid] = np.nan
    xy = img.copy()
    cam = getattr(pose, "cam", None)
    if cam is not None:  # handheld / following camera: measure in one fixed reference view (Phase 1c)
        from app.pipeline.adapt import apply_path
        xy = apply_path(img, cam)
    if direction == "right_to_left":
        xy[..., 0] = pose.width - xy[..., 0]
    xy[..., 1] = pose.height - xy[..., 1]
    hn = getattr(pose, "head_neck", None)
    return Kinematics(fps=fps, width=pose.width, height=pose.height, xy=xy, vis=vis, valid=valid, img_xy=img,
                      bowling_arm=arm, direction=direction, interpolated_frac=interp_count / max(1, n * nj),
                      identity_log=id_log, head_neck=None if hn is None else hn[..., :2])


def swap_legs(kin: Kinematics) -> Kinematics:
    """Return kinematics with left/right leg labels (knee, ankle, heel, toe) swapped for the whole clip."""
    import dataclasses

    xy, img, vis, valid = kin.xy.copy(), kin.img_xy.copy(), kin.vis.copy(), kin.valid.copy()
    for a in LEG_PARTS:
        li, ri = J[f"left_{a}"], J[f"right_{a}"]
        for arr in (xy, img, vis, valid):
            arr[:, [li, ri]] = arr[:, [ri, li]]
    return dataclasses.replace(kin, xy=xy, img_xy=img, vis=vis, valid=valid)


def legs_consistent(kin: Kinematics, ffc: int) -> bool | None:
    """At front foot contact the front ankle must be ahead (+x, mirrored frame) of the back ankle."""
    f = kin.joint(kin.side("ankle", "front"))[ffc]
    b = kin.joint(kin.side("ankle", "back"))[ffc]
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(b))):
        return None
    return bool(f[0] > b[0])

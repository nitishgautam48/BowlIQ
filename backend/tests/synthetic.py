"""Synthetic bowler kinematics with known event frames (used by the tests and the sample-clip generator)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.types import N_JOINTS, J, PoseSequence


@dataclass
class SynthTruth:
    bfc: int
    ffc: int
    release: int
    px_per_m: float
    height_m: float


def _foot_track(n: int, fps: float, plants: list[tuple[int, int, float]], lift: float) -> np.ndarray:
    """plants: (on_frame, off_frame, x_m). Linear flight with a sinusoidal lift between plants. Returns (n, 2) metres."""
    pos = np.zeros((n, 2))
    plants = sorted(plants)
    for f in range(n):
        placed = False
        for on, off, x in plants:
            if on <= f <= off:
                pos[f] = (x, 0.0)
                placed = True
                break
        if placed:
            continue
        prev = [p for p in plants if p[1] < f]
        nxt = [p for p in plants if p[0] > f]
        if prev and nxt:
            (_, off, x0), (on, _, x1) = prev[-1], nxt[0]
            t = (f - off) / (on - off)
            pos[f] = (x0 + (x1 - x0) * t, lift * np.sin(np.pi * t))
        elif prev:  # after last plant: swing forward
            _, off, x0 = prev[-1]
            dt = (f - off) / fps
            pos[f] = (x0 + 4.0 * dt, lift * min(1.0, dt * 6))
        else:
            on, _, x1 = nxt[0]
            dt = (on - f) / fps
            pos[f] = (x1 - 4.0 * dt, lift * min(1.0, dt * 6))
    return pos


def synth_pose(direction: str = "left_to_right", fps: float = 120.0, n: int = 132, bfc: int = 62, ffc: int = 82,
               release: int = 94, width: int = 1280, height: int = 720, px_per_m: float = 260.0,
               height_m: float = 1.80, arm: str = "right") -> tuple[PoseSequence, SynthTruth]:
    """Raw pose (image coords, y down) for a running bowler with known BFC/FFC/release frames."""
    back_side = arm
    front_side = "left" if arm == "right" else "right"
    run_v = 5.0  # m/s
    t = np.arange(n) / fps
    # hip: constant run-up speed, decelerating after FFC
    hip_x = np.where(np.arange(n) <= ffc, run_v * t, run_v * ffc / fps + 1.8 * (t - ffc / fps))
    step = int(0.24 * fps)
    # back foot plants: earlier run-up strides, then BFC (planted until just after FFC)
    back_plants = [(bfc - 2 * step - 6, bfc - 2 * step + 4, hip_x[bfc - 2 * step] + 0.1),
                   (bfc, ffc + 8, hip_x[bfc] + 0.15)]
    front_plants = [(bfc - step - 6, bfc - step + 4, hip_x[bfc - step] + 0.1),
                    (ffc, n + 10, hip_x[bfc] + 0.15 + 1.45)]
    back = _foot_track(n, fps, back_plants, 0.18)
    front = _foot_track(n, fps, front_plants, 0.22)
    hip_y = 0.95 + 0.03 * np.sin(2 * np.pi * t / 0.24)
    hip = np.stack([hip_x, hip_y], 1)
    trunk_lean = np.clip((np.arange(n) - ffc) / (release - ffc + 1e-9), 0, 1) * np.radians(30)
    neck = hip + np.stack([0.55 * np.sin(trunk_lean), 0.55 * np.cos(trunk_lean)], 1)
    nose = neck + np.stack([0.08 + 0.2 * np.sin(trunk_lean), 0.2 * np.ones(n)], 1)
    # bowling wrist: low until FFC, Gaussian peak exactly at release
    peak = np.exp(-0.5 * ((np.arange(n) - release) / 6.0) ** 2)
    wr = neck + np.stack([0.1 - 0.4 * peak + 0.2, -0.45 + 1.1 * peak], 1)
    nb_wr = neck + np.stack([0.3 * np.ones(n), -0.35 * np.ones(n)], 1)
    kp_m = np.zeros((n, N_JOINTS, 2))
    kp_m[:, J["nose"]] = nose
    for s in ("left", "right"):
        kp_m[:, J[f"{s}_eye"]] = nose + [0.02, 0.03]
        kp_m[:, J[f"{s}_ear"]] = nose + [-0.06, 0.02]
        kp_m[:, J[f"{s}_shoulder"]] = neck + [0.0 if s == arm else 0.03, -0.03]
        kp_m[:, J[f"{s}_hip"]] = hip + [0.0 if s == arm else 0.02, 0.0]
    kp_m[:, J[f"{arm}_wrist"]] = wr
    kp_m[:, J[f"{arm}_elbow"]] = 0.5 * (neck + wr)
    other = front_side
    kp_m[:, J[f"{other}_wrist"]] = nb_wr
    kp_m[:, J[f"{other}_elbow"]] = 0.5 * (neck + nb_wr)
    for s, foot in ((back_side, back), (front_side, front)):
        kp_m[:, J[f"{s}_ankle"]] = foot + [0.0, 0.08]
        kp_m[:, J[f"{s}_heel"]] = foot + [-0.06, 0.03]  # heel and toe move rigidly with the planted foot
        kp_m[:, J[f"{s}_foot_index"]] = foot + [0.16, 0.01]
        mid = 0.5 * (kp_m[:, J[f"{s}_hip"]] + kp_m[:, J[f"{s}_ankle"]])
        kp_m[:, J[f"{s}_knee"]] = mid + [0.08, 0.0]
    # metres → pixels, start 1.5 m from the left edge, ground at y = height - 60
    x_px = 80 + kp_m[..., 0] * px_per_m
    y_px = (height - 60) - kp_m[..., 1] * px_per_m
    if direction == "right_to_left":
        x_px = width - x_px
    kp = np.dstack([x_px, y_px, np.full(x_px.shape, 0.95)])
    return (PoseSequence(fps=fps, width=width, height=height, kp=kp, device="cpu", detection_rate=1.0),
            SynthTruth(bfc=bfc, ffc=ffc, release=release, px_per_m=px_per_m, height_m=height_m))

"""Stage 8: ball speed from the ball itself (side-on clip, needs metres-per-pixel from the bowler's height).

After release, the ball is tracked for up to 20 frames: MOG2 background subtraction, a small-round-blob filter sized
from the real ball diameter, and nearest-to-predicted-path association. Speed = slope of a straight-line fit of
position vs time. Returns not_measured (never a guess) when unscaled or the ball can't be followed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.pipeline.metrics import TILT_LOW_DEG
from app.pipeline.params import P, frames

B = P.ball
BALL_DIAMETER_M = B.diameter_m
MIN_POINTS = B.min_points
MAX_SPEED_MS = B.max_speed_ms  # upper bound for the association gate
PLAUSIBLE_KMH = tuple(B.plausible_kmh)


@dataclass
class BallSpeedResult:
    value_kmh: Optional[float]
    plus_minus: Optional[float]
    reason: Optional[str]
    frames_tracked: int
    track: list[tuple[int, float, float]]


def _none(reason: str, n: int = 0) -> BallSpeedResult:
    return BallSpeedResult(None, None, reason, n, [])


def measure_ball_speed(video_path: str, fps: float, release_frame: int, wrist_img: np.ndarray,
                       person_boxes: np.ndarray, metres_per_pixel: Optional[float], direction: str,
                       camera_motion: Optional[float], ground_tilt_deg: Optional[float] = None,
                       scale_rel_uncertainty: float = 0.0) -> BallSpeedResult:
    if not metres_per_pixel:
        return _none("Add your height to measure ball speed (it sets the scale in metres).")
    if camera_motion is not None and camera_motion >= P.camera.moving_wps:
        return _none("The camera moves, so the ball's path can't be measured. Use a tripod.")
    if ground_tilt_deg is not None and ground_tilt_deg > TILT_LOW_DEG:
        return _none(f"The run isn't square to the camera (ground line tilted {ground_tilt_deg:.0f}°), so the ball's "
                     "speed along the pitch would be underestimated. Film side-on, square to the crease.")
    wx, wy = wrist_img[release_frame] if release_frame < len(wrist_img) else (np.nan, np.nan)
    if not (np.isfinite(wx) and np.isfinite(wy)):
        return _none("The bowling hand isn't visible at release.")
    d_px = BALL_DIAMETER_M / metres_per_pixel
    if d_px < B.min_diameter_px:
        return _none(f"The ball is only {d_px:.1f} px wide at this distance; film closer or at a higher resolution.")
    area_exp = np.pi * (d_px / 2) ** 2
    a_min, a_max = max(4.0, 0.25 * area_exp), 5.0 * area_exp + 20
    step_max = MAX_SPEED_MS / fps / metres_per_pixel  # px per frame at 180 km/h
    sign = 1.0 if direction == "left_to_right" else -1.0

    cap = cv2.VideoCapture(video_path)
    warm = max(0, release_frame - int(0.5 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, warm)
    mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=False)
    f = warm
    while f <= release_frame:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return _none("The clip ends at release, so the ball can't be followed.")
        mog.apply(frame, learningRate=-1)
        f += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    track: list[tuple[int, float, float]] = []
    last = np.array([wx, wy], float)
    vel: Optional[np.ndarray] = None
    misses = 0
    for k in range(1, frames(B.track_s, fps) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        fi = release_frame + k
        fg = mog.apply(frame, learningRate=0.002)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gap = k - (track[-1][0] - release_frame) if track else k
        pred = last + (vel * gap if vel is not None else 0)
        gate = step_max * gap * 1.3 + 2 * d_px if vel is None else max(3 * d_px, 0.35 * np.linalg.norm(vel) * gap + 2 * d_px)
        box = person_boxes[fi] if fi < len(person_boxes) else None
        best, bd = None, np.inf
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (a_min <= area <= a_max):
                continue
            per = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (per * per) if per > 0 else 0
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            aspect = max(rw, rh) / max(1e-6, min(rw, rh))
            # a fast ball is a short streak (motion blur), so allow elongation up to ~3.5:1 but not thin limbs
            if circ < 0.3 or aspect > 3.5 or min(rw, rh) < 0.5 * d_px:
                continue
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            p = np.array([cx, cy])
            if vel is None and (p[0] - wx) * sign < -d_px:  # first point must move towards the batter
                continue
            inside = box is not None and bool(np.all(np.isfinite(box))) and box[0] < cx < box[2] and box[1] < cy < box[3]
            if inside and k > 3:  # by now the ball has left the bowler; blobs on the body are limbs
                continue
            d = np.linalg.norm(p - pred)
            if d < gate and d < bd:
                best, bd = p, d
        if best is None:
            misses += 1
            if misses > 3 and track:
                break
            continue
        misses = 0
        if track:
            prev_f, px, py = track[-1]
            v_now = (best - np.array([px, py])) / (fi - prev_f)
            vel = v_now if vel is None else 0.5 * vel + 0.5 * v_now
        track.append((fi, float(best[0]), float(best[1])))
        last = best
    cap.release()

    if len(track) < MIN_POINTS:
        return _none(f"The ball was found in only {len(track)} frames after release (need {MIN_POINTS}). "
                     "Film against a plain background with good light.", len(track))
    t = np.array([(fi - release_frame) / fps for fi, _, _ in track])
    xs = np.array([x for _, x, _ in track])
    ys = np.array([y for _, _, y in track])
    A = np.vstack([t, np.ones_like(t)]).T
    (vx, x0), *_ = np.linalg.lstsq(A, xs, rcond=None)
    (vy, y0), *_ = np.linalg.lstsq(A, ys, rcond=None)
    speed_px = float(np.hypot(vx, vy))
    kmh = speed_px * metres_per_pixel * 3.6
    n = len(t)
    resid = np.hypot(xs - (vx * t + x0), ys - (vy * t + y0))
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if rms > 3 * d_px:
        return _none("The detected ball positions don't form a straight path; the track is unreliable.", n)
    sxx = float(np.sum((t - t.mean()) ** 2))
    se_px = rms / np.sqrt(sxx) if sxx > 0 and n > 2 else speed_px
    span = t[-1] - t[0]
    quant_px = 2.0 / span if span > 0 else speed_px  # ±1 px at each end of the track
    pm_track = (1.96 * se_px + quant_px) * metres_per_pixel * 3.6
    if fps < 50:
        pm_track *= 1.5
    # the body-height scale error multiplies the whole speed, so it adds in quadrature (95 % interval)
    pm = float(np.hypot(pm_track, 1.96 * scale_rel_uncertainty * kmh))
    if not (PLAUSIBLE_KMH[0] <= kmh <= PLAUSIBLE_KMH[1]):
        return _none(f"The tracked object moved at {kmh:.0f} km/h, outside a plausible bowling speed, so it was rejected.", n)
    return BallSpeedResult(round(kmh, 1), round(float(pm), 1), None, n, track)

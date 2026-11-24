"""Generate backend/samples/sample_delivery.mp4: a synthetic side-on fast-bowling clip drawn with OpenCV.

The figure is a shaded humanoid (1.80 m tall at 260 px/m) driven by the same synthetic kinematics used in the tests,
with known BFC/FFC/release frames, a ball released at ~118 km/h, filmed at 120 fps, 1280×720. It exists so the full
pipeline can be exercised without a real clip. Replace it with a real Creative Commons clip for meaningful results
(see README → "Sample clip").

Usage:  python scripts/make_sample.py [--out samples/sample_delivery.mp4]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline.types import J  # noqa: E402
from tests.synthetic import synth_pose  # noqa: E402

SKIN = (120, 160, 205)
SHIRT = (235, 235, 238)
TROUSERS = (225, 228, 230)
SHOE = (40, 40, 45)


def capsule(img, a, b, r, color):
    a = tuple(int(v) for v in a)
    b = tuple(int(v) for v in b)
    cv2.line(img, a, b, color, int(2 * r), cv2.LINE_AA)
    cv2.circle(img, a, int(r), color, -1, cv2.LINE_AA)
    cv2.circle(img, b, int(r), color, -1, cv2.LINE_AA)
    shade = tuple(max(0, c - 40) for c in color)
    cv2.line(img, a, b, shade, 2, cv2.LINE_AA)


def background(w, h):
    bg = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        t = y / h
        bg[y] = (int(200 - 60 * t), int(170 - 30 * t), int(140 - 50 * t)) if y < h * 0.55 else (60, int(130 - 30 * t), 70)
    cv2.rectangle(bg, (0, int(h * 0.55)), (w, int(h * 0.58)), (90, 150, 170), -1)
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 12, (h, w, 1), dtype=np.uint8)
    return cv2.add(bg, np.repeat(noise, 3, axis=2))


def draw_person(img, P, s):
    g = lambda n: P[J[n]]  # noqa: E731
    nose = g("nose")
    ls, rs, lh, rh = g("left_shoulder"), g("right_shoulder"), g("left_hip"), g("right_hip")
    neck = 0.5 * (ls + rs)
    hip = 0.5 * (lh + rh)
    # far-side limbs first (left side for a right-arm bowler seen from his right)
    capsule(img, g("left_hip"), g("left_knee"), 0.075 * s, (200, 203, 206))
    capsule(img, g("left_knee"), g("left_ankle"), 0.06 * s, (200, 203, 206))
    capsule(img, g("left_ankle"), g("left_ankle") + [0.13 * s, 0.03 * s], 0.035 * s, SHOE)
    capsule(img, g("left_shoulder"), g("left_elbow"), 0.05 * s, (215, 215, 218))
    capsule(img, g("left_elbow"), g("left_wrist"), 0.04 * s, SKIN)
    # torso
    perp = np.array([-(neck - hip)[1], (neck - hip)[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-6) * 0.13 * s
    poly = np.array([neck + perp, neck - perp, hip - perp * 0.85, hip + perp * 0.85]).astype(np.int32)
    cv2.fillConvexPoly(img, poly, SHIRT, cv2.LINE_AA)
    cv2.polylines(img, [poly], True, (180, 180, 185), 2, cv2.LINE_AA)
    capsule(img, neck, nose + [0, 0.1 * s], 0.045 * s, SKIN)
    # head with face
    head = nose + np.array([-0.04 * s, -0.02 * s])
    cv2.circle(img, tuple(head.astype(int)), int(0.11 * s), SKIN, -1, cv2.LINE_AA)
    cv2.ellipse(img, tuple((head + [-0.02 * s, -0.06 * s]).astype(int)), (int(0.11 * s), int(0.06 * s)), 0, 180, 360, (30, 30, 35), -1, cv2.LINE_AA)
    eye = head + np.array([0.06 * s, -0.01 * s])
    cv2.circle(img, tuple(eye.astype(int)), max(2, int(0.012 * s)), (30, 30, 30), -1, cv2.LINE_AA)
    cv2.circle(img, tuple((head + [0.11 * s, 0.02 * s]).astype(int)), max(2, int(0.02 * s)), (100, 135, 185), -1, cv2.LINE_AA)
    cv2.line(img, tuple((head + [0.05 * s, 0.06 * s]).astype(int)), tuple((head + [0.09 * s, 0.06 * s]).astype(int)), (60, 60, 120), 2, cv2.LINE_AA)
    cv2.circle(img, tuple((head + [-0.02 * s, 0.0]).astype(int)), max(3, int(0.025 * s)), (100, 135, 180), -1, cv2.LINE_AA)
    # near-side limbs
    capsule(img, g("right_hip"), g("right_knee"), 0.08 * s, TROUSERS)
    capsule(img, g("right_knee"), g("right_ankle"), 0.065 * s, TROUSERS)
    capsule(img, g("right_ankle"), g("right_ankle") + [0.13 * s, 0.03 * s], 0.035 * s, SHOE)
    capsule(img, g("right_shoulder"), g("right_elbow"), 0.052 * s, SHIRT)
    capsule(img, g("right_elbow"), g("right_wrist"), 0.042 * s, SKIN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "samples" / "sample_delivery.mp4"))
    args = ap.parse_args()
    fps, w, h = 120.0, 1280, 720
    pose, truth = synth_pose("left_to_right", fps=fps, n=150, bfc=70, ffc=92, release=104, width=w, height=h, px_per_m=200.0)
    kp = pose.kp[..., :2]
    s = truth.px_per_m
    bg = background(w, h)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ball_v = 118 / 3.6 * s / fps  # px per frame
    rel_w = kp[truth.release, J["right_wrist"]]
    for f in range(len(kp)):
        img = bg.copy()
        draw_person(img, kp[f], s)
        if f > truth.release:
            k = f - truth.release
            c = rel_w + np.array([ball_v * k, 0.25 * ball_v * k])
            if 0 <= c[0] < w and 0 <= c[1] < h:
                cv2.circle(img, tuple(c.astype(int)), int(0.036 * s), (40, 40, 200), -1, cv2.LINE_AA)
        elif f <= truth.release:
            cv2.circle(img, tuple(kp[f, J["right_wrist"]].astype(int)), int(0.036 * s), (40, 40, 200), -1, cv2.LINE_AA)
        p.stdin.write(img.tobytes())
    p.stdin.close()
    p.wait()
    print(f"Wrote {out} ({len(kp)} frames at {fps:.0f} fps). Ground truth: BFC {truth.bfc}, FFC {truth.ffc}, release {truth.release}.")


if __name__ == "__main__":
    main()

"""Annotated video + overlay frames: skeleton, knee and trunk angle arcs with degree labels, BFC/FFC/RELEASE labels,
frame counter, 0.25× slow motion around release (±0.3 s). H.264 via ffmpeg (yuv420p, faststart)."""
from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.biomech.angles import joint_angle, segment_angle_vs_vertical
from app.core.errors import BowlIQError
from app.core.ffmpeg import SAFE_INPUT, ffmpeg_path, h264_args, require_ffmpeg, run
from app.pipeline.types import SKELETON_EDGES, J, Kinematics, ProgressFn

GREEN = (118, 213, 43)
AMBER = (36, 165, 245)
BLUE = (248, 189, 56)
WHITE = (238, 237, 236)
DARK = (13, 11, 10)
SLOWMO_FACTOR = 4
SLOWMO_HALF_S = 0.3


@dataclass
class OverlayData:
    fps: float
    img_xy: np.ndarray  # (F, 17, 2) image coords
    valid: np.ndarray  # (F, 17)
    arm: str
    events: dict[str, int]
    knee_deg: np.ndarray
    trunk_deg: np.ndarray
    view: str = "side"  # "second": a front-on / behind clip → side bend, arm-from-vertical and separation instead
    front: str = ""  # front leg side from the limb roles ("" = opposite the bowling arm)

    @classmethod
    def from_kinematics(cls, kin: Kinematics, events: dict[str, int]) -> "OverlayData":
        fr = kin.side("knee", "front").split("_")[0]  # front leg ROLE (planted at FFC)
        knee = joint_angle(kin.xy[:, J[f"{fr}_hip"]], kin.xy[:, J[f"{fr}_knee"]], kin.xy[:, J[f"{fr}_ankle"]])
        trunk = segment_angle_vs_vertical(kin.mid("left_hip", "right_hip"), kin.mid("left_shoulder", "right_shoulder"))
        return cls(kin.fps, kin.img_xy, kin.valid, kin.bowling_arm, events, knee, trunk, front=fr)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, fps=self.fps, img_xy=self.img_xy, valid=self.valid, arm=self.arm,
                            ev_names=np.array(list(self.events)), ev_frames=np.array(list(self.events.values())),
                            knee=self.knee_deg, trunk=self.trunk_deg, view=self.view, front=self.front)

    @classmethod
    def load(cls, path: Path) -> "OverlayData":
        z = np.load(path, allow_pickle=False)
        ev = {str(k): int(v) for k, v in zip(z["ev_names"], z["ev_frames"], strict=True)}
        view = str(z["view"]) if "view" in z.files else "side"
        front = str(z["front"]) if "front" in z.files else ""
        return cls(float(z["fps"]), z["img_xy"], z["valid"], str(z["arm"]), ev, z["knee"], z["trunk"], view, front)


def slowmo_range(release: int, fps: float, n: int) -> tuple[int, int]:
    h = int(round(SLOWMO_HALF_S * fps))
    return max(0, release - h), min(n - 1, release + h)


def _label(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float, bg=DARK, thick: int = 1) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    pad = max(3, int(4 * scale))
    x = min(max(0, x), img.shape[1] - tw - 2 * pad)
    y = min(max(th + 2 * pad, y), img.shape[0] - 2)
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y + base), bg, -1)
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y + base), color, 1)
    cv2.putText(img, text, (x + pad, y - pad + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _arc(img: np.ndarray, c: np.ndarray, a: np.ndarray, b: np.ndarray, r: int, color, label: str, s: float) -> None:
    a1 = math.degrees(math.atan2(a[1] - c[1], a[0] - c[0]))
    a2 = math.degrees(math.atan2(b[1] - c[1], b[0] - c[0]))
    d = (a2 - a1) % 360
    if d > 180:
        a1, a2, d = a2, a1, 360 - d
    cv2.ellipse(img, (int(c[0]), int(c[1])), (r, r), 0, a1, a1 + d, color, max(2, int(2 * s)), cv2.LINE_AA)
    mid = math.radians(a1 + d / 2)
    lx = c[0] + math.cos(mid) * (r + 18 * s)
    ly = c[1] + math.sin(mid) * (r + 18 * s)
    _label(img, label, (int(lx - 18 * s), int(ly + 8 * s)), color, 0.5 * s, thick=max(1, int(s)))


def knee_angle_shown(ov: OverlayData, f: int) -> bool:
    """The front-knee angle is only meaningful from back foot contact until just after release, and only while the
    two legs are apart (when they cross, which leg is 'front' is ambiguous)."""
    bfc, rel = ov.events.get("BFC"), ov.events.get("release")
    if bfc is None or rel is None or not (bfc <= f <= rel + int(round(0.15 * ov.fps))):
        return False
    P = ov.img_xy[f]
    la, ra, lh, rh = (P[J[n]] for n in ("left_ankle", "right_ankle", "left_hip", "right_hip"))
    if not (np.all(np.isfinite(la)) and np.all(np.isfinite(ra))):
        return False
    leg = np.linalg.norm(0.5 * (lh + rh) - 0.5 * (la + ra)) if np.all(np.isfinite(lh)) and np.all(np.isfinite(rh)) else np.nan
    return not (np.isfinite(leg) and abs(la[0] - ra[0]) < 0.08 * leg)


PINK, VIOLET = (182, 114, 244), (250, 139, 167)  # BGR
MIN_LINE_WIDTH = 0.35  # shoulder/hip line shorter than this × torso = seen end-on (as in pipeline/metrics.py)


def _second_view_angles(img: np.ndarray, P: np.ndarray, V: np.ndarray, arm: str, s: float) -> None:
    """Front-on / behind clip: side bend, bowling arm from vertical (while the arm is up) and the shoulder–hip line
    separation (only when neither line is seen end-on). Same rules as the browser overlay."""
    lh, rh, ls, rs = J["left_hip"], J["right_hip"], J["left_shoulder"], J["right_shoulder"]
    if not (V[lh] and V[rh] and V[ls] and V[rs]):
        return
    mh, ms = 0.5 * (P[lh] + P[rh]), 0.5 * (P[ls] + P[rs])
    torso = float(np.linalg.norm(ms - mh))
    if torso < 5:
        return
    up = mh + np.array([0.0, -0.9 * torso])
    cv2.line(img, tuple(mh.astype(int)), tuple(ms.astype(int)), BLUE, max(1, int(2 * s)), cv2.LINE_AA)
    _arc(img, mh, up, ms, int(28 * s), BLUE, f"bend {joint_angle(up, mh, ms):.0f} deg", s)
    sh, wr = J[f"{arm}_shoulder"], J[f"{arm}_wrist"]
    if V[sh] and V[wr] and P[wr][1] < P[sh][1]:
        vup = P[sh] + np.array([0.0, -float(np.linalg.norm(P[wr] - P[sh]))])
        _arc(img, P[sh], vup, P[wr], int(22 * s), AMBER, f"arm {joint_angle(vup, P[sh], P[wr]):.0f} deg", s)
    sw, hw = float(np.linalg.norm(P[ls] - P[rs])), float(np.linalg.norm(P[lh] - P[rh]))
    if sw >= MIN_LINE_WIDTH * torso and hw >= MIN_LINE_WIDTH * torso:
        for a, b, c in ((P[ls], P[rs], PINK), (P[lh], P[rh], VIOLET)):
            d = b - a
            cv2.line(img, tuple((a - 0.25 * d).astype(int)), tuple((b + 0.25 * d).astype(int)), c, max(1, int(2 * s)), cv2.LINE_AA)
        sep = abs(np.degrees(np.arctan2(*(P[rs] - P[ls])[::-1]) - np.arctan2(*(P[rh] - P[lh])[::-1]))) % 180
        sep = min(sep, 180 - sep)
        _label(img, f"sep {sep:.0f} deg", (int(max(P[ls][0], P[rs][0]) + 14 * s), int(ms[1])), PINK, 0.55 * s)


def draw_overlay(img: np.ndarray, f: int, ov: OverlayData, skeleton: bool = True, angles: bool = True,
                 counter: bool = True, slowmo: bool = False, highlight: Optional[str] = None) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(0.8, min(w, h) / 600.0)
    if f >= len(ov.img_xy):
        return img
    P = ov.img_xy[f]
    V = ov.valid[f]
    fr = ov.front or ("left" if ov.arm == "right" else "right")
    nj = P.shape[0]  # 21 joints, or 17 for analyses made before foot points existed
    if skeleton:
        for a, b in SKELETON_EDGES:
            ia, ib = J[a], J[b]
            if ia < nj and ib < nj and V[ia] and V[ib]:
                col = AMBER if (a.startswith(ov.arm) and ("elbow" in a or "wrist" in b)) else GREEN
                cv2.line(img, tuple(P[ia].astype(int)), tuple(P[ib].astype(int)), col, max(2, int(3 * s)), cv2.LINE_AA)
        for j in range(nj):
            if V[j]:
                cv2.circle(img, tuple(P[j].astype(int)), max(3, int(4 * s)), DARK, -1, cv2.LINE_AA)
                cv2.circle(img, tuple(P[j].astype(int)), max(3, int(4 * s)), GREEN, max(1, int(2 * s)), cv2.LINE_AA)
    if angles and ov.view == "second":
        _second_view_angles(img, P, V, ov.arm, s)
    elif angles:
        hip, knee, ank = J[f"{fr}_hip"], J[f"{fr}_knee"], J[f"{fr}_ankle"]
        if knee_angle_shown(ov, f) and V[hip] and V[knee] and V[ank] and np.isfinite(ov.knee_deg[f]):
            _arc(img, P[knee], P[hip], P[ank], int(22 * s), AMBER, f"{ov.knee_deg[f]:.0f} deg", s)
        lh, rh, ls, rs = J["left_hip"], J["right_hip"], J["left_shoulder"], J["right_shoulder"]
        if V[lh] and V[rh] and V[ls] and V[rs] and np.isfinite(ov.trunk_deg[f]):
            mh = 0.5 * (P[lh] + P[rh])
            ms = 0.5 * (P[ls] + P[rs])
            up = mh + np.array([0, -60 * s])
            for k in range(0, int(60 * s), int(8 * s) or 1):
                cv2.line(img, (int(mh[0]), int(mh[1] - k)), (int(mh[0]), int(mh[1] - k - 4 * s)), BLUE, 1, cv2.LINE_AA)
            cv2.line(img, tuple(mh.astype(int)), tuple(ms.astype(int)), BLUE, max(1, int(2 * s)), cv2.LINE_AA)
            _arc(img, mh, up, ms, int(30 * s), BLUE, f"{abs(ov.trunk_deg[f]):.0f} deg", s)
    if highlight and highlight in J and V[J[highlight]]:
        p = P[J[highlight]].astype(int)
        cv2.circle(img, tuple(p), int(10 * s), AMBER, max(2, int(2 * s)), cv2.LINE_AA)
    for name, ef in ov.events.items():
        if 0 <= f - ef <= max(2, int(0.04 * ov.fps)):
            _label(img, name.upper(), (int(w / 2 - 50 * s), int(48 * s)), (4, 20, 10), 0.9 * s, bg=GREEN, thick=2)
    if counter:
        _label(img, f"F {f:03d}  {f / ov.fps:.3f} s", (int(12 * s), h - int(12 * s)), WHITE, 0.55 * s)
    if slowmo:
        _label(img, "0.25x", (w - int(80 * s), h - int(12 * s)), AMBER, 0.55 * s)
    return img


SEEK_GOP = 8  # keyframe every 8 frames in the interactive (overlay-free) video: frame-accurate seeks stay fast


class _Encoder:
    def __init__(self, out: Path, w: int, h: int, fps: float, gop: int | None = None):
        require_ffmpeg()
        self.out = out
        self.log = tempfile.TemporaryFile()  # noqa: SIM115 (closed in close())
        cmd = [ffmpeg_path(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps:.6f}", "-i", "-", "-an", *h264_args(21, gop),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        try:
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except OSError:
            self.log.close()
            raise

    def write(self, frame: np.ndarray, times: int = 1) -> None:
        b = frame.tobytes()
        for _ in range(times):
            self.p.stdin.write(b)

    def close(self, ok: bool = True) -> None:
        """Finish (ok=True) or abort (ok=False). Never masks the original error; always reaps the process."""
        try:
            if self.p.stdin and not self.p.stdin.closed:
                self.p.stdin.close()
        except BrokenPipeError:
            ok = False
        if ok:
            rc = self.p.wait(timeout=600)
        else:
            self.p.kill()
            rc = self.p.wait()
        self.log.seek(0)
        tail = self.log.read()[-600:].decode("utf-8", "replace")
        self.log.close()
        if ok and rc != 0:
            logging.getLogger("bowliq.render").error("ffmpeg encode failed: %s", tail)
            raise BowlIQError("RENDER_FAILED", "Video encoding failed.",
                              "Check that your ffmpeg build includes libx264 (the Gyan.FFmpeg build does).", 500)


def render_videos(src: Path, ov: OverlayData, annotated: Path, clean: Path, progress: Optional[ProgressFn] = None) -> tuple[int, int]:
    """Write the annotated MP4 and an overlay-free MP4 with the identical slow-motion timeline. Returns slowmo range."""
    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s0, s1 = slowmo_range(ov.events.get("release", n // 2), ov.fps, n)
    encoders: list[_Encoder] = []
    ok = False
    try:
        ea = _Encoder(annotated, w, h, ov.fps)
        encoders.append(ea)
        ec = _Encoder(clean, w, h, ov.fps, gop=SEEK_GOP)
        encoders.append(ec)
        f = 0
        while True:
            got, frame = cap.read()
            if not got:
                break
            rep = SLOWMO_FACTOR if s0 <= f <= s1 else 1
            ec.write(frame, rep)
            draw_overlay(frame, f, ov, slowmo=rep > 1)
            ea.write(frame, rep)
            f += 1
            if progress and f % 10 == 0:
                progress(min(1.0, f / max(1, n)))
        ok = True
    finally:
        cap.release()
        errors = []
        for e in encoders:
            try:
                e.close(ok)
            except BowlIQError as exc:
                errors.append(exc)
        if ok and errors:
            raise errors[0]
    return s0, s1


def encode_seekable(src: Path, dst: Path) -> None:
    """Overlay-free copy of a processed clip with a keyframe every SEEK_GOP frames (fast frame-accurate seeking)."""
    require_ffmpeg()
    run([ffmpeg_path(), "-y", "-v", "error", *SAFE_INPUT, "-i", str(src), "-an", *h264_args(21, SEEK_GOP),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])


def frame_jpeg(src: Path, n: int, ov: OverlayData, highlight: Optional[str] = None, quality: int = 88) -> Optional[bytes]:
    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 0 or n >= total:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    draw_overlay(frame, n, ov, highlight=highlight)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None

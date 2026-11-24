"""Stage 1: clip quality. Metadata checks, camera motion and view classification.

Hard failures only for unreadable files or clips under 1 s. Everything else lowers the level with a reason.
"""
from __future__ import annotations

import contextlib
import math
from fractions import Fraction
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import BowlIQError
from app.core.ffmpeg import probe
from app.pipeline.params import P
from app.pipeline.types import J, PoseSequence, QualityReport, VideoMeta

LEVELS = ["Low", "Medium", "High"]


def _rate(s: str | None) -> float:
    if not s or s in ("0/0", "0"):
        return 0.0
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict) -> int:
    rot = 0
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        with contextlib.suppress(ValueError):
            rot = int(float(tags["rotate"]))
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            with contextlib.suppress(TypeError, ValueError):
                rot = int(float(sd["rotation"]))
    return int(rot) % 360


def probe_video(path: Path) -> VideoMeta:
    """Read metadata with ffprobe and confirm OpenCV can decode a frame."""
    data = probe(path)
    vs = next(s for s in data["streams"] if s.get("codec_type") == "video")
    avg, nominal = _rate(vs.get("avg_frame_rate")), _rate(vs.get("r_frame_rate"))
    fps = avg or nominal
    vfr = bool(avg and nominal and abs(avg - nominal) / nominal > 0.02)
    # Variable-frame-rate phone clips are converted at their nominal rate (e.g. 59.94), not the average (e.g. 58.2):
    # every real frame is kept and only repeated where the phone skipped one, so each frame's time stays correct
    # (measured: mean timing error −0.9 ms vs −21 ms, worst −18 ms vs −36 ms when resampling at the average rate).
    process_fps = nominal if (vfr and nominal and avg and avg < nominal <= min(1.5 * avg, 240.0)) else None
    w, h = int(vs.get("width") or 0), int(vs.get("height") or 0)
    # non-square pixels (some phone / messaging-app exports): work in display proportions, as the browser shows it
    sar = _rate(vs.get("sample_aspect_ratio", "1:1").replace(":", "/")) or 1.0
    if abs(sar - 1.0) > 0.01:
        w = int(round(w * sar))
    rot = _rotation(vs)
    if rot in (90, 270):
        w, h = h, w
    dur = float(vs.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb = int(vs.get("nb_frames") or 0) or int(round(dur * fps))
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    if not ok or fps <= 0 or w <= 0 or h <= 0:
        raise BowlIQError(
            "UNREADABLE_VIDEO",
            "The video could not be decoded.",
            "The file may be damaged or use an unsupported codec. Export it as MP4 (H.264) and upload again.",
        )
    return VideoMeta(path=str(path), fps=fps, width=w, height=h, frame_count=nb, duration_s=dur, rotation=rot,
                     codec=str(vs.get("codec_name", "")), vfr=vfr, process_fps=process_fps)


def resolve_trim(duration_s: float, trim_start: Optional[float], trim_end: Optional[float]) -> tuple[float, float]:
    """Single source of truth for trim validation (used by the upload route and the pipeline).

    Returns (start, end) in seconds within the clip. Raises for invalid order, < 1 s or > 30 s.
    """
    for v in (trim_start, trim_end):
        if v is not None and not math.isfinite(v):
            raise BowlIQError("INVALID_TRIM", "Trim times must be finite numbers.", "Move the trim handles and try again.", 400)
    if trim_start is not None and trim_start < 0:
        raise BowlIQError("INVALID_TRIM", "Trim start can't be negative.", "Move the start handle to 0 s or later.", 400)
    if trim_start is not None and trim_end is not None and trim_end <= trim_start:
        raise BowlIQError("INVALID_TRIM", "Trim end must be after trim start.", "Move the end handle after the start handle.", 400)
    start = trim_start or 0.0
    end = min(trim_end, duration_s) if trim_end is not None else duration_s
    dur = end - start
    if dur < settings.min_duration_s:
        raise BowlIQError(
            "CLIP_TOO_SHORT", f"The selected section is {max(0.0, dur):.2f} s long; a delivery needs at least 1 s.",
            "Choose a section that includes the last run-up strides, the delivery stride and release.")
    if dur > settings.max_duration_s:
        raise BowlIQError("CLIP_TOO_LONG", f"The selected section is {dur:.1f} s long; the limit is 30 s.",
                          "Trim the clip to one delivery (usually 2–5 s).")
    return start, end


def _check(label: str, value: str, status: str, note: str) -> dict:
    return {"label": label, "value": value, "status": status, "note": note}


def metadata_checks(meta: VideoMeta, duration_after_trim: float) -> tuple[list[dict], list[str], str]:
    """Return (checks, reasons, level) from metadata alone (duration limits are enforced by resolve_trim)."""
    checks: list[dict] = []
    reasons: list[str] = []
    level = 2
    fps = meta.fps
    q = P.quality
    if fps >= q.fps_good:
        checks.append(_check("Frame rate", f"{fps:.0f} fps", "ok", "Good for timing release."))
    elif fps >= q.fps_ok:
        checks.append(_check("Frame rate", f"{fps:.0f} fps", "ok", "Fine. 120 fps would time release more precisely."))
        level = min(level, 1)
        reasons.append(f"{fps:.0f} fps: release may fall between frames.")
    else:
        checks.append(_check("Frame rate", f"{fps:.0f} fps", "warn",
                             "Release timing will be approximate and ball speed less precise. Film at 60 fps or more."))
        level = 0
        reasons.append(f"{fps:.0f} fps is below 50 fps: event timing is approximate.")
    short = min(meta.width, meta.height)
    if short >= q.res_good:
        checks.append(_check("Resolution", f"{meta.width}×{meta.height}", "ok", ""))
    elif short >= q.res_ok:
        checks.append(_check("Resolution", f"{meta.width}×{meta.height}", "warn", "Joints are small; 720p or more is better."))
        level = min(level, 1)
        reasons.append("Resolution below 720p.")
    else:
        checks.append(_check("Resolution", f"{meta.width}×{meta.height}", "warn", "Very low resolution; joints may be missed."))
        level = 0
        reasons.append("Resolution below 480p.")
    checks.append(_check("Length", f"{duration_after_trim:.1f} s", "ok", "Long enough for one delivery."))
    if meta.vfr:
        checks.append(_check("Frame timing", "Variable frame rate", "warn",
                             "The phone recorded at a varying rate; frames are resampled to a constant rate, so "
                             "event timing may be off by one frame."))
        level = min(level, 1)
        reasons.append("Variable frame rate: a few frames the phone skipped are filled by repeating the previous one, "
                       "so timing may be off by one frame at those moments.")
    orient = "Landscape" if meta.width >= meta.height else "Portrait"
    rot_note = f" (rotated {meta.rotation}° from metadata)" if meta.rotation else ""
    checks.append(_check("Orientation", orient + rot_note, "ok" if orient == "Landscape" else "warn",
                         "" if orient == "Landscape" else "Landscape keeps more of the run-up in view."))
    return checks, reasons, LEVELS[level]


def person_boxes(pose: PoseSequence, pad: float = 0.25) -> np.ndarray:
    """(F, 4) x0, y0, x1, y1 boxes around the tracked person (NaN when missing)."""
    kp = pose.kp
    out = np.full((kp.shape[0], 4), np.nan)
    has = np.isfinite(kp[:, :, 0]).any(axis=1)  # no nan-reductions over all-NaN rows → no warnings, thread-safe
    if has.any():
        x = kp[has, :, 0]
        y = kp[has, :, 1]
        x0, x1 = np.nanmin(x, axis=1), np.nanmax(x, axis=1)
        y0, y1 = np.nanmin(y, axis=1), np.nanmax(y, axis=1)
        w, h = x1 - x0, y1 - y0
        out[has] = np.stack([x0 - pad * w - 0.15 * h, y0 - pad * h, x1 + pad * w + 0.15 * h, y1 + pad * h], axis=1)
    return out


def background_shift(g1: np.ndarray, g2: np.ndarray, mask: np.ndarray, fps: float, gap: int = 2) -> float:
    """Apparent background shift between two grey frames `gap` apart, in frame widths per second: mean absolute
    difference ÷ mean gradient ≈ pixels moved (for small shifts), so texture, exposure, resolution and frame rate
    cancel out."""
    a, b = g1.astype(np.float32), g2.astype(np.float32)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = float(np.hypot(gx, gy)[mask].mean())
    diff = float(np.abs(a - b)[mask].mean())
    return diff / max(grad, 1e-3) / gap * fps / g1.shape[1]


def camera_motion(video_path: Path, boxes: np.ndarray, samples: int = 24) -> Optional[float]:
    """Median background shift (frame widths per second) between frames 2 apart, outside the person box."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    if n < 4:
        cap.release()
        return None
    idx = np.unique(np.linspace(0, n - 3, min(samples, n - 2)).astype(int))
    vals: list[float] = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok1, f1 = cap.read()
        cap.read()
        ok2, f2 = cap.read()
        if not (ok1 and ok2):
            continue
        g1 = cv2.GaussianBlur(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        g2 = cv2.GaussianBlur(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = np.ones(g1.shape, bool)
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


SHRINK_BEHIND = 0.5  # run-up body size late/early below this: running away from the camera (filmed from behind)
GROW_FRONT = 2.0  # above this: running straight at the camera
SIDE_BAND = (0.8, 1.25)  # roughly constant size while crossing the frame: side-on
SIDE_PROFILE_W = 0.3  # shoulder width ÷ torso below this: shoulders seen edge-on, i.e. from the side
SIDE_STRIDE = 0.5  # feet ≥ half a body length apart across the picture in the stride


def _body_length(k: np.ndarray) -> np.ndarray:
    """Posture-robust body size per frame: longer leg (thigh + shin) + trunk, in pixels."""
    def seg(a: str, b: str) -> np.ndarray:
        return np.linalg.norm(k[:, J[a], :2] - k[:, J[b], :2], axis=1)
    leg = np.fmax(seg("left_hip", "left_knee") + seg("left_knee", "left_ankle"),
                  seg("right_hip", "right_knee") + seg("right_knee", "right_ankle"))
    trunk = np.linalg.norm(0.5 * (k[:, J["left_shoulder"], :2] + k[:, J["right_shoulder"], :2])
                           - 0.5 * (k[:, J["left_hip"], :2] + k[:, J["right_hip"], :2]), axis=1)
    return leg + trunk


def world_kp(pose: PoseSequence) -> np.ndarray:
    """Keypoints in the fixed reference view when the camera moved (Phase 1c), else as filmed."""
    if getattr(pose, "cam", None) is None:
        return pose.kp
    from app.pipeline.adapt import apply_path
    kp = pose.kp.copy()
    kp[..., :2] = apply_path(pose.kp[..., :2], pose.cam)
    return kp


def view_features(pose: PoseSequence, runup_end: Optional[int] = None) -> Optional[dict]:
    """Geometry of the run-up (frames with the bowler before `runup_end`, else the first 60% of them)."""
    kp_all = world_kp(pose)
    idx = np.where(np.isfinite(kp_all[:, 0, 0]))[0]
    if runup_end is not None:
        pre = idx[idx < runup_end]
        idx = pre if len(pre) >= 6 else idx[: max(6, int(0.6 * len(idx)))]
    else:
        idx = idx[: max(6, int(0.6 * len(idx)))]
    k = kp_all[idx]
    if len(k) < 5:
        return None
    hip = 0.5 * (k[:, J["left_hip"], :2] + k[:, J["right_hip"], :2])
    torso = np.linalg.norm(0.5 * (k[:, J["left_shoulder"], :2] + k[:, J["right_shoulder"], :2]) - hip, axis=1)
    size = _body_length(k)
    ok = np.isfinite(hip[:, 0]) & np.isfinite(torso) & (torso > 1) & np.isfinite(size)
    if ok.sum() < 5:
        return None
    s = size[ok]
    q = max(2, len(s) // 4)
    dx = k[ok, J["left_shoulder"], 0] - k[ok, J["right_shoulder"], 0]
    # how far apart the feet get sideways in the picture (delivery stride), in body lengths, over the whole visible
    # span: only possible when filmed from the side, whatever the camera does (a following camera kills "travel")
    ka = kp_all[np.isfinite(kp_all[:, 0, 0])]
    sa = _body_length(ka)
    spread = np.abs(ka[:, J["left_ankle"], 0] - ka[:, J["right_ankle"], 0]) / np.where(sa > 1, sa, np.nan) * 2
    return {
        "stride_spread": float(np.nanpercentile(spread, 90)) if np.isfinite(spread).any() else 0.0,
        "travel": float((np.percentile(hip[ok, 0], 90) - np.percentile(hip[ok, 0], 10)) / np.median(s) * 2),
        "width": float(np.median(np.abs(dx) / torso[ok])),
        "size_ratio": float(np.median(s[-q:]) / max(1e-6, np.median(s[:q]))),
        # MediaPipe labels anatomical sides: facing the camera, the left shoulder appears on the image right
        "left_on_right": float(np.mean(dx > 0)),
    }


CUT_MIN, CUT_X_MEDIAN = P.scene_cuts.min_change, P.scene_cuts.x_median  # colour-histogram change marking a hard cut


def scene_cuts(video_path: Path) -> list[int]:
    """Frames where a new shot starts (hard cuts: highlight reels, screen-recorded replays that loop)."""
    cap = cv2.VideoCapture(str(video_path))
    prev, d = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        small = cv2.resize(f, (160, int(160 * f.shape[0] / max(1, f.shape[1]))))
        h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 16], [0, 180, 0, 256])
        h = cv2.normalize(h, h).flatten()
        d.append(0.0 if prev is None else float(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)))
        prev = h
    cap.release()
    if len(d) < 3:
        return []
    arr = np.array(d)
    thr = max(CUT_MIN, CUT_X_MEDIAN * float(np.median(arr[1:])))
    return [i for i in range(1, len(arr)) if arr[i] > thr]


def keep_one_shot(pose: PoseSequence, cuts: list[int]) -> Optional[tuple[int, int]]:
    """Restrict the bowler (and every detection) to the shot where he is seen most. Returns that shot's frames."""
    n = pose.kp.shape[0]
    bounds = [0, *cuts, n]
    shots = [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    seen = np.isfinite(pose.kp[:, 0, 0])
    a, b = max(shots, key=lambda sh: int(seen[sh[0]:sh[1] + 1].sum()))
    pose.kp[:a] = np.nan
    pose.kp[b + 1:] = np.nan
    if pose.raw is not None:
        for f in list(range(0, a)) + list(range(b + 1, n)):
            if f < len(pose.raw.dets):
                pose.raw.dets[f] = []
    if pose.active_span is not None:
        lo, hi = max(pose.active_span[0], a), min(pose.active_span[1], b)
        pose.active_span = (lo, hi) if lo <= hi else (a, b)
    return a, b


def classify_view(pose: PoseSequence, runup_end: Optional[int] = None) -> str:
    """side-on | front-on (facing the bowler) | behind (behind the bowler) | diagonal | unknown, from the run-up:
    how much the bowler crosses the frame, how much bigger/smaller he gets, and how wide the shoulders look.

    Calibrated on the real clips: side-on 0.97–1.00 size ratio; the 19°-oblique WhatsApp clip 0.57 (diagonal);
    a behind-the-bowler clip 0.33."""
    f = view_features(pose, runup_end)
    if f is None:
        return "unknown"
    r, travel, width = f["size_ratio"], f["travel"], f["width"]
    if SIDE_BAND[0] <= r <= SIDE_BAND[1] and travel <= 1.2 and width < SIDE_PROFILE_W and f["stride_spread"] >= SIDE_STRIDE:
        # a camera following the bowler: he hardly crosses the frame, but a thin shoulder profile and feet spread wide
        # across the picture in the stride can only be seen from the side (broadcast replay: width 0.11, spread 0.93)
        return "side-on"
    if r < SHRINK_BEHIND:
        return "behind"
    if r > GROW_FRONT:
        return "front-on"
    if SIDE_BAND[0] <= r <= SIDE_BAND[1]:
        if travel > 1.2 and width < 0.55:
            return "side-on"
        if width >= 0.55 and travel < 1.2:  # square shoulders, little sideways travel: on the line of the run
            return "front-on" if f["left_on_right"] >= 0.5 else "behind"
        return "side-on" if travel > 1.2 else "unknown"
    # clearly growing or shrinking but not straight at/away from the camera
    if travel > 1.0:
        return "diagonal"
    return "front-on" if r > 1 else "behind"


def finalize(meta: VideoMeta, duration: float, checks: list[dict], reasons: list[str], level: str,
             motion: Optional[float], view: str, expected_view: str | tuple[str, ...], detection_rate: float) -> QualityReport:
    lvl = LEVELS.index(level)
    cam = P.camera
    if motion is None:
        checks.append(_check("Camera", "Not checked", "warn", "Too little background visible to check for camera motion."))
    elif motion < cam.steady_wps:
        checks.append(_check("Camera", "Steady", "ok", ""))
    elif motion < cam.moving_wps:
        checks.append(_check("Camera", "Some movement", "warn", "Use a tripod; camera movement adds error to distances."))
        lvl = min(lvl, 1)
        reasons.append("Some camera movement detected.")
    else:
        checks.append(_check("Camera", "Moving", "warn", "The camera pans or shakes. Speeds and distances are unreliable."))
        lvl = 0
        reasons.append("Strong camera motion: speeds and distances are unreliable.")
    expected = (expected_view,) if isinstance(expected_view, str) else tuple(expected_view)
    if view in expected:
        checks.append(_check("View", view.capitalize(), "ok", ""))
    elif view == "diagonal":
        checks.append(_check("View", "Diagonal", "warn", "Filmed at an angle: only measurements that stay reliable at "
                             "this angle are reported, at lower confidence."))
        lvl = min(lvl, 1)
    else:
        want = " or ".join(expected)
        note = f"This looks {view} rather than {want}." if view != "unknown" else "Could not confirm the camera angle."
        checks.append(_check("View", view.capitalize() if view != "unknown" else "Unclear", "warn", note))
        lvl = min(lvl, 0 if view != "unknown" else 1)
        reasons.append(note)
    if detection_rate < P.quality.detection_low:
        lvl = 0
        reasons.append(f"The bowler was detected in only {detection_rate * 100:.0f}% of frames.")
    elif detection_rate < P.quality.detection_ok:
        lvl = min(lvl, 1)
        reasons.append(f"The bowler was detected in {detection_rate * 100:.0f}% of frames.")
    checks.append(_check("Skeleton", f"{detection_rate * 100:.0f}% of frames", "ok" if detection_rate >= P.quality.detection_ok else "warn",
                         "" if detection_rate >= P.quality.detection_ok else "Joints were missing in some frames (blur, occlusion or framing)."))
    if not reasons:
        reasons.append(f"{meta.fps:.0f} fps, steady camera, {view}, bowler visible throughout.")
    return QualityReport(level=LEVELS[lvl], reasons=reasons, fps=meta.fps, width=meta.width, height=meta.height,
                         duration_s=round(duration, 3), rotation=meta.rotation, camera_motion=motion,
                         view_detected=view, checks=checks)

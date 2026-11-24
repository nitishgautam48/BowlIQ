"""Stage 2: trim, auto-rotate, resize (long side ≤ 1280), constant frame rate at the true fps, H.264 temp MP4."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from app.core.config import settings
from app.core.ffmpeg import SAFE_INPUT, ffmpeg_path, h264_args, require_ffmpeg, run
from app.pipeline.types import VideoMeta


def target_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    s = min(1.0, long_side / max(w, h))
    tw = max(2, int(round(w * s / 2)) * 2)
    th = max(2, int(round(h * s / 2)) * 2)
    return tw, th


ROTATE_FILTERS = {"none": "", "cw90": "transpose=1,", "ccw90": "transpose=2,", "180": "hflip,vflip,"}


def preprocess(src: Path, dst: Path, meta: VideoMeta, trim_start: Optional[float], trim_end: Optional[float],
               rotate: str = "none") -> VideoMeta:
    """rotate: an extra rotation on top of the metadata one (clip_context.check_rotation found the bowler sideways or
    upside down)."""
    require_ffmpeg()
    w, h = (meta.height, meta.width) if rotate in ("cw90", "ccw90") else (meta.width, meta.height)
    tw, th = target_size(w, h, settings.max_long_side)
    cmd = [ffmpeg_path(), "-y", "-v", "error"]
    if trim_start is not None and trim_start > 0:
        cmd += ["-ss", f"{trim_start:.3f}"]
    cmd += [*SAFE_INPUT, "-i", str(src)]
    if trim_end is not None:
        dur = trim_end - (trim_start or 0.0)
        cmd += ["-t", f"{dur:.3f}"]
    fps = meta.process_fps or meta.fps
    # the fps filter places each output frame at the nearest source frame in time; "-fps_mode cfr -r" resampling was
    # measured to show frames ~21 ms late on average on variable-frame-rate phone clips
    cmd += ["-vf", f"{ROTATE_FILTERS.get(rotate, '')}scale={tw}:{th}:flags=area,setsar=1,fps={fps:.6f}:round=near", "-fps_mode", "passthrough",
            "-an", *h264_args(17), "-pix_fmt", "yuv420p", str(dst)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(cmd)
    cap = cv2.VideoCapture(str(dst))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoMeta(path=str(dst), fps=fps, width=tw, height=th, frame_count=n, duration_s=n / fps, rotation=0,
                     codec="h264")

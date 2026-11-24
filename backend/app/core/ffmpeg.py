"""ffmpeg/ffprobe helpers (subprocess with list args, no shell). Untrusted media is only read from local files."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from app.core.errors import BowlIQError

log = logging.getLogger("bowliq.ffmpeg")
# Never let a crafted file make ffmpeg open network URLs (e.g. an HLS playlist disguised as .mp4).
SAFE_INPUT = ["-protocol_whitelist", "file,pipe"]
MP4_BOXES = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    return ffmpeg_path() is not None and ffprobe_path() is not None


FFMPEG_MISSING_FIX = (
    "Install ffmpeg and make sure it is on PATH. On Windows: `winget install Gyan.FFmpeg`, then open a new terminal "
    "and check `ffmpeg -version`."
)
UNREADABLE_FIX = "The file may be damaged or use an unsupported codec. Export it as MP4 (H.264) and upload again."


def require_ffmpeg() -> None:
    if not ffmpeg_available():
        raise BowlIQError("FFMPEG_MISSING", "ffmpeg/ffprobe was not found on this computer.", FFMPEG_MISSING_FIX, 500)


def looks_like_mp4_mov(path: Path) -> bool:
    """ISO-BMFF (MP4/MOV) files start with a box header: 4-byte size + 4-byte type."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return len(head) >= 8 and head[4:8] in MP4_BOXES


def probe(path: Path) -> dict:
    """Return ffprobe JSON (streams + format) or raise UNREADABLE_VIDEO."""
    require_ffmpeg()
    if not looks_like_mp4_mov(path):
        raise BowlIQError("UNREADABLE_VIDEO", "The file is not an MP4 or MOV video.", UNREADABLE_FIX)
    cmd = [ffprobe_path(), "-v", "error", *SAFE_INPUT, "-print_format", "json", "-show_streams", "-show_format",
           str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BowlIQError("UNREADABLE_VIDEO", "Reading the video took too long.", UNREADABLE_FIX) from exc
    if out.returncode != 0:
        log.info("ffprobe failed: %s", out.stderr.decode("utf-8", "replace")[-400:])
        raise BowlIQError("UNREADABLE_VIDEO", "The file could not be read as a video.", UNREADABLE_FIX)
    data = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
    if not any(s.get("codec_type") == "video" for s in data.get("streams", [])):
        raise BowlIQError("UNREADABLE_VIDEO", "The file has no video track.", "Upload an MP4 or MOV clip that contains video.")
    return data


def run(cmd: list[str], timeout: int = 600) -> None:
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BowlIQError("FFMPEG_FAILED", "Video processing took too long.", "Try a shorter clip.", 500) from exc
    if out.returncode != 0:
        log.error("ffmpeg failed (rc=%s): %s", out.returncode, out.stderr.decode("utf-8", "replace")[-800:])
        raise BowlIQError("FFMPEG_FAILED", "The video could not be converted.",
                          "Export the clip as MP4 (H.264) and try again. Details are in the backend log.", 500)


@lru_cache(maxsize=1)
def video_encoder() -> str:
    """'h264_nvenc' when the GPU encoder actually works (trial encode), else 'libx264'. BOWLIQ_VIDEO_ENCODER overrides."""
    from app.core.config import settings

    pref = settings.video_encoder
    if pref in ("x264", "libx264", "cpu") or not ffmpeg_available():
        return "libx264"
    try:
        out = subprocess.run([ffmpeg_path(), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                              "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=30, check=False)
        if out.returncode == 0:
            return "h264_nvenc"
        log.info("NVENC unavailable, using libx264: %s", out.stderr.decode("utf-8", "replace")[-200:].strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    if pref == "nvenc":
        log.warning("BOWLIQ_VIDEO_ENCODER=nvenc but NVENC failed its trial encode; using libx264")
    return "libx264"


def h264_args(quality: int, gop: int | None = None) -> list[str]:
    """Encoder arguments for visually ~equivalent quality (x264 CRF ≈ NVENC CQ + 2).

    gop: keyframe interval in frames. Short GOPs (and no B-frames) make frame-accurate seeking near-instant, which the
    interactive players need; the default long GOP is smaller and fine for files that are only played or downloaded."""
    if video_encoder() == "h264_nvenc":
        args = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(quality + 2), "-b:v", "0"]
    else:
        args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(quality)]
    if gop:
        args += ["-g", str(gop), "-keyint_min", str(gop), "-bf", "0"]
    return args

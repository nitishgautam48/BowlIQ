"""Re-encode the interactive (overlay-free) videos of existing analyses with short keyframe intervals, so the Compare
page and frame stepping seek instantly. New analyses are already encoded this way.

Usage: python scripts/reencode_videos.py
"""
from __future__ import annotations

import itertools
import json
import subprocess

from app.core.config import settings
from app.core.ffmpeg import ffmpeg_path, ffprobe_path, h264_args, require_ffmpeg
from app.report.render_video import SEEK_GOP


def keyframe_interval(path) -> int | None:
    """Largest gap between keyframes in the first ~300 frames (None if it can't be read)."""
    out = subprocess.run([ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#300",
                          "-show_entries", "frame=key_frame", "-of", "json", str(path)], capture_output=True, timeout=120)
    try:
        frames = json.loads(out.stdout or b"{}").get("frames", [])
    except ValueError:
        return None
    keys = [i for i, f in enumerate(frames) if f.get("key_frame") == 1]
    if len(keys) < 2:
        return len(frames) or None
    return max(b - a for a, b in itertools.pairwise(keys))


def main(argv: list[str] | None = None) -> int:
    require_ffmpeg()
    root = settings.storage_dir / "analyses"
    done = skipped = 0
    for clean in sorted(root.glob("*/clean.mp4")) if root.exists() else []:
        gap = keyframe_interval(clean)
        if gap is not None and gap <= SEEK_GOP:
            skipped += 1
            continue
        tmp = clean.with_name("clean.reenc.mp4")
        cmd = [ffmpeg_path(), "-y", "-v", "error", "-i", str(clean), "-an", *h264_args(21, SEEK_GOP),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
        if subprocess.run(cmd, capture_output=True, timeout=1800).returncode == 0:
            tmp.replace(clean)
            done += 1
            print(f"re-encoded {clean.parent.name}")
        else:
            tmp.unlink(missing_ok=True)
            print(f"failed {clean.parent.name}")
    print(f"Done: {done} re-encoded, {skipped} already fine.")
    return 0

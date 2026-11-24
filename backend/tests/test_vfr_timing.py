"""Variable-frame-rate phone clips must keep each frame at its true time after preprocessing (Phase 2 found the old
'-fps_mode cfr' resampling at the average rate showed frames 19–35 ms late during the delivery)."""
import subprocess

import numpy as np

from app.core.ffmpeg import ffmpeg_path
from app.pipeline.preprocess import preprocess
from app.pipeline.quality import probe_video


def test_vfr_clip_frames_keep_their_true_time(tmp_path):
    src = tmp_path / "vfr.mp4"
    # 60 fps source whose brightness encodes the frame index (2 grey levels per frame); every 9th frame is dropped
    # while keeping timestamps, like a phone skipping frames under load
    subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", "nullsrc=s=160x120:r=60:d=2",
                    "-vf", "geq=lum='min(250,2*N)':cb=128:cr=128,select='not(eq(mod(n\\,9)\\,4))'",
                    "-fps_mode", "passthrough", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(src)], check=True)
    meta = probe_video(src)
    assert meta.vfr and meta.process_fps is not None and abs(meta.process_fps - 60) < 0.5
    out = preprocess(src, tmp_path / "prep.mp4", meta, None, None)
    raw = subprocess.run([ffmpeg_path(), "-v", "error", "-i", str(tmp_path / "prep.mp4"), "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
    w, h = out.width, out.height
    lum = np.frombuffer(raw, np.uint8).reshape(-1, h * w * 3 // 2)[:, : h * w].mean(1)  # untouched Y plane
    errs = []
    for j, y in enumerate(lum):
        n = int(round(y / 2))  # source frame index shown in output frame j
        if n < 124:  # brightness saturates after that
            errs.append(n / 60 - j / out.fps)
    errs = np.array(errs)
    assert len(errs) > 100
    assert abs(errs.mean()) < 0.006  # no systematic delay
    assert np.all(np.abs(errs) <= 1 / 60 + 1e-3)  # never more than one frame (only where the phone skipped one)

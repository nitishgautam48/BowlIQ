"""Invariance: the same delivery must give the same answer however it is filmed.

Fast tests (keypoints): mirror (direction and arm flip), frame rate 30/60/120/240, scale ×0.5/×1.5, a 10 % crop,
a 90° rotation (caught by the rotation check). Video tests (marked slow, ~1–2 min with the CPU pose model): the
bundled synthetic sample clip mirrored, resampled to 30/60 fps, scaled to 50 %/150 %, cropped 10 % and rotated 90°
with a rotation tag, each through the real pipeline.

Tolerance: ±1 frame at the native rate; for a resampled clip, the timing error may also include one frame of the new
rate.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from app.pipeline import clip_context as CX
from app.pipeline.consensus import detect_delivery
from app.pipeline.scale import estimate_scale
from app.pipeline.smooth import smooth_pose
from app.pipeline.types import ALL_JOINTS, J
from tests.synthetic import synth_pose

TRUTH = dict(n=132, bfc=62, ffc=82, release=94)  # at 120 fps


def _events(pose, arm=None, direction=None):
    wkp = pose.kp
    a = arm or CX.detect_arm(wkp, pose.fps).value
    d = direction or CX.detect_direction(wkp, CX.detect_arm(wkp, pose.fps).details.get("peak_frame")).value
    kin = smooth_pose(pose, bowling_arm=a, direction=d)
    ev = detect_delivery(kin, estimate_scale(kin, None).body_height_px, strict=False)
    return kin, ev


def _mirror(pose):
    """Flip the picture left↔right: x mirrored and (as a pose model would) the anatomical labels swapped."""
    kp = pose.kp.copy()
    kp[..., 0] = pose.width - 1 - kp[..., 0]
    for name in ALL_JOINTS:
        if name.startswith("left_"):
            r = name.replace("left_", "right_")
            kp[:, [J[name], J[r]]] = kp[:, [J[r], J[name]]]
    pose.kp = kp
    return pose


def _check(ev, truth, fps, tol_frames=1):
    for k in ("bfc", "ffc", "release"):
        got, want = getattr(ev, k).frame, getattr(truth, k)
        assert abs(got - want) <= tol_frames, f"{k}: {got} vs {want} at {fps} fps"


def test_mirror_flips_direction_and_arm_only():
    pose, truth = synth_pose("left_to_right", arm="right", fps=120, **TRUTH)
    kin, ev = _events(pose)
    assert (kin.bowling_arm, kin.direction) == ("right", "left_to_right")
    _check(ev, truth, 120)
    pose2, _ = synth_pose("left_to_right", arm="right", fps=120, **TRUTH)
    kin2, ev2 = _events(_mirror(pose2))
    assert (kin2.bowling_arm, kin2.direction) == ("left", "right_to_left")
    _check(ev2, truth, 120)


@pytest.mark.parametrize("fps", [30, 60, 120, 240])
def test_frame_rate(fps):
    s = fps / 120
    pose, truth = synth_pose(fps=float(fps), n=int(TRUTH["n"] * s), bfc=int(TRUTH["bfc"] * s), ffc=int(TRUTH["ffc"] * s),
                             release=int(TRUTH["release"] * s))
    _, ev = _events(pose)
    _check(ev, truth, fps)


@pytest.mark.parametrize("factor", [0.5, 1.5])
def test_scale(factor):
    pose, truth = synth_pose(fps=120, **TRUTH)
    pose.kp[..., :2] *= factor
    pose.width, pose.height = int(pose.width * factor), int(pose.height * factor)
    kin, ev = _events(pose)
    assert kin.bowling_arm == "right" and kin.direction == "left_to_right"
    _check(ev, truth, 120)


def test_crop_10_percent():
    pose, truth = synth_pose(fps=120, **TRUTH)
    dx, dy = 0.05 * pose.width, 0.05 * pose.height
    pose.kp[..., 0] -= dx
    pose.kp[..., 1] -= dy
    pose.width, pose.height = int(pose.width * 0.9), int(pose.height * 0.9)
    _, ev = _events(pose)
    _check(ev, truth, 120)


def test_rotation_is_detected_from_the_pose():
    pose, _ = synth_pose(fps=120, **TRUTH)
    assert CX.check_rotation(pose.kp).value == "none"
    rot = pose.kp.copy()  # picture turned 90° clockwise: (x, y) → (H − y, x); the head now points right
    rot[..., 0], rot[..., 1] = pose.height - pose.kp[..., 1], pose.kp[..., 0]
    assert CX.check_rotation(rot).value == "ccw90"
    rot[..., 0] = pose.height - rot[..., 0]  # head pointing left
    assert CX.check_rotation(rot).value == "cw90"
    up = pose.kp.copy()
    up[..., 1] = pose.height - up[..., 1]
    assert CX.check_rotation(up).value == "180"


def test_view_and_direction_have_confidence_and_reasons():
    pose, _ = synth_pose(fps=120, **TRUTH)
    for d in (CX.classify_view(pose.kp), CX.detect_direction(pose.kp), CX.detect_arm(pose.kp, 120)):
        assert d.confidence in ("Low", "Medium", "High") and d.reasons
    assert CX.classify_view(pose.kp).value == "side-on"


# ───────────────────────────── video level (real pipeline) ─────────────────────────────

SAMPLE_TRUTH = {"bfc": 70, "ffc": 92, "release": 104}  # samples/sample_delivery.mp4, 120 fps
AUGMENT = {
    "mirror": ["-vf", "hflip"],
    "fps30": ["-vf", "fps=30"],
    "fps60": ["-vf", "fps=60"],
    "scale50": ["-vf", "scale=iw/2:ih/2"],
    "scale150": ["-vf", "scale=trunc(iw*1.5/2)*2:trunc(ih*1.5/2)*2"],
    "crop10": ["-vf", "crop=trunc(iw*0.9/2)*2:trunc(ih*0.9/2)*2"],
    "rot90meta": ["-vf", "transpose=1"],
}


def _run_clip(tmp_path, monkeypatch, src):
    from app.core.config import settings
    from app.core.storage import AnalysisFiles
    from app.pipeline import orchestrator as O
    from app.pipeline.pose import get_estimator

    files = AnalysisFiles("inv" + src.stem.replace("_", "")).ensure()
    shutil.copyfile(src, files.input("side", ".mp4"))
    st = O.prepare(files, {"bowler_name": "Inv"}, lambda s, p: None, get_estimator())
    res = O.analyse(files, st, {"bowler_name": "Inv"}, [], lambda s, p: None, render=False)
    assert settings.storage_dir
    return res


@pytest.mark.slow
@pytest.mark.parametrize("aug", list(AUGMENT))
def test_video_augmentation(aug, tmp_path, monkeypatch):
    from app.core.ffmpeg import ffmpeg_available, ffmpeg_path
    from pathlib import Path
    if not ffmpeg_available():
        pytest.skip("ffmpeg not installed")
    sample = Path(__file__).resolve().parents[1] / "samples" / "sample_delivery.mp4"
    out = tmp_path / f"aug_{aug}.mp4"
    subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-i", str(sample), *AUGMENT[aug], "-an", "-c:v", "libx264",
                    "-crf", "16", "-pix_fmt", "yuv420p", str(out)], check=True)
    if aug == "rot90meta":  # pixels turned clockwise, rotation tag says turn back: players (and BowlIQ) show it upright
        tagged = tmp_path / "aug_rot90meta_tagged.mp4"
        subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-display_rotation", "90", "-i", str(out), "-c", "copy",
                        str(tagged)], check=True)
        out = tagged
    res = _run_clip(tmp_path, monkeypatch, out)
    fps = res["video"]["fps"]
    tol_s = 1 / 120 + (1 / fps if abs(fps - 120) > 1 else 0) + 1e-6
    for k, f120 in SAMPLE_TRUTH.items():
        t = res["events"][k]["frame"] / fps
        assert abs(t - f120 / 120) <= tol_s, f"{aug} {k}: {t:.4f}s vs {f120 / 120:.4f}s"
    assert res["bowling_arm"] == ("left" if aug == "mirror" else "right")
    assert res["direction"] == ("right_to_left" if aug == "mirror" else "left_to_right")
    assert res["context"]["view"]["value"] == "side-on"

"""Pose-engine selection, GPU→CPU fallback, and the real-footage robustness fixes found while adding the GPU engine."""
import numpy as np
import pytest

from app.core.errors import BowlIQError
from app.pipeline.events import detect_events, find_contacts, head_y
from app.pipeline.pose import FallbackPoseEstimator, MediaPipePoseEstimator, get_estimator
from app.pipeline.scale import estimate_scale
from app.pipeline.smooth import smooth_pose
from app.pipeline.types import J
from tests.synthetic import synth_pose


class _Fails:
    name = "ultralytics_test"

    def estimate(self, *a, **k):
        raise BowlIQError("POSE_NOT_FOUND", "not found", "fix")


class _Works:
    name = "mediapipe_test"

    def __init__(self):
        self.pose, _ = synth_pose()

    def estimate(self, *a, **k):
        return self.pose


def test_fallback_engine_used_when_gpu_engine_finds_nobody():
    est = FallbackPoseEstimator(_Fails(), _Works())
    pose = est.estimate("x.mp4", 120.0)
    assert est.used_fallback and pose.kp.shape[1] == 21


def test_fallback_does_not_hide_other_errors():
    class _Broken(_Fails):
        def estimate(self, *a, **k):
            raise BowlIQError("POSE_INIT_FAILED", "boom", "fix")

    with pytest.raises(BowlIQError) as e:
        FallbackPoseEstimator(_Broken(), _Works()).estimate("x.mp4", 120.0)
    assert e.value.code == "POSE_INIT_FAILED"


def test_mediapipe_backend_selected():
    assert isinstance(get_estimator(), MediaPipePoseEstimator)  # BOWLIQ_POSE_BACKEND=mediapipe in conftest


def test_nvenc_falls_back_to_x264():
    from app.core.ffmpeg import h264_args

    assert h264_args(21)[:2] == ["-c:v", "libx264"]  # BOWLIQ_VIDEO_ENCODER=x264 in conftest


def test_gradual_foot_stop_is_detected():
    """Real front-ankle track (frames 62–80 of a 58 fps phone clip, YOLO keypoints, mirrored, image y) whose true
    front-foot contact was confirmed frame by frame at 76. The foot decelerates over several frames and the ankle
    keeps sinking after the horizontal stop; the old 3-frame rule missed it entirely."""
    x = [195, 221, 247, 272, 294, 315, 331, 348, 360, 370, 380, 390, 397, 401, 404, 406, 408, 408, 406]
    y_img = [327, 330, 331, 332, 334, 338, 350, 351, 353, 350, 344, 336, 325, 314, 301, 288, 277, 271, 268]
    track = np.stack([np.array(x, float), -np.array(y_img, float)], 1)  # Y up
    contacts = find_contacts(track, 58.09, body_h=231.0)
    first = 62 + contacts[0][0] if contacts else None
    assert first is not None and abs(first - 76) <= 1


def test_release_check_uses_head_fallback_when_nose_hidden():
    pose, truth = synth_pose("left_to_right")
    pose.kp[truth.release - 10:truth.release + 11, J["nose"], 2] = 0.05  # face turned away at release (longer than the 0.1 s gap fill)
    kin = smooth_pose(pose)
    assert not np.isfinite(kin.joint("nose")[truth.release, 1])
    assert np.isfinite(head_y(kin, truth.release, 300.0))
    ev = detect_events(kin, estimate_scale(kin, None).body_height_px)
    assert abs(ev.release.frame - truth.release) <= 1


def test_empty_lead_in_is_ignored_not_rejected():
    """A clip that starts with seconds of empty field (bowler on screen for only ~1 s of 6 s) is analysed on the
    bowler's window instead of failing the visibility check (real 11.29.51 clip: bowler in 13–19 % of frames)."""
    from app.pipeline.pose import finalize_candidates

    pose, truth = synth_pose("left_to_right")
    lead = 400  # ~3.3 s of empty frames at 120 fps
    cands = [[] for _ in range(lead)] + [[p] for p in pose.kp]
    seq = finalize_candidates(cands, 120.0, pose.width, pose.height, "cpu")
    assert seq.active_span[0] == lead and seq.detection_rate > 0.95
    assert np.isnan(seq.kp[: lead, 0, 0]).all()
    kin = smooth_pose(seq)
    ev = detect_events(kin, estimate_scale(kin, None).body_height_px)
    assert abs(ev.ffc.frame - (lead + truth.ffc)) <= 1 and abs(ev.release.frame - (lead + truth.release)) <= 1


def test_too_short_appearance_still_rejected():
    from app.pipeline.pose import finalize_candidates

    pose, _ = synth_pose("left_to_right")
    cands = [[] for _ in range(300)] + [[p] for p in pose.kp[:30]] + [[] for _ in range(300)]  # 0.25 s on screen
    with pytest.raises(BowlIQError) as e:
        finalize_candidates(cands, 120.0, pose.width, pose.height, "cpu")
    assert e.value.code == "POSE_NOT_FOUND"


def test_release_must_follow_ffc_by_a_few_frames():
    from app.pipeline import events as E

    assert E.MIN_FFC_RELEASE_S >= 0.03


def test_interactive_video_uses_short_keyframe_interval():
    from app.core.ffmpeg import h264_args
    from app.report.render_video import SEEK_GOP

    args = h264_args(21, SEEK_GOP)
    assert args[args.index("-g") + 1] == str(SEEK_GOP) and "-bf" in args
    assert "-g" not in h264_args(21)

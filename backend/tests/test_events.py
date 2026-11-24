import pytest

from app.core.errors import BowlIQError
from app.pipeline.events import detect_events
from app.pipeline.scale import estimate_scale
from app.pipeline.smooth import smooth_pose
from tests.synthetic import synth_pose


@pytest.mark.parametrize("direction", ["left_to_right", "right_to_left"])
@pytest.mark.parametrize("fps,scale", [(120, 1.0), (240, 2.0), (60, 0.5)])
def test_events_within_one_frame(direction, fps, scale):
    pose, truth = synth_pose(direction, fps=fps, n=int(132 * scale), bfc=int(62 * scale), ffc=int(82 * scale),
                             release=int(94 * scale))
    kin = smooth_pose(pose)
    assert kin.direction == direction
    assert kin.bowling_arm == "right"
    ev = detect_events(kin, estimate_scale(kin, None).body_height_px)
    assert abs(ev.bfc.frame - truth.bfc) <= 1
    assert abs(ev.ffc.frame - truth.ffc) <= 1
    assert abs(ev.release.frame - truth.release) <= 1
    assert ev.bfc.frame < ev.ffc.frame < ev.release.frame


def test_left_arm_bowler():
    pose, truth = synth_pose("right_to_left", arm="left")
    kin = smooth_pose(pose)
    assert kin.bowling_arm == "left"
    ev = detect_events(kin, estimate_scale(kin, None).body_height_px)
    assert (abs(ev.bfc.frame - truth.bfc), abs(ev.ffc.frame - truth.ffc), abs(ev.release.frame - truth.release)) <= (1, 1, 1)


def test_no_run_up_fails_with_filming_fix():
    pose, _ = synth_pose("left_to_right")
    pose.kp[:, :, 0] = pose.kp[0:1, :, 0]  # everybody frozen in place
    kin = smooth_pose(pose)
    with pytest.raises(BowlIQError) as e:
        detect_events(kin, estimate_scale(kin, None).body_height_px)
    assert e.value.code == "EVENTS_NOT_FOUND"
    assert "tripod" in e.value.fix


def test_heel_to_flat_settling_still_detected():
    """Real footage: the front ankle keeps sinking for a few frames after it stops moving forward."""
    pose, truth = synth_pose("left_to_right")
    fa = 15  # left ankle (front foot of a right-arm bowler)
    for k in range(1, 4):  # ankle still 3, 2, 1 × 8 px above the ground after the horizontal stop
        pose.kp[truth.ffc + k - 1, fa, 1] -= 8 * (4 - k)
    kin = smooth_pose(pose)
    ev = detect_events(kin, estimate_scale(kin, None).body_height_px)
    assert abs(ev.ffc.frame - truth.ffc) <= 1


# ───── failures found on real 30 fps phone clips (the bound, swapped leg labels) ─────
import numpy as np  # noqa: E402

from app.pipeline.types import J  # noqa: E402


def _bound_with_hands_up(pose, truth):
    """Both hands held high above the head through the bound before back foot contact (a plateau, not a peak)."""
    kp = pose.kp
    a, b = truth.bfc - 20, truth.bfc - 2
    for f in range(a, b):
        top = np.nanmin(kp[f, :, 1])
        for w in ("left_wrist", "right_wrist"):
            kp[f, J[w], 1] = top - 40
    return pose


def test_bound_with_both_hands_up_is_not_taken_for_release():
    pose, truth = synth_pose(fps=30.0, n=66, bfc=31, ffc=41, release=47)
    kin = smooth_pose(_bound_with_hands_up(pose, truth))
    ev = detect_events(kin, estimate_scale(kin, None).body_height_px)
    assert ev.release.frame > truth.bfc  # not the bound (hands up from bfc−20 to bfc−2)
    assert abs(ev.release.frame - truth.release) <= 2 and abs(ev.ffc.frame - truth.ffc) <= 1


def test_legs_labelled_the_wrong_way_round_for_the_whole_clip():
    pose, truth = synth_pose(fps=60.0, n=132, bfc=62, ffc=82, release=94)
    for part in ("hip", "knee", "ankle", "heel", "foot_index"):
        pose.kp[:, [J[f"left_{part}"], J[f"right_{part}"]]] = pose.kp[:, [J[f"right_{part}"], J[f"left_{part}"]]]
    kin = smooth_pose(pose, bowling_arm="right")
    ev = detect_events(kin, estimate_scale(kin, None).body_height_px)
    assert ev.legs_swapped
    assert abs(ev.bfc.frame - truth.bfc) <= 1 and abs(ev.ffc.frame - truth.ffc) <= 1

import numpy as np

from app.pipeline.smooth import detect_bowling_arm, fix_left_right_swaps
from app.pipeline.types import J
from tests.synthetic import synth_pose


def test_left_right_leg_swaps_are_undone():
    pose, _ = synth_pose("left_to_right")
    clean = pose.kp.copy()
    bad = clean.copy()
    for f in (40, 41, 55, 90):  # MediaPipe-style label swaps of knee+ankle
        for a in ("knee", "ankle"):
            bad[f, [J[f"left_{a}"], J[f"right_{a}"]]] = bad[f, [J[f"right_{a}"], J[f"left_{a}"]]]
    fixed, n = fix_left_right_swaps(bad)
    assert n >= 4
    assert np.allclose(fixed[:, J["left_ankle"], :2], clean[:, J["left_ankle"], :2])
    assert np.allclose(fixed[:, J["right_knee"], :2], clean[:, J["right_knee"], :2])


def test_bowling_arm_found_when_release_is_early():
    """Release in the first half of the clip (long follow-through) and a front arm that reaches up in the coil."""
    pose, _truth = synth_pose("left_to_right", n=200, bfc=40, ffc=60, release=72)
    kp = pose.kp
    nose_y = kp[:, J["nose"], 1]
    kp[30:45, J["left_wrist"], 1] = nose_y[30:45] - 40  # front arm 40 px above the nose in the coil
    assert detect_bowling_arm(kp) == "right"

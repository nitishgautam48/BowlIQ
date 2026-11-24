import numpy as np
import pytest

from app.pipeline.metrics import FRONT_KEYS, front_metrics, side_metrics
from app.pipeline.types import N_JOINTS, DeliveryEvents, EventResult, J, Kinematics, Scale

N, FPS, BFC, FFC, REL = 80, 120.0, 40, 50, 60


def _kin() -> Kinematics:
    xy = np.full((N, N_JOINTS, 2), np.nan)
    xy[:, :17] = 0.0
    t = np.arange(N)
    hip_x = 2.0 * t  # 2 px/frame = 240 px/s run-up
    for j in range(17):
        xy[:, j, 0] = hip_x
        xy[:, j, 1] = 100.0
    xy[:, J["nose"], 1] = 180.0
    for s in ("left", "right"):
        xy[:, J[f"{s}_shoulder"], 1] = 160.0
        xy[:, J[f"{s}_hip"], 1] = 100.0
        xy[:, J[f"{s}_hip"], 0] = hip_x
        xy[:, J[f"{s}_shoulder"], 0] = hip_x + 20.0  # trunk leans forward: atan(20/60)
    # back (right) ankle at x=10 at BFC, front (left) ankle at x=170 at FFC → stride 160 px
    xy[:, J["right_ankle"]] = [10.0, 0.0]
    knee_deg = np.full(N, 170.0)
    knee_deg[FFC:FFC + 4] = [170, 165, 160, 155]
    knee_deg[FFC + 4:REL + 1] = np.linspace(156, 160, REL - FFC - 3)
    K = np.array([170.0, 50.0])
    for f in range(N):
        th = np.radians(knee_deg[f])
        xy[f, J["left_knee"]] = K
        xy[f, J["left_ankle"]] = K + [0.0, -50.0]
        if f > BFC:  # the front hip follows the knee only after the run-up window used for runup_speed
            xy[f, J["left_hip"]] = K + 50.0 * np.array([np.sin(th), -np.cos(th)])
    mid_hip = 0.5 * (xy[:, J["left_hip"]] + xy[:, J["right_hip"]])
    for s in ("left", "right"):
        xy[:, J[f"{s}_shoulder"]] = mid_hip + [20.0, 60.0]  # trunk leans forward by atan(20/60)
    xy[:, J["right_wrist"], 1] = 230.0
    xy[:, J["nose"]] = mid_hip + [25.0, 80.0]
    vis = np.full((N, N_JOINTS), 0.95)
    valid = np.ones((N, N_JOINTS), bool)
    valid[:, 17:] = False  # no foot points in this fixture
    return Kinematics(fps=FPS, width=1280, height=720, xy=xy, vis=vis, valid=valid, img_xy=xy.copy(),
                      bowling_arm="right", direction="left_to_right")


def _events() -> DeliveryEvents:
    return DeliveryEvents(EventResult("BFC", BFC, "High", "ok", 1.0), EventResult("FFC", FFC, "High", "ok", 1.0),
                          EventResult("release", REL, "High", "ok", 1.0))


def _by_key(ms):
    return {m.key: m for m in ms}


def test_side_metrics_known_skeleton():
    ms = _by_key(side_metrics(_kin(), _events(), Scale(200.0, 0, None, "body_relative"), "High", "", 0.01))
    assert ms["stride_length_pct_height"].value == pytest.approx(80.0)
    assert ms["front_knee_angle_ffc"].value == pytest.approx(170.0)
    assert ms["front_knee_angle_release"].value == pytest.approx(160.0)
    assert ms["front_knee_flexion_ffc_to_release"].value == pytest.approx(15.0)
    assert ms["front_knee_flexion_ffc_to_release"].frame == FFC + 3
    assert ms["trunk_forward_flexion_release"].value == pytest.approx(np.degrees(np.arctan2(20, 60)))
    assert ms["release_height_pct_height"].value == pytest.approx((230.0 - 0.0) / 200.0 * 100)
    assert ms["runup_speed"].unit == "% height/s"
    assert ms["runup_speed"].value == pytest.approx(240.0 / 200.0 * 100, rel=1e-6)
    capped = {"bowling_elbow_extension", "front_foot_landing_angle_ffc"}  # always Low / Medium by design (2D, small points)
    assert all(m.confidence == "High" for m in ms.values() if m.measured and m.key not in capped)


def test_runup_in_metres_when_height_given():
    ms = _by_key(side_metrics(_kin(), _events(), Scale(200.0, 0, 1.8 / 200.0, "height"), "High", "", 0.01))
    assert ms["runup_speed"].unit == "m/s"
    assert ms["runup_speed"].value == pytest.approx(240.0 * 1.8 / 200.0)


def test_low_quality_lowers_confidence_and_camera_motion_blocks_speed():
    ms = _by_key(side_metrics(_kin(), _events(), Scale(200.0, 0, None, "body_relative"), "Low", "30 fps", 0.2))
    assert ms["stride_length_pct_height"].confidence == "Low"
    assert not ms["runup_speed"].measured and "camera" in ms["runup_speed"].reason.lower()


def test_missing_joint_is_not_measured():
    kin = _kin()
    kin.valid[REL, J["left_knee"]] = False
    ms = _by_key(side_metrics(kin, _events(), Scale(200.0, 0, None, "body_relative"), "High", "", 0.01))
    assert ms["front_knee_angle_release"].measured is False
    assert ms["front_knee_angle_release"].value is None


def test_front_metrics_without_clip():
    ms = front_metrics(None, None, "Low", "", "Add a front-on clip")
    assert {m.key for m in ms} == set(FRONT_KEYS)
    assert all(not m.measured and m.reason == "Add a front-on clip" for m in ms)


def test_front_rotation_capped_at_medium():
    kin = _kin()
    xy = kin.xy
    xy[:, J["left_shoulder"]] = [0.0, 160.0]
    xy[:, J["right_shoulder"]] = [40.0, 160.0]
    xy[:, J["left_hip"]] = [0.0, 100.0]
    xy[:, J["right_hip"]] = [40.0, 100.0]
    th = np.radians(35)
    xy[FFC - 5, J["right_shoulder"]] = [40 * np.cos(th), 160 + 40 * np.sin(th)]  # 35° shoulder-line rotation
    ms = _by_key(front_metrics(kin, _events(), "High", "", ""))
    assert ms["shoulder_counter_rotation"].value == pytest.approx(35.0)
    assert ms["shoulder_counter_rotation"].confidence == "Medium"
    assert "2D projection of a 3D rotation" in ms["shoulder_counter_rotation"].confidence_reason
    assert ms["hip_shoulder_separation_bfc"].value == pytest.approx(0.0, abs=1e-6)


def test_oblique_camera_withholds_distances():
    kin = _kin()
    kin.xy[:, J["right_ankle"], 1] = -60.0  # back foot lands 60 px lower → ground line tilted ~20°
    ms = _by_key(side_metrics(kin, _events(), Scale(200.0, 0, None, "body_relative"), "High", "", 0.01))
    assert not ms["stride_length_pct_height"].measured and "square" in ms["stride_length_pct_height"].reason
    assert not ms["runup_speed"].measured
    assert ms["front_knee_angle_ffc"].measured


def test_planted_front_ankle_drift_marks_leg_unreliable():
    kin = _kin()
    kin.xy[REL, J["left_ankle"]] += [60.0, 0.0]  # tracked ankle jumps 30 % of height while the foot is planted
    ms = _by_key(side_metrics(kin, _events(), Scale(200.0, 0, None, "body_relative"), "High", "", 0.01))
    for k in ("front_knee_angle_release", "front_knee_flexion_ffc_to_release", "release_height_pct_height"):
        assert not ms[k].measured and "planted" in ms[k].reason
    assert ms["front_knee_angle_ffc"].measured


def test_end_on_shoulder_line_is_not_measured():
    """Real rear-view clip: the shoulders overlapped in the picture and the line angle jumped -120° → 22°, which used
    to report a 122° counter-rotation (a false load flag)."""
    kin = _kin()
    xy = kin.xy
    xy[:, J["left_hip"]] = [0.0, 100.0]
    xy[:, J["right_hip"]] = [40.0, 100.0]
    xy[:, J["left_shoulder"]] = [18.0, 160.0]
    xy[:, J["right_shoulder"]] = [22.0, 161.0]  # 4 px apart vs 60 px torso: seen end-on
    ms = _by_key(front_metrics(kin, _events(), "High", "", ""))
    assert not ms["shoulder_counter_rotation"].measured and "overlap" in ms["shoulder_counter_rotation"].reason
    assert not ms["hip_shoulder_separation_bfc"].measured

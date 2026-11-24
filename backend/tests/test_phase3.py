"""Phase 3: camera-angle detection, the new measurements per angle (known geometry), and combining the two clips."""
import numpy as np
import pytest

from app.pipeline import quality as Q
from app.pipeline.metrics import _second_body_height, front_metrics, side_metrics
from app.pipeline.orchestrator import confirm_events, events_agree, map_events
from app.pipeline.types import N_JOINTS, DeliveryEvents, EventResult, J, Kinematics, PoseSequence, Scale
from tests.synthetic import synth_pose
from tests.test_metrics import BFC, FFC, FPS, N, REL, _by_key, _events, _kin


def _ang(a, b, c):
    u, v = np.asarray(a) - b, np.asarray(c) - b
    return np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))


# ───────────────────────────── side-on additions ─────────────────────────────

def test_side_phase3_known_geometry():
    kin = _kin()
    xy = kin.xy
    xy[BFC, J["right_knee"]] = [60.0, 50.0]  # back leg at BFC: hip (80,100), knee (60,50), ankle (10,0)
    kin.valid[FFC, [J["left_heel"], J["left_foot_index"]]] = True
    xy[FFC, J["left_heel"]] = [165.0, 2.0]
    xy[FFC, J["left_foot_index"]] = [185.0, 0.0]  # toe slightly lower: toe-first landing
    xy[52, J["left_wrist"], 1] = 250.0  # front arm highest 8 frames before release
    # bowling arm: upper arm rotates -40° → +80° from FFC to release, elbow angle opens 150° → 170°
    sh = xy[:, J["right_shoulder"]]
    for f in range(FFC, REL + 1):
        u = (f - FFC) / (REL - FFC)
        a, e = np.radians(-40 + 120 * u), np.radians(150 + 20 * u)
        xy[f, J["right_elbow"]] = sh[f] + 30 * np.array([np.cos(a), np.sin(a)])
        back = a + np.pi  # elbow → shoulder direction
        xy[f, J["right_wrist"]] = xy[f, J["right_elbow"]] + 30 * np.array([np.cos(back + e), np.sin(back + e)])
    ms = _by_key(side_metrics(kin, _events(), Scale(200.0, 0, None, "body_relative"), "High", "", 0.01))
    assert ms["head_ahead_of_front_foot_release"].value == pytest.approx((xy[REL, J["nose"], 0] - 170.0) / 2.0)
    assert ms["back_knee_angle_bfc"].value == pytest.approx(_ang([80, 100], np.array([60.0, 50.0]), [10, 0]))
    h = xy[FFC, J["left_hip"]]
    assert ms["front_leg_plant_angle_ffc"].value == pytest.approx(np.degrees(np.arctan2(170.0 - h[0], h[1] - 0.0)))
    assert ms["release_ahead_of_front_foot"].value == pytest.approx((xy[REL, J["right_wrist"], 0] - 170.0) / 2.0)
    assert ms["front_foot_landing_angle_ffc"].value == pytest.approx(np.degrees(np.arctan2(-2.0, 20.0)))
    assert ms["front_foot_landing_angle_ffc"].confidence == "Medium"
    assert ms["front_arm_pulldown_ms"].value == pytest.approx((REL - 52) / FPS * 1000)
    el = ms["bowling_elbow_extension"]
    assert el.value == pytest.approx(170 - (150 + 20 * 3 / 10), abs=0.01)  # upper arm horizontal nearest frame FFC+3
    assert el.confidence == "Low" and "legal" in el.confidence_reason
    assert 0 < ms["speed_kept_at_ffc"].value < 100
    assert ms["head_bob_runup"].measured and ms["head_bob_runup"].value == pytest.approx(0.0, abs=1e-6)


def test_side_phase3_honest_not_measured():
    kin = _kin()  # fixture has no heel/toe points (like the GPU engine)
    ms = _by_key(side_metrics(kin, _events(), Scale(200.0, 0, None, "body_relative"), "High", "", 0.2))
    assert not ms["front_foot_landing_angle_ffc"].measured and "heel" in ms["front_foot_landing_angle_ffc"].reason
    assert not ms["head_bob_runup"].measured and "camera" in ms["head_bob_runup"].reason.lower()
    assert not ms["speed_kept_at_ffc"].measured
    kin.xy[:, J["right_ankle"], 1] = -60.0  # oblique: along-pitch distances withheld
    ms = _by_key(side_metrics(kin, _events(), Scale(200.0, 0, None, "body_relative"), "High", "", 0.01))
    assert not ms["head_ahead_of_front_foot_release"].measured and not ms["release_ahead_of_front_foot"].measured


# ───────────────────────────── second clip ─────────────────────────────

CX = 640.0


def _second(mirror: bool = False) -> Kinematics:
    """Right-arm bowler seen facing the camera (left side on the image right). mirror=True: seen from behind."""
    xy = np.full((N, N_JOINTS, 2), np.nan)
    for side, dx in (("left", 20.0), ("right", -20.0)):
        xy[:, J[f"{side}_hip"]] = [CX + dx, 100.0]
        xy[:, J[f"{side}_knee"]] = [CX + dx, 50.0]
        xy[:, J[f"{side}_ankle"]] = [CX + dx, 0.0]
        xy[:, J[f"{side}_shoulder"]] = [CX + dx, 160.0]
        xy[:, J[f"{side}_elbow"]] = [CX + dx * 1.5, 130.0]
        xy[:, J[f"{side}_wrist"]] = [CX + dx * 1.5, 110.0]
        for p in ("eye", "ear"):
            xy[:, J[f"{side}_{p}"]] = [CX + dx * 0.3, 175.0]
    xy[:, J["nose"]] = [CX, 180.0]
    xy[REL, J["nose"]] = [CX + 30.0, 180.0]  # head out towards the non-bowling side at release
    xy[REL, J["right_wrist"]] = [CX - 40.0, 220.0]  # bowling arm 18.4° from vertical, towards the bowling side
    xy[REL, J["left_knee"]] = [CX + 10.0, 50.0]  # front knee 10 px inside the hip–ankle line
    t = np.arange(N)
    drift = np.clip(t / BFC, 0, 1) * 20.0  # body moves 20 px sideways over the run-up visible before BFC
    xy[..., 0] += drift[:, None]  # the whole body drifts
    if mirror:
        xy[..., 0] = 2 * CX - xy[..., 0]
    vis = np.full((N, N_JOINTS), 0.95)
    valid = np.isfinite(xy[..., 0])
    return Kinematics(fps=FPS, width=1280, height=720, xy=xy, vis=vis, valid=valid, img_xy=xy.copy(),
                      bowling_arm="right", direction="left_to_right")


@pytest.mark.parametrize("mirror,view", [(False, "front-on"), (True, "behind")])
def test_second_clip_known_geometry_same_from_front_or_behind(mirror, view):
    kin = _second(mirror)
    bh = _second_body_height(kin)
    ms = _by_key(front_metrics(kin, _events(), "High", "", "", view=view))
    assert all(m.view == view for m in ms.values())
    assert ms["head_lateral_offset_release"].value == pytest.approx((30.0 - 20.0) / bh[REL] * 100)  # vs front ankle
    assert ms["bowling_arm_angle_release"].value == pytest.approx(np.degrees(np.arctan2(20.0, 60.0)))
    assert ms["front_knee_inward_release"].value == pytest.approx(10.0 / (np.hypot(10, 50) * 2) * 100)
    # front foot lands 40 px towards the non-bowling side of the back foot: open (negative)
    assert ms["front_foot_alignment"].value == pytest.approx(-40.0 / bh[FFC] * 100)
    assert ms["runup_line_drift"].value == pytest.approx(20.0 / bh[BFC] * 100, rel=0.05)
    assert ms["trunk_lateral_flexion_release"].measured


def test_diagonal_second_clip_reports_only_robust_metrics_at_low():
    ms = _by_key(front_metrics(_second(), _events(), "High", "", "", view="diagonal"))
    for k in ("hip_shoulder_separation_bfc", "shoulder_counter_rotation", "front_foot_alignment",
              "front_knee_inward_release", "runup_line_drift"):
        assert not ms[k].measured and "diagonal" in ms[k].reason
    for k in ("trunk_lateral_flexion_release", "head_lateral_offset_release", "bowling_arm_angle_release"):
        assert ms[k].measured and ms[k].confidence == "Low" and "diagonal" in ms[k].confidence_reason


def test_overlapping_hips_mean_no_left_right():
    kin = _second()
    for f in range(N):
        kin.xy[f, J["left_hip"], 0] = kin.xy[f, J["right_hip"], 0] = CX
    ms = _by_key(front_metrics(kin, _events(), "High", "", "", view="front-on"))
    assert not ms["head_lateral_offset_release"].measured and "left and right" in ms["head_lateral_offset_release"].reason


# ───────────────────────────── combining the clips ─────────────────────────────

def _ev(b, f, r, conf="Medium"):
    return DeliveryEvents(EventResult("BFC", b, conf, "b", 1), EventResult("FFC", f, conf, "f", 1),
                          EventResult("release", r, conf, "r", 1))


def test_side_events_carried_into_second_clip_through_sync():
    m = map_events(_ev(60, 72, 80), 60.0, -0.5, 30.0, 200, "High", "sound")
    assert (m.bfc.frame, m.ffc.frame, m.release.frame) == (15, 21, 25)  # (60/60 − 0.5) × 30 = 15
    assert m.bfc.confidence == "Medium" and "sound sync" in m.bfc.reason
    assert map_events(_ev(60, 72, 80), 60.0, -5.0, 30.0, 200, "High", "sound") is None  # before the clip starts
    low = map_events(_ev(60, 72, 80, "High"), 60.0, 0.0, 60.0, 200, "Low", "sound")
    assert low.release.confidence == "Low"


def test_agreeing_views_raise_confidence():
    a, b = _ev(15, 21, 25), _ev(16, 21, 27)
    assert events_agree(a, b, 30.0) and not events_agree(a, _ev(15, 26, 25), 30.0)
    up = confirm_events(_ev(1, 2, 3, "Low"), "confirmed")
    assert up.ffc.confidence == "Medium" and "confirmed" in up.ffc.reason
    assert confirm_events(_ev(1, 2, 3, "High"), "x").bfc.confidence == "High"


# ───────────────────────────── camera angle ─────────────────────────────

def _scaled(pose: PoseSequence, start: float, end: float, travel: float) -> PoseSequence:
    """Same run-up seen from another angle: body size goes start→end, sideways travel scaled by `travel`."""
    kp = pose.kp.copy()
    n = kp.shape[0]
    s = np.linspace(start, end, n)[:, None]
    c = np.nanmean(kp[:, :, :2], axis=1)
    c0 = np.nanmean(c, axis=0)
    cc = c0 + (c - c0) * [travel, 0.0]
    kp[:, :, :2] = cc[:, None, :] + (kp[:, :, :2] - c[:, None, :]) * s[:, :, None]
    return PoseSequence(fps=pose.fps, width=pose.width, height=pose.height, kp=kp, device="cpu")


def test_camera_angle_classification():
    pose, _ = synth_pose()
    assert Q.classify_view(pose) == "side-on"
    assert Q.classify_view(_scaled(pose, 1.0, 0.1, 0.1)) == "behind"  # runs away from the camera (real clip: 0.33)
    assert Q.classify_view(_scaled(pose, 0.25, 1.5, 0.1)) == "front-on"  # runs at the camera
    assert Q.classify_view(_scaled(pose, 1.0, 0.3, 1.0)) == "diagonal"  # crosses the frame while moving away


def test_camera_following_the_bowler_is_still_side_on():
    """Broadcast replays follow the bowler, so he never crosses the frame; a thin shoulder profile with the feet
    spread wide across the picture in the stride is still side-on (this used to be rejected as the wrong angle)."""
    pose, _ = synth_pose()
    kp = pose.kp.copy()
    hip = np.nanmean(kp[:, [J["left_hip"], J["right_hip"]], 0], axis=1)
    kp[..., 0] -= (hip - np.nanmean(hip))[:, None]  # keep him centred, like a following camera
    follow = PoseSequence(fps=pose.fps, width=pose.width, height=pose.height, kp=kp, device="cpu")
    f = Q.view_features(follow)
    assert f["travel"] < 1.2 and f["stride_spread"] >= 0.5
    assert Q.classify_view(follow) == "side-on"

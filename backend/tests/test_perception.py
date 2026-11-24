"""Perception upgrade: ByteTrack identities, primary/secondary fusion, left/right identity correction with roles,
3D gating rules. No model downloads: the networks themselves are exercised in the slow/regression runs."""
import numpy as np
import pytest

from app.pipeline import identity, lift3d
from app.pipeline.fusion import fuse
from app.pipeline.metrics import Measurement
from app.pipeline.rtm import ByteTracker, halpe_to_ours
from app.pipeline.scale import estimate_scale
from app.pipeline.smooth import smooth_pose
from app.pipeline.types import DeliveryEvents, EventResult, J
from tests.synthetic import synth_pose


# ───────────────────────────── ByteTrack ─────────────────────────────

def test_bytetrack_keeps_identities_through_weak_detections():
    tr = ByteTracker(high=0.5, low=0.1, match_iou=0.3, max_lost=10)
    ids = {"a": set(), "b": set()}
    for f in range(30):
        a = [10 + 4 * f, 50, 60 + 4 * f, 200, 0.9 if f % 5 else 0.2]  # every 5th frame blurred: low score
        b = [400 - 3 * f, 60, 450 - 3 * f, 210, 0.9]
        out = tr.update(f, np.array([a, b], float))
        for tid, d in out:
            ids["a" if (d[0] < 250 and f < 20) or abs(d[0] - a[0]) < 1 else "b"].add(tid)
    assert len(ids["a"]) == 1 and len(ids["b"]) == 1 and ids["a"] != ids["b"]


def test_halpe_mapping_takes_heel_and_big_toe():
    kp = np.arange(52, dtype=float).reshape(26, 2)
    out = halpe_to_ours(kp, np.ones(26))
    assert tuple(out[J["left_heel"], :2]) == tuple(kp[24]) and tuple(out[J["right_foot_index"], :2]) == tuple(kp[21])


# ───────────────────────────── fusion ─────────────────────────────

def test_disagreeing_joints_lose_confidence_and_missing_ones_are_filled():
    p = np.zeros((3, 21, 3))
    p[..., 2] = 0.9
    s = p.copy()
    s[1, J["left_wrist"], :2] = 30.0  # 30 px away at a body height of 100 px = 0.3 BH: strong disagreement
    p[2, J["right_ankle"]] = [np.nan, np.nan, 0.0]  # primary missed it, secondary has it
    f, _d, summ = fuse(p, s, 100.0)
    assert f[1, J["left_wrist"], 2] < 0.3 and f[0, J["left_wrist"], 2] == pytest.approx(0.9)
    assert np.isfinite(f[2, J["right_ankle"], 0]) and summ["joints_filled_from_secondary"] == 1


# ───────────────────────────── identity ─────────────────────────────

def _swap_legs(kp, frames):
    for part in ("knee", "ankle", "heel", "foot_index"):
        a, b = J[f"left_{part}"], J[f"right_{part}"]
        kp[frames, a], kp[frames, b] = kp[frames, b].copy(), kp[frames, a].copy()


def test_short_leg_swap_is_undone_and_logged():
    pose, _ = synth_pose(fps=120, n=132, bfc=62, ffc=82, release=94)
    clean = pose.kp.copy()
    _swap_legs(pose.kp, np.arange(70, 76))  # inside the delivery stride: legs well apart
    fixed, log = identity.correct_swaps(pose.kp, 120.0, 400.0)
    ok = np.isfinite(clean[70:76, J["left_ankle"], 0])
    assert np.allclose(fixed[70:76, J["left_ankle"], :2][ok], clean[70:76, J["left_ankle"], :2][ok])
    assert log and log[0]["limb"] == "legs" and log[0]["frames"] == [70, 75] and "swapped back" in log[0]["reason"]


def test_clean_clip_gets_no_corrections():
    pose, _ = synth_pose(fps=120, n=132, bfc=62, ffc=82, release=94)
    _, log = identity.correct_swaps(pose.kp, 120.0, 400.0)
    assert log == []


def test_roles_follow_the_feet_not_the_labels():
    """Legs labelled the wrong way round for the whole clip: the role 'front leg' still goes to the foot that lands
    at FFC, so front-knee metrics use the right leg."""
    pose, truth = synth_pose(fps=120, n=132, bfc=62, ffc=82, release=94)
    _swap_legs(pose.kp, np.arange(pose.kp.shape[0]))
    kin = smooth_pose(pose, bowling_arm="right", direction="left_to_right")
    ev = DeliveryEvents(EventResult("BFC", truth.bfc, "High", "", 1), EventResult("FFC", truth.ffc, "High", "", 1),
                        EventResult("release", truth.release, "High", "", 1))
    roles = identity.assign_roles(kin, ev, "right", estimate_scale(kin, None).body_height_px)
    assert roles["front_leg"] == "right" and roles["back_leg"] == "left"  # labels swapped → roles swapped too
    import dataclasses
    kin2 = dataclasses.replace(kin, roles=roles)
    assert kin2.side("knee", "front") == "right_knee" and kin2.side("wrist", "bowling") == "right_wrist"


# ───────────────────────────── 3D gating ─────────────────────────────

def _m(key, value, conf="High"):
    return Measurement(key=key, value=value, unit="deg", view="side-on", frame=10, event="FFC", confidence=conf,
                       confidence_reason="ok", measured=value is not None)


def _events():
    return DeliveryEvents(EventResult("BFC", 5, "High", "", 1), EventResult("FFC", 10, "High", "", 1),
                          EventResult("release", 15, "High", "", 1))


def test_plane_offset_geometry():
    assert lift3d.plane_offset_deg(np.array([0, 0, 1.0])) == pytest.approx(0, abs=0.01)
    assert lift3d.plane_offset_deg(np.array([1.0, 0, 0])) == pytest.approx(90)


@pytest.mark.parametrize("v2,v3,off,expect_src,expect_conf", [
    (170.0, 165.0, 10.0, "2D side-on", "High"),   # plane near the picture, sources agree → 2D, High
    (170.0, 150.0, 10.0, "2D side-on", "Low"),    # agree-check fails → Low
    (170.0, 150.0, 40.0, "3D lifted", "Medium"),  # plane turned away: 2D foreshortened → 3D, Medium
    (None, 150.0, 40.0, "3D lifted", "Medium"),   # only 3D available
])
def test_gating_rules(v2, v3, off, expect_src, expect_conf, monkeypatch):
    monkeypatch.setitem(lift3d.P.angles.source, "front_knee_angle_ffc", "auto")
    meas = [_m("front_knee_angle_ffc", v2)]
    a3 = {"front_knee_angle_ffc": {"value": v3, "frame": 10, "plane_offset_deg": off, "event": "FFC"}}
    out, rep = lift3d.apply_sources(meas, a3, _events(), "side-on")
    m = out[0]
    assert rep["front_knee_angle_ffc"]["source"] == expect_src and m.confidence == expect_conf
    assert m.value == pytest.approx(v3 if expect_src == "3D lifted" else v2)
    assert m.confidence_reason.startswith(f"Source: {expect_src}")


def test_metric_with_3d_off_keeps_its_2d_value(monkeypatch):
    monkeypatch.setitem(lift3d.P.angles.source, "front_knee_angle_ffc", "off")
    meas = [_m("front_knee_angle_ffc", 170.0)]
    a3 = {"front_knee_angle_ffc": {"value": 140.0, "frame": 10, "plane_offset_deg": 40.0, "event": "FFC"}}
    out, rep = lift3d.apply_sources(meas, a3, _events(), "side-on")
    assert out[0] is meas[0] and "front_knee_angle_ffc" not in rep


def test_transverse_rotations_come_from_3d_even_without_a_second_clip():
    meas = [Measurement(key="hip_shoulder_separation_bfc", value=None, unit="deg", view="front-on", frame=None,
                        event="BFC", confidence=None, confidence_reason="Add a front-on clip", measured=False)]
    a3 = {"hip_shoulder_separation_bfc": {"value": 24.0, "frame": 5, "plane_offset_deg": 90.0, "event": "BFC"}}
    out, _ = lift3d.apply_sources(meas, a3, _events(), "side-on")
    assert out[0].measured and out[0].view == "3d" and out[0].value == pytest.approx(24.0)


def test_wrist_collapsing_onto_the_bowling_hand_is_dropped_but_side_on_hips_are_kept():
    pose, truth = synth_pose(fps=120, n=132, bfc=62, ffc=82, release=94)
    f = truth.release + 2
    lw, rw = J["left_wrist"], J["right_wrist"]
    pose.kp[f, lw, :2] = pose.kp[f, rw, :2] + 1.0  # the left label jumps onto the bowling hand
    out, log = identity.suppress_duplicates(pose.kp, 400.0)
    assert out[f, lw, 2] == 0.0 and out[f, rw, 2] > 0
    assert any(e["joint"] == "left_wrist" and e["frames"][0] == f for e in log)
    side = pose.kp.copy()
    side[:, J["left_hip"], :2] = side[:, J["right_hip"], :2]  # hips overlapping all clip long (side-on): genuine
    _out2, log2 = identity.suppress_duplicates(side, 400.0)
    assert not any(e["joint"].endswith("hip") for e in log2)


def test_3d_export_for_the_viewer_has_one_entry_per_frame_and_the_roles():
    import json
    pose, _truth = synth_pose(fps=120, n=132, bfc=62, ffc=82, release=94)
    kin = smooth_pose(pose, bowling_arm="right", direction="left_to_right")
    import dataclasses
    kin = dataclasses.replace(kin, roles={"front_leg": "left", "back_leg": "right"})
    xyz = np.zeros((kin.n, 17, 3))
    covered = np.ones(kin.n, bool)
    covered[:5] = False
    out = lift3d.export(lift3d.Lifted(xyz, covered), kin, DeliveryEvents(
        EventResult("BFC", 10, "High", "", 1), EventResult("FFC", 20, "High", "", 1), EventResult("release", 30, "High", "", 1)))
    assert len(out["frames"]) == kin.n and out["frames"][0] is None and len(out["frames"][5]) == 17
    assert out["roles"] == {"front_leg": "left", "back_leg": "right", "bowling_arm": "right"}
    assert all(a in out["joints"] and b in out["joints"] for a, b in out["bones"])
    json.dumps(out)  # serialisable


# ───────────────────────────── roles in every filming direction ─────────────────────────────

def _roles_for(pose, truth, direction, arm="right"):
    kin = smooth_pose(pose, bowling_arm=arm, direction=direction)
    ev = DeliveryEvents(EventResult("BFC", truth.bfc, "High", "", 1), EventResult("FFC", truth.ffc, "High", "", 1),
                        EventResult("release", truth.release, "High", "", 1))
    return kin, ev, identity.assign_roles(kin, ev, arm, estimate_scale(kin, None).body_height_px)


@pytest.mark.parametrize("direction", ["left_to_right", "right_to_left"])
@pytest.mark.parametrize("arm", ["right", "left"])
def test_front_leg_is_found_from_either_side_of_the_pitch(direction, arm):
    """Right- and left-arm bowlers filmed from both sides (so the front leg is sometimes the far, hidden one)."""
    pose, truth = synth_pose(direction=direction, arm=arm)
    _kin, _ev, roles = _roles_for(pose, truth, direction, arm)
    front = "left" if arm == "right" else "right"
    assert roles["front_leg"] == front and roles["confidence"] in ("High", "Medium") and roles["relabel"] is None


def test_front_leg_from_position_when_the_hidden_foot_gives_no_landing_signal():
    """Filmed from the bowling-arm side the front foot is behind the back leg at FFC; the pose model then often holds
    the occluded foot still (no landing speed drop). Being ahead along the run still identifies it."""
    pose, truth = synth_pose(direction="right_to_left")
    lf = [J[f"left_{p}"] for p in ("ankle", "heel", "foot_index")]
    w = 12
    pose.kp[truth.ffc - w:truth.ffc + 1, lf, :2] = pose.kp[truth.ffc, lf, :2]  # frozen in the frames before landing
    _kin, _ev, roles = _roles_for(pose, truth, "right_to_left")
    assert roles["front_leg"] == "left"


def test_labels_flipped_between_bfc_and_ffc_are_swapped_back():
    """An uncaught label flip as the legs pass each other between BFC and FFC: the labels at BFC then point the other
    way to those at FFC. The role check finds it and swaps the earlier frames back."""
    pose, truth = synth_pose(direction="right_to_left")
    clean, ev, _r0 = _roles_for(pose, truth, "right_to_left")
    cross = truth.bfc + 8  # the flip lands a few frames after BFC, while the front leg swings through
    flipped = identity.swap_leg_labels(clean, 0, cross)  # what the smoothed skeleton looks like after the flip
    roles = identity.assign_roles(flipped, ev, "right", estimate_scale(clean, None).body_height_px)
    assert roles["front_leg"] == "left" and roles["relabel"] is not None
    s0, s1 = roles["relabel"]
    assert s0 <= truth.bfc < s1 <= truth.ffc
    fixed = identity.swap_leg_labels(flipped, s0, s1)
    assert fixed.identity_log[-1]["reason"].startswith("labels flipped")
    w = 6  # at BFC the right (back) foot is the planted one again
    vr = np.linalg.norm(np.diff(fixed.joint("right_ankle")[truth.bfc:truth.bfc + w], axis=0), axis=1).mean()
    vl = np.linalg.norm(np.diff(fixed.joint("left_ankle")[truth.bfc:truth.bfc + w], axis=0), axis=1).mean()
    assert vr < vl


# ───────────────────────────── 2D-anchored 3D ─────────────────────────────

def test_anchored_3d_keeps_the_2d_picture_and_restores_foreshortened_limbs():
    """x/y come from the video's 2D keypoints exactly; a shin that MotionBERT says points toward the camera (shorter in
    2D than its length elsewhere) gets that depth back, so its 3D length is constant."""
    T = 40
    H = lift3d.HJ
    x2 = np.zeros((T, 17, 2))
    x2[:, H["r_hip"]] = [10, 0]
    x2[:, H["l_hip"]] = [-10, 0]
    x2[:, H["r_knee"]] = [10, 50]
    x2[:, H["l_knee"]] = [-10, 50]
    x2[:, H["r_ankle"]] = [10, 100]
    x2[:, H["l_ankle"]] = [-10, 100]
    for name, xy in (("spine", [0, -40]), ("thorax", [0, -80]), ("nose", [0, -100]), ("head", [0, -115]),
                     ("l_shoulder", [-20, -80]), ("r_shoulder", [20, -80]), ("l_elbow", [-20, -40]),
                     ("r_elbow", [20, -40]), ("l_wrist", [-20, 0]), ("r_wrist", [20, 0])):
        x2[:, H[name]] = xy
    short = slice(20, 30)  # right shin pointing toward the camera: 30 px in 2D instead of 50
    x2[short, H["r_ankle"]] = [10, 80]
    mb = np.zeros((T, 17, 3))
    mb[..., :2] = x2 / 100.0  # MotionBERT agrees on x/y (its own units) …
    mb[short, H["r_ankle"], 2] = -0.4  # … and says the ankle came toward the camera there
    conf = np.ones((T, 17))
    out = lift3d.anchor_to_2d(mb, x2, conf, np.ones(T, bool), fps=50.0)
    assert np.allclose(out[..., :2], x2)  # the 2D picture is kept exactly
    shin = np.linalg.norm(out[:, H["r_ankle"]] - out[:, H["r_knee"]], axis=1)
    assert shin[25] == pytest.approx(50.0, abs=4) and out[25, H["r_ankle"], 2] < out[25, H["r_knee"], 2]
    assert np.allclose(out[5, :, 2], 0.0, atol=1e-6)  # a flat frame stays flat: its 3D angles equal the 2D ones

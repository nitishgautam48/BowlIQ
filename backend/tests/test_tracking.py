"""Phase 1: the right person is analysed (tap-to-select, appearance memory, ignore zones, ranked retry) and the
upload API validates the user's guidance."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.routes import parse_tap, parse_zones
from app.core.errors import BowlIQError
from app.pipeline.orchestrator import _selection
from app.pipeline.tracking import Det, RawDetections, appearance_distance, candidate_count, select
from app.pipeline.types import BowlerSelection, J
from tests.apiclient import ensure_sample, upload
from tests.synthetic import synth_pose

NB = 48  # bins per body part in the appearance signature


def _desc(hue_bin: int) -> np.ndarray:
    h = np.zeros(NB)
    h[hue_bin] = 1.0
    return np.concatenate([h, h])


RED, BLUE = _desc(0), _desc(30)


def _bystander(pose, x: float, size: float = 1.0) -> np.ndarray:
    """A person standing still at horizontal pixel x (the bowler's first frame, shifted and scaled)."""
    f0 = int(np.where(np.isfinite(pose.kp[:, J["left_hip"], 0]))[0][0])
    kp = pose.kp[f0].copy()
    c = np.nanmean(kp[:, :2], axis=0)
    kp[:, :2] = (kp[:, :2] - c) * size + np.array([x, c[1]])
    return kp


def _scene(walker_desc=RED, stander_desc=BLUE, stander_x=200.0, stander_size=1.0):
    pose, _truth = synth_pose()
    stand = _bystander(pose, stander_x, stander_size)
    dets = []
    for f in range(pose.kp.shape[0]):
        frame = [Det(stand.copy(), desc=stander_desc)]
        if np.isfinite(pose.kp[f, J["left_hip"], 0]):
            frame.append(Det(pose.kp[f].copy(), desc=walker_desc))
        dets.append(frame)
    return RawDetections(dets, pose.fps, pose.width, pose.height, "cpu"), pose, stand


def _is_bowler(seq, pose) -> bool:
    ok = np.isfinite(seq.kp[:, J["left_hip"], 0]) & np.isfinite(pose.kp[:, J["left_hip"], 0])
    return ok.sum() > 0.8 * np.isfinite(pose.kp[:, J["left_hip"], 0]).sum() and np.allclose(
        seq.kp[ok, J["left_hip"], :2], pose.kp[ok, J["left_hip"], :2])


def test_appearance_distance():
    assert appearance_distance(RED, RED) == pytest.approx(0.0)
    assert appearance_distance(RED, BLUE) == pytest.approx(1.0)
    assert appearance_distance(RED, None) is None


def test_auto_picks_the_runner_over_a_still_bystander():
    raw, pose, _ = _scene()
    seq = select(raw)
    assert seq.selected_by == "auto" and _is_bowler(seq, pose)
    assert seq.people_detected_max == 2


def test_tap_picks_the_tapped_person():
    raw, pose, stand = _scene()
    f = 70
    hip = pose.kp[f, J["left_hip"], :2]
    seq = select(raw, BowlerSelection(tap=(f, float(hip[0]), float(hip[1]))))
    assert seq.selected_by == "tap" and _is_bowler(seq, pose)
    # tapping the bystander follows the bystander (who then fails the bowling checks downstream)
    c = np.nanmean(stand[:, :2], axis=0)
    seq2 = select(raw, BowlerSelection(tap=(f, float(c[0]), float(c[1]))))
    assert np.allclose(np.nanmedian(seq2.kp[:, J["left_hip"], 0]), stand[J["left_hip"], 0])


def test_tap_on_empty_space_is_rejected():
    raw, _pose, _ = _scene()
    with pytest.raises(BowlIQError) as e:
        select(raw, BowlerSelection(tap=(70, 5.0, 5.0)))
    assert e.value.code == "BOWLER_NOT_FOUND_AT_TAP"


def test_appearance_stops_the_track_jumping_to_someone_who_looks_different():
    """The bowler runs right past a bystander: without appearance the nearest person could be taken."""
    pose, _ = synth_pose()
    mid = 80
    hip_x = float(pose.kp[mid, J["left_hip"], 0])
    raw, _, stand = _scene(stander_x=hip_x + 15)
    seq = select(raw)
    assert _is_bowler(seq, pose)
    assert not np.any(np.isclose(seq.kp[:, J["left_hip"], 0], stand[J["left_hip"], 0]))


def test_ignore_zone_drops_people_inside_it():
    raw, pose, stand = _scene()
    xs, ys = stand[:, 0], stand[:, 1]
    zone = (float(np.nanmin(xs)) - 10, float(np.nanmin(ys)) - 10, float(np.nanmax(xs)) + 10, float(np.nanmax(ys)) + 10)
    assert candidate_count(raw) >= 1
    seq = select(raw, BowlerSelection(ignore=[zone]))
    assert _is_bowler(seq, pose)
    assert seq.people_detected_max == 2  # still reported (raw), but never analysed


def test_ranked_retry_reaches_the_next_candidate_then_stops():
    raw, pose, _ = _scene()
    first = select(raw, rank=0)
    assert _is_bowler(first, pose)
    with pytest.raises(BowlIQError) as e:
        for r in range(1, 4):
            s = select(raw, rank=r)
            assert s.selected_by == "auto-retry" and not _is_bowler(s, pose)
    assert e.value.code == "POSE_NOT_FOUND"


def test_selection_maps_normalised_guidance_to_pixels():
    sel = _selection([2.5, 0.5, 0.25], [[0.8, 0.9, 0.6, 0.1]], 1.0, 100.0, 1000, 800)
    assert sel.tap == (150, 500.0, 200.0)
    assert sel.ignore == [(600.0, 80.0, 800.0, 720.0)]
    assert _selection(None, [], 0.0, 30.0, 10, 10) is None


def test_parse_tap_and_zones():
    assert parse_tap("1.5, 0.2, 0.9", "Tap") == [1.5, 0.2, 0.9]
    assert parse_tap("", "Tap") is None
    assert parse_zones("0.6,0.1,0.9,0.5; 0.5,0.5,0.1,0.9", "Zone") == [[0.6, 0.1, 0.9, 0.5], [0.1, 0.5, 0.5, 0.9]]
    for bad in ("1,2", "a,b,c", "1,nan,0.5", "-1,0.5,0.5", "1,1.5,0.5"):
        with pytest.raises(BowlIQError):
            parse_tap(bad, "Tap")
    for bad in ("0,0,1", "0,0,1,2", "0.1,0.1,0.105,0.5", "0,0,1,1", ";".join(["0,0,0.1,0.1"] * 9)):
        with pytest.raises(BowlIQError):
            parse_zones(bad, "Zone")


@pytest.fixture(scope="module")
def client():
    ensure_sample()
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_api_rejects_bad_guidance(client):
    assert upload(client, side_tap="0.5,2,0.5").json()["code"] == "INVALID_REQUEST"
    assert upload(client, side_ignore="0,0,1,1").json()["code"] == "INVALID_REQUEST"
    r = upload(client, side_tap="999,0.5,0.5")
    assert r.status_code == 400 and "trimmed" in r.json()["message"]


def test_joints_pinned_to_the_frame_edge_are_not_visible():
    """Real behind-view clip: the bowler entered at the left edge and the model put his hidden shoulder at x = 1 px."""
    raw, _pose, _ = _scene()
    for frame in raw.dets:
        for d in frame:
            if d.desc is RED:
                d.kp[J["left_shoulder"], 0] = 1.0
    seq = select(raw)
    ok = np.isfinite(seq.kp[:, J["left_shoulder"], 0])
    assert ok.any() and np.all(seq.kp[ok, J["left_shoulder"], 2] == 0.0)
    assert np.all(seq.kp[ok, J["right_shoulder"], 2] > 0)

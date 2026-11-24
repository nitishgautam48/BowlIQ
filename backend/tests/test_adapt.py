"""Phase 1c: adapting to imperfect video (camera movement, small bowler)."""
import cv2
import numpy as np
import pytest

from app.pipeline.adapt import apply_path, estimate_camera_path
from app.pipeline.orchestrator import _correction_helps, bowler_px
from app.pipeline.types import N_JOINTS, DeliveryEvents, EventResult, J, Kinematics, PoseSequence

W, H, N, FPS = 320, 240, 40, 30.0


def _scene_video(path, shift_per_frame=(3.0, 1.0), turn_per_frame=0.3):
    """Textured background seen by a camera that pans and turns, with a 'person' blob moving the other way."""
    rng = np.random.default_rng(0)
    big = cv2.GaussianBlur((rng.random((900, 1200)) * 255).astype(np.uint8), (0, 0), 2.5)
    big = cv2.cvtColor(cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    truth = []
    for f in range(N):
        cx, cy = 600 + shift_per_frame[0] * f, 450 + shift_per_frame[1] * f
        M = cv2.getRotationMatrix2D((cx, cy), turn_per_frame * f, 1.0)
        M[:, 2] += [W / 2 - cx, H / 2 - cy]
        frame = cv2.warpAffine(big, M, (W, H))
        px = int(250 - 4 * f)
        cv2.rectangle(frame, (px, 80), (px + 30, 200), (0, 0, 255), -1)  # the person (masked out)
        out.write(frame)
        truth.append(np.vstack([M, [0, 0, 1]]))  # big-image → frame f
    out.release()
    return truth, [[(250 - 4 * f - 20, 60, 250 - 4 * f + 50, 220)] for f in range(N)]


def test_camera_path_recovers_pan_and_turn(tmp_path):
    truth, boxes = _scene_video(tmp_path / "pan.avi")
    path = estimate_camera_path(tmp_path / "pan.avi", boxes, ref=20)
    assert path.ok and path.moved and path.failed_frames == 0
    # a fixed world point seen in frame f must land where the reference frame saw it
    world = np.array([620.0, 470.0, 1.0])
    ref_px = (truth[20] @ world)[:2]
    for f in (0, 10, 39):
        px = (truth[f] @ world)[:2]
        mapped = apply_path(px[None, None, :], type(path)(path.to_ref[f:f + 1], 0, True, True))[0, 0]
        assert np.linalg.norm(mapped - ref_px) < 2.0
    assert path.max_turn_deg == pytest.approx(0.3 * 20, abs=1.0)


def test_steady_camera_is_left_alone(tmp_path):
    _, boxes = _scene_video(tmp_path / "still.avi", (0.0, 0.0), 0.0)
    path = estimate_camera_path(tmp_path / "still.avi", boxes, ref=20)
    assert not path.moved and path.max_shift_px < 1.5


def _kin(front_world_drift: float, front_raw_drift: float) -> tuple[Kinematics, DeliveryEvents]:
    n = 60
    xy = np.zeros((n, N_JOINTS, 2))
    img = np.zeros((n, N_JOINTS, 2))
    a = J["left_ankle"]
    xy[30:50, a, 0] = np.linspace(0, front_world_drift, 20)
    img[30:50, a, 0] = np.linspace(0, front_raw_drift, 20)
    kin = Kinematics(fps=60.0, width=640, height=480, xy=xy, vis=np.ones((n, N_JOINTS)), valid=np.ones((n, N_JOINTS), bool),
                     img_xy=img, bowling_arm="right", direction="left_to_right")
    ev = DeliveryEvents(EventResult("BFC", 20, "High", "", 1), EventResult("FFC", 30, "High", "", 1),
                        EventResult("release", 49, "High", "", 1))
    return kin, ev


def test_correction_kept_only_if_the_planted_foot_is_stiller():
    assert _correction_helps(*_kin(front_world_drift=2.0, front_raw_drift=40.0))  # pan undone: foot still
    assert not _correction_helps(*_kin(front_world_drift=30.0, front_raw_drift=5.0))  # correction made it worse


def test_bowler_size():
    kp = np.full((20, N_JOINTS, 3), np.nan)
    kp[:, J["nose"]] = [100, 100, 1]
    kp[:, J["left_shoulder"]] = kp[:, J["right_shoulder"]] = [100, 115, 1]
    kp[:, J["left_ankle"]] = kp[:, J["right_ankle"]] = [100, 160, 1]
    pose = PoseSequence(fps=30, width=640, height=480, kp=kp, device="cpu", active_span=(0, 19))
    assert bowler_px(pose) == pytest.approx(60.0)

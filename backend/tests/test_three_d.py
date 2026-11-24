"""Phase 4: true 3D trunk rotation from two synced views, checked against a 3D bowler with known rotations."""
import numpy as np
import pytest

from app.pipeline.metrics import front_metrics
from app.pipeline.three_d import counter_rotation, reconstruct, separation_at
from app.pipeline.types import N_JOINTS, DeliveryEvents, EventResult, J, Kinematics

SIDE_FPS, SEC_FPS, OFFSET = 60.0, 30.0, 0.5  # the second phone started 0.5 s later
BFC_T, FFC_T, REL_T = 2.0, 2.2, 2.35


def theta_sh(t):  # shoulders: −20° at BFC, smoothly counter-rotate ~25° around 2.12 s, then open towards release
    t = np.asarray(t, float)
    return -20 - 25 * np.exp(-(((t - 2.12) / 0.05) ** 2)) + 100 / (1 + np.exp(-(t - 2.27) / 0.025))


def theta_hip(t):  # hips 10° at BFC → hip–shoulder separation 30°
    return np.interp(t, [0, BFC_T, REL_T, 4], [10, 10, 70, 70])


def _bowler(t: float) -> dict:
    """3D joints (X along the pitch, Y up, Z towards the bowler's left = front side for a right-armer), metres."""
    p = {}
    for part, half, y, th in (("shoulder", 0.2, 1.5, theta_sh(t)), ("hip", 0.15, 1.0, theta_hip(t))):
        u = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])  # (X, Z) from right (bowling) to left
        p[f"left_{part}"] = np.array([half * u[0], y, half * u[1]])
        p[f"right_{part}"] = np.array([-half * u[0], y, -half * u[1]])
    for s, z in (("left", 0.12), ("right", -0.12)):
        p[f"{s}_knee"] = np.array([0.0, 0.5, z])
        p[f"{s}_ankle"] = np.array([0.0, 0.0, z])
        p[f"{s}_elbow"] = np.array([0.0, 1.2, 2 * z])
        p[f"{s}_wrist"] = np.array([0.0, 1.0, 2 * z])
    p["nose"] = np.array([0.05, 1.7, 0.0])
    run = 5.0 * min(t - FFC_T, 0.0) + 1.0 * max(t - FFC_T, 0.0)  # 5 m/s run-up along the pitch, slowing after FFC
    return {k: v + np.array([run, 0.0, 0.0]) for k, v in p.items()}


def _kin(n, fps, project) -> Kinematics:
    xy = np.full((n, N_JOINTS, 2), np.nan)
    for f in range(n):
        for name, P in _bowler(project.time(f)).items():
            xy[f, J[name]] = project(P, f)
    valid = np.isfinite(xy[..., 0])
    return Kinematics(fps=fps, width=1280, height=720, xy=xy, vis=np.full((n, N_JOINTS), 0.95), valid=valid,
                      img_xy=xy.copy(), bowling_arm="right", direction="left_to_right")


class Side:
    def time(self, f):
        return f / SIDE_FPS

    def __call__(self, P, f):
        return np.array([640 + 120 * P[0], 120 * P[1]])


class Behind:
    """Behind the bowler, 8° off square, bowler shrinking as he runs away (perspective), 1 px noise."""
    rng = np.random.default_rng(0)

    def time(self, j):
        return j / SEC_FPS - OFFSET

    def __call__(self, P, j):
        a = np.radians(8)
        s = 280 / (1 + 0.004 * j)
        x = (-P[2] * np.cos(a) + P[0] * np.sin(a)) * s  # bowler's left appears on the image left from behind
        return np.array([640 + x, P[1] * s]) + self.rng.normal(0, 1.0, 2)


def _ev(fps, bfc_t=BFC_T, ffc_t=FFC_T, rel_t=REL_T):
    return DeliveryEvents(EventResult("BFC", round(bfc_t * fps), "High", "", 1),
                          EventResult("FFC", round(ffc_t * fps), "High", "", 1),
                          EventResult("release", round(rel_t * fps), "High", "", 1))


def _truth(ev):
    t = np.arange(ev.bfc.frame, ev.ffc.frame + 1) / SIDE_FPS
    sh = theta_sh(t)
    return abs(float(theta_sh(BFC_T) - theta_hip(BFC_T))), float(np.max(np.abs(sh - sh[0])))


class Square(Behind):
    def __call__(self, P, j):
        s = 280 / (1 + 0.004 * j)
        return np.array([640 - P[2] * s, P[1] * s]) + self.rng.normal(0, 1.0, 2)


class Turned20(Behind):
    def __call__(self, P, j):
        a, s = np.radians(20), 280 / (1 + 0.004 * j)
        return np.array([640 + (-P[2] * np.cos(a) + P[0] * np.sin(a)) * s, P[1] * s]) + self.rng.normal(0, 1.0, 2)


@pytest.mark.parametrize("camera,tol", [(Square, 2.5), (Behind, 3.0), (Turned20, 3.0)])
def test_3d_rotation_matches_the_true_values(camera, tol):
    """Within 3° of the truth with the second camera square, 8° or 20° off square (the turn is measured from the
    run-up and undone; without that, 8° off square cost 5° of counter-rotation)."""
    side, sec = _kin(200, SIDE_FPS, Side()), _kin(150, SEC_FPS, camera())
    ev = _ev(SIDE_FPS)
    rot = reconstruct(side, sec, OFFSET, ev)  # front_t = side_t + offset
    assert rot.reason is None and rot.yaw_deg is not None
    hss, cr_true = _truth(ev)
    assert separation_at(rot, ev.bfc.frame) == pytest.approx(hss, abs=tol)
    cr, f = counter_rotation(rot, ev.bfc.frame, ev.ffc.frame)
    assert cr == pytest.approx(cr_true, abs=tol)
    assert abs(f / SIDE_FPS - 2.12) < 0.05


def test_single_camera_cannot_see_it_but_3d_can():
    """Shoulders square to the pitch at BFC overlap in the behind view (2D: not measured); 3D still measures them."""
    sec = _kin(150, SEC_FPS, Behind())
    sec_ev = _ev(SEC_FPS, BFC_T + OFFSET, FFC_T + OFFSET, REL_T + OFFSET)
    two_d = {m.key: m for m in front_metrics(sec, sec_ev, "High", "", "", view="behind")}
    assert not two_d["hip_shoulder_separation_bfc"].measured or abs(two_d["hip_shoulder_separation_bfc"].value - 30) > 5


def test_wrong_offset_or_missing_coverage_is_refused():
    side, sec = _kin(200, SIDE_FPS, Side()), _kin(60, SEC_FPS, Behind())  # second clip ends before release
    assert reconstruct(side, sec, OFFSET, _ev(SIDE_FPS)).reason is not None

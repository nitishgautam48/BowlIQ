"""3D lifting with MotionBERT (Zhu et al., ICCV 2023; pretrained on Human3.6M, lite model fine-tuned for 3D pose).

The corrected 2D sequence (after identity correction and smoothing) is mapped to the Human3.6M 17-joint layout,
resampled to the 50 fps MotionBERT was trained at (in REAL time, so slow-motion clips are handled), normalised the
way the authors do for in-the-wild video (whole-sequence box → [-1, 1]), lifted in windows of up to 243 frames and
resampled back to the clip's frames. The output is in camera coordinates (x right, y down, z away from the camera),
root-relative, in normalised units — fine for angles, which are scale-free.

Joint angles are then computed in 3D with a body frame: up = −y (camera roughly level), forward = the horizontal
direction from the back foot to the front foot (the line of the run), lateral = up × forward.

Weights: the official checkpoint from huggingface.co/walterzhu/MotionBERT, downloaded once and checked by SHA-256.
Needs PyTorch (CPU is fine: ~1–2 s per clip; CUDA FP16 when available). Without it the analysis keeps 2D angles.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings
from app.core.errors import BowlIQError
from app.pipeline.params import P
from app.pipeline.three_d import Rotation3D, counter_rotation, separation_at
from app.pipeline.types import J, Kinematics

log = logging.getLogger("bowliq.lift3d")
L3 = P.lifting
CKPT_URL = ("https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/"
            "FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin")
CKPT_SHA = "9811155371db4ca5d20f31a36a232d41012e12e1333882888a564d741861148f"
H36M = ["pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "spine", "thorax", "nose", "head",
        "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist"]
HJ = {n: i for i, n in enumerate(H36M)}
_lock = threading.Lock()
_model = None


def available() -> bool:
    try:
        import torch  # noqa: F401
        return bool(L3.enabled)
    except ImportError:
        return False


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ensure_checkpoint() -> Path:
    d = settings.models_dir / "motionbert"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "FT_MB_lite_MB_ft_h36m_global_lite.bin"
    if p.exists() and p.stat().st_size > 10_000_000:
        return p
    with _lock:
        if p.exists():
            return p
        tmp = p.with_suffix(".part")
        log.info("Downloading the MotionBERT checkpoint")
        try:
            with urllib.request.urlopen(CKPT_URL, timeout=600) as r, open(tmp, "wb") as f:  # nosec B310 (fixed https URL)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise BowlIQError("MODEL_DOWNLOAD_FAILED", "Could not download the 3D lifting model.",
                              "Check the internet connection once; 2D angles are used meanwhile.", 500) from exc
        if _sha(tmp) != CKPT_SHA:
            tmp.unlink(missing_ok=True)
            raise BowlIQError("MODEL_INVALID", "The 3D lifting model failed its checksum.", "Delete backend/models/motionbert.", 500)
        tmp.replace(p)
    return p


def _load():
    global _model
    if _model is None:
        import torch

        from app.pipeline.motionbert.DSTformer import DSTformer
        m = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5, num_heads=8, mlp_ratio=4, num_joints=17,
                      maxlen=243, att_fuse=True)
        ck = torch.load(ensure_checkpoint(), map_location="cpu", weights_only=True)["model_pos"]
        m.load_state_dict({k.removeprefix("module."): v for k, v in ck.items()}, strict=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = m.to(dev).eval()
        _model = (m, dev)
    return _model


# ───────────────────────────── 2D → H36M ─────────────────────────────

def to_h36m(kin: Kinematics) -> tuple[np.ndarray, np.ndarray]:
    """(F, 17, 2) image coordinates and (F, 17) confidence in the Human3.6M layout. Missing joints are interpolated
    over time (confidence 0 there)."""
    xy, vis, valid = kin.img_xy, kin.vis, kin.valid
    n = kin.n

    def g(name: str) -> tuple[np.ndarray, np.ndarray]:
        j = J[name]
        c = np.where(valid[:, j], np.clip(vis[:, j], 0, 1), 0.0)
        return xy[:, j], c

    def mid(a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
        (pa, ca), (pb, cb) = g(a), g(b)
        return 0.5 * (pa + pb), np.minimum(ca, cb)

    out = np.full((n, 17, 2), np.nan)
    conf = np.zeros((n, 17))
    pel, cp = mid("left_hip", "right_hip")
    hn = kin.head_neck
    if hn is not None and np.isfinite(hn[:, 1]).any():
        neck, cn = hn[:, 1], np.where(np.isfinite(hn[:, 1, 0]), 0.8, 0.0)
        head, ch = hn[:, 0], np.where(np.isfinite(hn[:, 0, 0]), 0.8, 0.0)
    else:
        neck, cn = mid("left_shoulder", "right_shoulder")
        nose, cno = g("nose")
        head, ch = nose + L3.head_from_nose * (nose - neck), np.minimum(cno, cn)
    rows = {"pelvis": (pel, cp), "r_hip": g("right_hip"), "r_knee": g("right_knee"), "r_ankle": g("right_ankle"),
            "l_hip": g("left_hip"), "l_knee": g("left_knee"), "l_ankle": g("left_ankle"),
            "spine": (0.5 * (pel + neck), np.minimum(cp, cn)), "thorax": (neck, cn), "nose": g("nose"),
            "head": (head, ch), "l_shoulder": g("left_shoulder"), "l_elbow": g("left_elbow"), "l_wrist": g("left_wrist"),
            "r_shoulder": g("right_shoulder"), "r_elbow": g("right_elbow"), "r_wrist": g("right_wrist")}
    for name, (p, c) in rows.items():
        out[:, HJ[name]] = p
        conf[:, HJ[name]] = np.where(np.isfinite(p[:, 0]), c, 0.0)
    idx = np.arange(n)
    for j in range(17):
        for d in range(2):
            ok = np.isfinite(out[:, j, d])
            if ok.sum() >= 2:
                out[:, j, d] = np.interp(idx, idx[ok], out[ok, j, d])
    return out, conf


def _normalise(x: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """MotionBERT's in-the-wild normalisation (utils_data.crop_scale): whole-sequence box → [-1, 1]."""
    ok = conf > 0
    pts = x[ok] if ok.any() else x.reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    s = max(x1 - x0, y1 - y0, 1e-6)
    xs, ys = (x0 + x1 - s) / 2, (y0 + y1 - s) / 2
    out = (x - np.array([xs, ys])) / s
    return (out - 0.5) * 2


@dataclass
class Lifted:
    xyz: np.ndarray  # (F, 17, 3) camera coords, H36M order, NaN where not lifted
    covered: np.ndarray  # (F,) bool: frames inside the bowler's span


def lift(kin: Kinematics) -> Optional[Lifted]:
    """3D pose for every frame of the bowler's span, or None when there isn't enough of the body to lift."""
    import torch

    x2, conf = to_h36m(kin)
    span = np.where((conf > 0).sum(axis=1) >= L3.min_joints)[0]
    if len(span) < 3 or not np.isfinite(x2).all():
        return None
    a, b = int(span[0]), int(span[-1])
    t = np.arange(a, b + 1) / kin.fps  # real time (slow motion already folded into kin.fps)
    t50 = np.arange(t[0], t[-1] + 1e-9, 1.0 / L3.model_fps)
    if len(t50) < 2:
        return None
    seq = np.empty((len(t50), 17, 3))
    for j in range(17):
        for d in range(2):
            seq[:, j, d] = np.interp(t50, t, x2[a:b + 1, j, d])
        seq[:, j, 2] = np.interp(t50, t, conf[a:b + 1, j])
    seq[..., :2] = _normalise(seq[..., :2], seq[..., 2])
    m, dev = _load()
    win = int(L3.window)
    T = len(seq)
    acc = np.zeros((T, 17, 3))
    cnt = np.zeros((T, 1, 1))
    starts = list(range(0, max(1, T - win + 1), max(1, win // 2)))
    if starts[-1] + win < T:
        starts.append(T - win)
    with torch.no_grad():
        for s0 in starts:
            chunk = seq[s0:s0 + win]
            inp = torch.tensor(chunk[None], dtype=torch.float32, device=dev)
            if dev == "cuda" and L3.fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = m(inp)
            else:
                out = m(inp)
            o = out[0].float().cpu().numpy()
            acc[s0:s0 + len(chunk)] += o
            cnt[s0:s0 + len(chunk)] += 1
    y50 = acc / np.maximum(cnt, 1)
    y50 = y50 - y50[:, :1]  # root-relative
    xyz = np.full((kin.n, 17, 3), np.nan)
    for j in range(17):
        for d in range(3):
            xyz[a:b + 1, j, d] = np.interp(t, t50, y50[:, j, d])
    covered = np.zeros(kin.n, bool)
    covered[a:b + 1] = True
    return Lifted(xyz, covered)


BONES = [("pelvis", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("pelvis", "l_hip"), ("l_hip", "l_knee"),
         ("l_knee", "l_ankle"), ("pelvis", "spine"), ("spine", "thorax"), ("thorax", "nose"), ("nose", "head"),
         ("thorax", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("thorax", "r_shoulder"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist")]


def export(L: Lifted, kin: Kinematics, events) -> dict:
    """The lifted pose for the 3D viewer (storage/debug/{id}/pose3d.json): camera coordinates (x right, y down, z away
    from the camera), pelvis at the origin, one entry per video frame (null outside the lifted span)."""
    frames = [None if not (L.covered[f] and np.isfinite(L.xyz[f]).all()) else np.round(L.xyz[f], 4).tolist()
              for f in range(kin.n)]
    roles = kin.roles or {}
    return {"fps": float(kin.fps), "n": int(kin.n), "joints": H36M, "bones": BONES,
            "roles": {"front_leg": roles.get("front_leg"), "back_leg": roles.get("back_leg"),
                      "bowling_arm": kin.bowling_arm},
            "events": {"bfc": int(events.bfc.frame), "ffc": int(events.ffc.frame), "release": int(events.release.frame)},
            "frames": frames}


# ───────────────────────────── 3D angles ─────────────────────────────

def _ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))


def _hj(side: str, part: str) -> int:
    return HJ[f"{side[0]}_{part}"]


def body_frame(L: Lifted, kin: Kinematics, events) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(up, forward, lateral) unit vectors: up = −y; forward = horizontal back-foot → front-foot direction (the run)."""
    fl, bl = kin.side("ankle", "front").split("_")[0], kin.side("ankle", "back").split("_")[0]
    p_f = L.xyz[events.ffc.frame, _hj(fl, "ankle")]
    p_b = L.xyz[events.bfc.frame, _hj(bl, "ankle")] if np.isfinite(L.xyz[events.bfc.frame]).all() else L.xyz[events.ffc.frame, _hj(bl, "ankle")]
    up = np.array([0.0, -1.0, 0.0])
    fwd = p_f - p_b
    if not np.all(np.isfinite(fwd)):
        return None
    fwd = fwd - up * np.dot(fwd, up)
    if np.linalg.norm(fwd) < 1e-6:
        return None
    fwd /= np.linalg.norm(fwd)
    lat = np.cross(up, fwd)
    return up, fwd, lat / np.linalg.norm(lat)


def plane_offset_deg(normal: np.ndarray) -> float:
    """How far a plane is turned from the image plane: 0° = parallel to the picture (2D sees it truly), 90° = edge-on."""
    n = normal / (np.linalg.norm(normal) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1))))


def angles(L: Lifted, kin: Kinematics, events) -> dict[str, dict]:
    """{metric key: {value, frame, plane_offset_deg, event}} for the angle metrics measurable in 3D."""
    out: dict[str, dict] = {}
    bf = body_frame(L, kin, events)
    fl, bl = kin.side("knee", "front").split("_")[0], kin.side("knee", "back").split("_")[0]
    arm = kin.bowling_arm
    other = "left" if arm == "right" else "right"
    X = L.xyz

    def ok(f: int) -> bool:
        return 0 <= f < kin.n and bool(L.covered[f]) and np.isfinite(X[f]).all()

    sagittal = None if bf is None else bf[2]

    def knee(side: str, f: int) -> Optional[tuple[float, float]]:
        if not ok(f):
            return None
        h, k, a = X[f, _hj(side, "hip")], X[f, _hj(side, "knee")], X[f, _hj(side, "ankle")]
        # The knee flexes in the body's sagittal plane (normal = lateral axis, from the line of the run). The
        # hip-knee-ankle plane itself is not used: for a nearly straight leg it is undefined, and its depth comes from
        # the lifted pose, so a wrongly bent 3D knee would decide its own gating.
        normal = sagittal if sagittal is not None else np.cross(h - k, a - k)
        return _ang(h, k, a), plane_offset_deg(normal)

    ffc, rel, bfc = events.ffc.frame, events.release.frame, events.bfc.frame
    for key, side, f, ev in (("front_knee_angle_ffc", fl, ffc, "FFC"), ("front_knee_angle_release", fl, rel, "release"),
                             ("back_knee_angle_bfc", bl, bfc, "BFC")):
        r = knee(side, f)
        if r:
            out[key] = {"value": r[0], "frame": f, "plane_offset_deg": r[1], "event": ev}
    if "front_knee_angle_ffc" in out and "front_knee_angle_release" in out:
        out["front_knee_flexion_ffc_to_release"] = {
            "value": out["front_knee_angle_ffc"]["value"] - min(
                (knee(fl, g) or (np.inf,))[0] for g in range(ffc, rel + 1)),
            "frame": ffc, "plane_offset_deg": out["front_knee_angle_ffc"]["plane_offset_deg"], "event": "FFC"}
    if bf is None:
        return out
    up, fwd, lat = bf
    if ok(rel):
        trunk = X[rel, HJ["thorax"]] - X[rel, HJ["pelvis"]]
        sag = trunk - lat * np.dot(trunk, lat)  # projection on the sagittal plane (up, forward)
        out["trunk_forward_flexion_release"] = {
            "value": float(np.degrees(np.arctan2(np.dot(sag, fwd), np.dot(sag, up)))), "frame": rel,
            "plane_offset_deg": plane_offset_deg(lat), "event": "release"}
        fr = trunk - fwd * np.dot(trunk, fwd)  # projection on the frontal plane (up, lateral)
        out["trunk_lateral_flexion_release"] = {
            "value": abs(float(np.degrees(np.arctan2(np.dot(fr, lat), np.dot(fr, up))))), "frame": rel,
            "plane_offset_deg": plane_offset_deg(fwd), "event": "release"}

    def theta(part: str) -> np.ndarray:
        th = np.full(kin.n, np.nan)
        for f in range(kin.n):
            if ok(f):
                v = X[f, _hj(other, part)] - X[f, _hj(arm, part)]  # bowling side → other side, as in three_d.py
                th[f] = np.degrees(np.arctan2(np.dot(v, lat), np.dot(v, fwd)))
        return th
    rot = Rotation3D(theta("shoulder"), theta("hip"), None)
    sep = separation_at(rot, bfc)
    if sep is not None:
        out["hip_shoulder_separation_bfc"] = {"value": sep, "frame": bfc, "plane_offset_deg": plane_offset_deg(up), "event": "BFC"}
    cr = counter_rotation(rot, bfc, ffc)
    if cr is not None:
        out["shoulder_counter_rotation"] = {"value": cr[0], "frame": cr[1], "plane_offset_deg": plane_offset_deg(up), "event": "FFC"}
    return out


# ───────────────────────────── view-aware source choice ─────────────────────────────

LEVELS = ["Low", "Medium", "High"]
SAME_CLIP_2D = {"front_knee_angle_ffc", "front_knee_angle_release", "front_knee_flexion_ffc_to_release",
                "back_knee_angle_bfc", "trunk_forward_flexion_release"}


def _min_level(*lv: Optional[str]) -> str:
    vals = [x for x in lv if x in LEVELS]
    return LEVELS[min(LEVELS.index(x) for x in vals)] if vals else "Medium"


def apply_sources(meas: list, a3: dict[str, dict], events, main_view: str) -> tuple[list, dict]:
    """Choose, per angle metric, between the 2D measurement and the 3D lifted value (angles.source / plane_max_deg)
    and set confidence: High only when 3D and a valid 2D agree within angles.agree_deg; Medium with one source; Low
    when they disagree. Returns (measurements, report {key: {2d, 3d, plane_offset_deg, source, confidence}})."""
    from app.pipeline.metrics import Measurement
    A = P.angles
    ev_conf = {"BFC": events.bfc.confidence, "FFC": events.ffc.confidence, "release": events.release.confidence}
    out, report = [], {}
    for m in meas:
        r3 = a3.get(m.key)
        rule = str(A.source.get(m.key, "auto"))
        if r3 is None or m.view == "3d" or rule == "off":  # nothing lifted, already true 3D (two synced clips), or
            # 3D switched off for this metric: the 2D measurement stands as it is
            out.append(m)
            continue
        same_clip = m.key in SAME_CLIP_2D or main_view in ("front-on", "behind")
        v2 = m.value if (m.measured and same_clip) else None
        off = r3["plane_offset_deg"]
        valid2 = v2 is not None and off <= A.plane_max_deg
        use3 = rule == "3d" or (rule == "auto" and not valid2) or (rule == "2d" and v2 is None)
        v3 = float(r3["value"])
        base = m.confidence if m.measured else ev_conf.get(r3["event"])
        if valid2:
            diff = abs(v3 - v2)
            lv = "High" if diff <= A.agree_deg else "Low"
            why = (f"3D lifted {v3:.0f}° and 2D side-on {v2:.0f}° agree within {A.agree_deg:g}°" if lv == "High" else
                   f"3D lifted {v3:.0f}° and 2D {v2:.0f}° differ by {diff:.0f}° (more than {A.agree_deg:g}°)")
            why += f"; the limb's plane is {off:.0f}° from the picture, so the 2D view is valid."
        elif v2 is not None:
            lv = "Medium"
            why = (f"The limb's plane is {off:.0f}° from the picture (over {A.plane_max_deg:g}°), so the 2D value "
                   f"({v2:.0f}°) is foreshortened; the 3D lifted value is used.")
        else:
            lv = "Medium"
            why = f"Only the 3D lifted value is available (MotionBERT, from the main clip; plane {off:.0f}° from the picture)."
        src = "3D lifted" if use3 else "2D side-on" if main_view not in ("front-on", "behind") else f"2D {main_view}"
        level = _min_level(lv, base)
        reason = f"Source: {src}. {why}" + (f" Limited by: {m.confidence_reason}" if level != lv and m.measured else "")
        value = v3 if use3 else v2
        report[m.key] = {"2d": None if v2 is None else round(v2, 2), "3d": round(v3, 2), "plane_offset_deg": round(off, 1),
                         "source": src, "confidence": level, "rule": rule}
        out.append(Measurement(key=m.key, value=float(value), unit=m.unit, view="3d" if use3 else m.view,
                               frame=int(r3["frame"] if use3 else (m.frame if m.frame is not None else r3["frame"])),
                               event=m.event, confidence=level, confidence_reason=reason, measured=True,
                               joint=m.joint, extra={**(m.extra or {}), "source": src, "value_2d": v2, "value_3d": v3,
                                                     "plane_offset_deg": off}))
    return out, report

"""What kind of clip this is, decided once, each with a confidence and the reasons:

  rotation   pose check of the (metadata-rotated) picture: head above the ankles, else the fix to apply
  direction  sign of the mid-hip travel over the RUN-UP only (the follow-through and walk-back are ignored)
  arm        the wrist with the larger vertical excursion and higher peak speed around its highest point
  view       side-on / front-on / behind / oblique from two independent cues (travel across vs towards the camera,
             and how wide the shoulders look), as a viewing angle with a margin to the class boundaries
  bowler     the person track whose run-up pattern scores best (tracking.score_track), never the biggest person

Everything is in body heights and seconds (config/pipeline.yaml → context), so the same rules hold for any camera,
resolution and frame rate. User overrides replace a detection and are marked source="user".
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from app.pipeline.params import P, frames
from app.pipeline.types import J, PoseSequence

LEVELS = ["Low", "Medium", "High"]
VIEWS = ("side-on", "front-on", "behind", "oblique")


@dataclass
class Detection:
    value: Any
    confidence: str = "Low"
    reasons: list[str] = field(default_factory=list)
    source: str = "auto"  # "auto" | "user"
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def user(value: Any, auto: Optional[Detection] = None) -> Detection:
    d = Detection(value, "High", ["Set by hand."], "user")
    if auto is not None:
        d.details = {"auto_value": auto.value, "auto_confidence": auto.confidence}
    return d


@dataclass
class ClipContext:
    rotation: Detection
    direction: Detection
    arm: Detection
    view: Detection
    bowler: Detection
    candidates: list[dict] = field(default_factory=list)
    roles: Optional[dict] = None  # identity.assign_roles
    identity_log: list = field(default_factory=list)  # left/right swap corrections
    fusion: Optional[dict] = None  # primary vs secondary pose estimator agreement
    engine: str = ""

    def to_dict(self) -> dict:
        return {"rotation": self.rotation.to_dict(), "direction": self.direction.to_dict(),
                "bowling_arm": self.arm.to_dict(), "view": self.view.to_dict(), "bowler": self.bowler.to_dict(),
                "candidates": self.candidates, "roles": self.roles, "identity_corrections": self.identity_log,
                "pose_engines": {"primary": self.engine, "agreement": self.fusion}}

    def low(self) -> list[str]:
        names = {"direction": self.direction, "bowling_arm": self.arm, "view": self.view, "bowler": self.bowler}
        return [k for k, d in names.items() if d.confidence == "Low" and d.source == "auto"]


# ───────────────────────────── helpers ─────────────────────────────

def _bh_series(kp: np.ndarray) -> np.ndarray:
    ank = 0.5 * (kp[:, J["left_ankle"], :2] + kp[:, J["right_ankle"], :2])
    return np.linalg.norm(kp[:, J["nose"], :2] - ank, axis=1) * P.scale.head_top_correction


def body_height(kp: np.ndarray) -> float:
    """BH in pixels: nose-to-ankle × 1.07, median over the upright frames (top of this clip's distribution)."""
    d = _bh_series(kp)
    d = d[np.isfinite(d)]
    if len(d) < P.general.min_samples:
        sh = 0.5 * (kp[:, J["left_shoulder"], :2] + kp[:, J["right_shoulder"], :2])
        hp = 0.5 * (kp[:, J["left_hip"], :2] + kp[:, J["right_hip"], :2])
        an = 0.5 * (kp[:, J["left_ankle"], :2] + kp[:, J["right_ankle"], :2])
        s = np.linalg.norm(sh - hp, axis=1) + np.linalg.norm(hp - an, axis=1)
        s = s[np.isfinite(s)]
        return float(np.median(s) / P.context.bowler.body_size_bh) if len(s) else float("nan")
    return float(np.median(d[d >= np.percentile(d, P.scale.upright_pct)]))


def _level(score: float, lo: float, hi: float) -> str:
    return "High" if score >= hi else ("Medium" if score >= lo else "Low")


def hand_peak_frame(kp: np.ndarray) -> Optional[int]:
    """Frame where either hand is highest above its shoulder line (image y down): a release estimate."""
    sh_y = 0.5 * (kp[:, J["left_shoulder"], 1] + kp[:, J["right_shoulder"], 1])
    best = None
    for w in ("left_wrist", "right_wrist"):
        h = sh_y - kp[:, J[w], 1]
        if np.isfinite(h).sum() >= P.general.min_samples:
            f = int(np.nanargmax(h))
            if best is None or h[f] > best[1]:
                best = (f, float(h[f]))
    return best[0] if best else None


# ───────────────────────────── rotation ─────────────────────────────

def check_rotation(kp: np.ndarray, metadata_rot: int = 0) -> Detection:
    """The picture is already rotated by its metadata. Verify with the pose: the head should be above the ankles.
    value: the extra rotation still needed ("none", "cw90", "ccw90", "180")."""
    R = P.context.rotation
    v = 0.5 * (kp[:, J["left_ankle"], :2] + kp[:, J["right_ankle"], :2]) - kp[:, J["nose"], :2]
    ok = np.all(np.isfinite(v), axis=1)
    meta = f"metadata rotation {metadata_rot}° applied" if metadata_rot else "no rotation in the metadata"
    if ok.sum() < P.general.min_samples:
        return Detection("none", "Low", [f"{meta}; too few frames with head and feet to check."])
    v = v[ok]
    up = float(np.mean(v[:, 1] > np.abs(v[:, 0])))
    down = float(np.mean(-v[:, 1] > np.abs(v[:, 0])))
    left = float(np.mean(v[:, 0] > np.abs(v[:, 1])))   # feet to the right of the head: head points left
    right = float(np.mean(-v[:, 0] > np.abs(v[:, 1])))
    det = {"upright": round(up, 3), "upside_down": round(down, 3), "head_left": round(left, 3), "head_right": round(right, 3)}
    if up >= R.upright_frac:
        return Detection("none", "High" if up >= R.high_frac else "Medium", [f"{meta}; head above the ankles in {up:.0%} of frames."], details=det)
    if down >= R.upright_frac:
        return Detection("180", "Medium", [f"{meta}; the bowler appears upside down in {down:.0%} of frames."], details=det)
    if left >= R.sideways_frac:
        return Detection("cw90", "Medium", [f"{meta}; the body lies sideways (head to the left) in {left:.0%} of frames."], details=det)
    if right >= R.sideways_frac:
        return Detection("ccw90", "Medium", [f"{meta}; the body lies sideways (head to the right) in {right:.0%} of frames."], details=det)
    return Detection("none", "Low", [f"{meta}; posture unclear (upright in only {up:.0%} of frames)."], details=det)


# ───────────────────────────── direction ─────────────────────────────

def runup_frames(kp: np.ndarray, release_hint: Optional[int] = None) -> np.ndarray:
    idx = np.where(np.isfinite(kp[:, J["left_hip"], 0]) & np.isfinite(kp[:, J["right_hip"], 0]))[0]
    if release_hint is not None:
        pre = idx[idx < release_hint]
        if len(pre) >= 2 * P.general.min_samples:
            return pre
    return idx[: max(2 * P.general.min_samples, int(P.context.direction.runup_frac * len(idx)))]


def detect_direction(kp: np.ndarray, release_hint: Optional[int] = None) -> Detection:
    """left_to_right | right_to_left: sign of the mid-hip displacement over the run-up (before release)."""
    D = P.context.direction
    seg = runup_frames(kp, release_hint)
    if len(seg) < 2 * P.general.min_samples:
        return Detection("left_to_right", "Low", ["Too little of the run-up is visible to tell the direction."])
    hip_x = 0.5 * (kp[seg, J["left_hip"], 0] + kp[seg, J["right_hip"], 0])
    q = max(1, len(seg) // 3)
    disp = float(np.median(hip_x[-q:]) - np.median(hip_x[:q]))
    bh = body_height(kp)
    disp_bh = disp / bh if np.isfinite(bh) and bh > 0 else 0.0
    value = "left_to_right" if disp >= 0 else "right_to_left"
    conf = _level(abs(disp_bh), D.runup_min_bh, D.runup_high_bh)
    arrow = "right" if disp >= 0 else "left"
    why = f"Hips move {abs(disp_bh):.1f} body heights to the {arrow} during the run-up."
    if conf == "Low":
        why += " That's little sideways travel (front-on/behind view or a following camera), so it's a guess."
    return Detection(value, conf, [why], details={"displacement_bh": round(disp_bh, 3), "frames": int(len(seg))})


# ───────────────────────────── bowling arm ─────────────────────────────

def detect_arm(kp: np.ndarray, fps: float) -> Detection:
    """The bowling arm swings through the biggest vertical range and fastest around its own highest point (the front
    arm also goes up in the coil, but slower and through a smaller range)."""
    A = P.context.arm
    bh = body_height(kp)
    if not np.isfinite(bh) or bh <= 0:
        return Detection("right", "Low", ["Body size couldn't be measured; right arm assumed."])
    sh_y = 0.5 * (kp[:, J["left_shoulder"], 1] + kp[:, J["right_shoulder"], 1])
    win = frames(A.window_s, fps)
    ev = {}
    for side in ("left", "right"):
        w = kp[:, J[f"{side}_wrist"], :2]
        h = (sh_y - w[:, 1]) / bh  # height above the shoulder line, BH (image y down)
        if np.isfinite(h).sum() < P.general.min_samples:
            ev[side] = (0.0, 0.0, None)
            continue
        hv = np.where(np.isfinite(h), h, -np.inf)
        order = np.argsort(hv)[::-1]
        pk = int(order[1] if len(order) > 1 and np.isfinite(hv[order[1]]) else order[0])  # 2nd highest: one bad frame can't decide
        a, b = max(0, pk - win), min(len(h), pk + win + 1)
        seg = h[a:b]
        exc = float(np.nanmax(seg) - np.nanmin(seg)) if np.isfinite(seg).sum() >= 2 else 0.0
        sp = np.linalg.norm(np.diff(w[a:b], axis=0), axis=1) * fps / bh
        # a high percentile, not the single maximum: one mislabelled frame (a label jumping to the other hand) can't
        # decide the bowling arm
        spd = float(np.nanpercentile(sp, A.speed_pct)) if np.isfinite(sp).sum() >= P.general.min_samples else 0.0
        ev[side] = (exc, spd, pk)
    exc_max = max(ev["left"][0], ev["right"][0], 1e-6)
    spd_max = max(ev["left"][1], ev["right"][1], 1e-6)
    score = {s: ev[s][0] / exc_max + ev[s][1] / spd_max for s in ev}
    value = "left" if score["left"] > score["right"] else "right"
    other = "right" if value == "left" else "left"
    margin = (score[value] - score[other]) / max(score[value], 1e-6)
    conf = _level(margin, A.margin_low, A.margin_high)
    why = (f"{value.capitalize()} hand: vertical swing {ev[value][0]:.2f} vs {ev[other][0]:.2f} body heights, peak "
           f"speed {ev[value][1]:.1f} vs {ev[other][1]:.1f} body heights/s.")
    return Detection(value, conf, [why], details={"margin": round(margin, 3), "peak_frame": ev[value][2],
                                                   "excursion_bh": {s: round(ev[s][0], 3) for s in ev},
                                                   "peak_speed_bh_s": {s: round(ev[s][1], 2) for s in ev}})


# ───────────────────────────── view ─────────────────────────────

def _body_length(k: np.ndarray) -> np.ndarray:
    def seg(a: str, b: str) -> np.ndarray:
        return np.linalg.norm(k[:, J[a], :2] - k[:, J[b], :2], axis=1)
    leg = np.fmax(seg("left_hip", "left_knee") + seg("left_knee", "left_ankle"),
                  seg("right_hip", "right_knee") + seg("right_knee", "right_ankle"))
    trunk = np.linalg.norm(0.5 * (k[:, J["left_shoulder"], :2] + k[:, J["right_shoulder"], :2])
                           - 0.5 * (k[:, J["left_hip"], :2] + k[:, J["right_hip"], :2]), axis=1)
    return leg + trunk


def view_cues(kp: np.ndarray, release_hint: Optional[int] = None) -> Optional[dict]:
    """Run-up geometry: sideways travel and size growth (body lengths), shoulder width ÷ torso, stride spread."""
    idx = runup_frames(kp, release_hint)
    k = kp[idx]
    if len(k) < 5:
        return None
    hip = 0.5 * (k[:, J["left_hip"], :2] + k[:, J["right_hip"], :2])
    torso = np.linalg.norm(0.5 * (k[:, J["left_shoulder"], :2] + k[:, J["right_shoulder"], :2]) - hip, axis=1)
    size = _body_length(k)
    ok = np.isfinite(hip[:, 0]) & np.isfinite(torso) & (torso > 1) & np.isfinite(size)
    if ok.sum() < 5:
        return None
    s = size[ok]
    q = max(2, len(s) // 4)
    dx = k[ok, J["left_shoulder"], 0] - k[ok, J["right_shoulder"], 0]
    ka = kp[np.isfinite(kp[:, 0, 0])]
    sa = _body_length(ka)
    spread = np.abs(ka[:, J["left_ankle"], 0] - ka[:, J["right_ankle"], 0]) / np.where(sa > 1, sa, np.nan)
    return {
        "travel_bl": float((np.percentile(hip[ok, 0], 90) - np.percentile(hip[ok, 0], 10)) / np.median(s)),
        "size_ratio": float(np.median(s[-q:]) / max(1e-6, np.median(s[:q]))),
        "shoulder_width": float(np.median(np.abs(dx) / torso[ok])),
        "left_on_right": float(np.mean(dx > 0)),  # facing the camera, the anatomical left shoulder is on the image right
        "stride_spread": float(np.nanpercentile(spread, P.context.view.stride_pct)) if np.isfinite(spread).any() else 0.0,
        "frames": int(ok.sum()),
    }


def classify_view(kp: np.ndarray, release_hint: Optional[int] = None) -> Detection:
    """Viewing angle (0° = square side-on, 90° = along the pitch) from three independent cues, then the class:
      motion    atan(depth travel ÷ sideways travel) over the run-up (when the bowler moves enough in the picture)
      shoulders asin(shoulder width ÷ torso ÷ frontal ratio) during the run-up
      stride    acos(widest foot separation across the picture ÷ a full delivery stride) — a stride seen end-on looks short
    """
    V = P.context.view
    c = view_cues(kp, release_hint)
    if c is None:
        return Detection("side-on", "Low", ["Too little of the run-up is visible to tell the camera angle; side-on assumed."])
    lat = c["travel_bl"]
    depth = abs(math.log(max(c["size_ratio"], 1e-6))) * V.camera_distance_bl
    motion = math.hypot(lat, depth)
    ang_shoulder = math.degrees(math.asin(min(1.0, c["shoulder_width"] / V.frontal_shoulder_ratio)))
    ang_stride = math.degrees(math.acos(min(1.0, c["stride_spread"] / V.stride_full_bl)))
    cues = [(ang_shoulder, 1.0), (ang_stride, 1.0)]
    reasons = []
    ang_motion = None
    if motion >= V.min_motion_bl:
        ang_motion = math.degrees(math.atan2(depth, lat))
        cues.append((ang_motion, min(1.0, motion / (2 * V.min_motion_bl))))
        reasons.append(f"Run-up travel: {lat:.1f} body lengths across the picture, size ×{c['size_ratio']:.2f} "
                       f"(≈ {ang_motion:.0f}° from side-on).")
    else:
        reasons.append("The bowler hardly moves in the picture (a following camera?), so the angle comes from the body "
                       "shape alone.")
    reasons.append(f"Shoulders look {c['shoulder_width']:.2f} × torso length wide (≈ {ang_shoulder:.0f}°); feet spread up "
                   f"to {c['stride_spread']:.2f} body lengths across the picture (≈ {ang_stride:.0f}°).")
    wsum = sum(w for _, w in cues)
    angle = sum(a * w for a, w in cues) / wsum
    spread_deg = max(a for a, _ in cues) - min(a for a, _ in cues)
    if angle < V.side_max_deg:
        value, edge = "side-on", V.side_max_deg - angle
    elif angle > V.front_min_deg:
        towards = c["size_ratio"] > 1.0 if abs(math.log(max(c["size_ratio"], 1e-6))) > V.min_log_growth else c["left_on_right"] >= 0.5
        value, edge = ("front-on" if towards else "behind"), angle - V.front_min_deg
        reasons.append("Coming towards the camera." if towards else "Running away from the camera.")
    else:
        value, edge = "oblique", min(angle - V.side_max_deg, V.front_min_deg - angle)
    conf = "High" if edge >= V.band_deg else ("Medium" if edge >= V.band_deg / 2 else "Low")
    if spread_deg > V.max_cue_disagreement_deg and conf != "Low":
        conf = "Medium" if conf == "High" else "Low"
        reasons.append("The cues disagree somewhat, so the camera angle is less certain.")
    return Detection(value, conf, reasons, details={**{k: round(v, 3) for k, v in c.items()}, "angle_deg": round(angle, 1),
                                                    "angle_motion_deg": None if ang_motion is None else round(ang_motion, 1),
                                                    "angle_shoulder_deg": round(ang_shoulder, 1),
                                                    "angle_stride_deg": round(ang_stride, 1)})


# ───────────────────────────── bowler ─────────────────────────────

def bowler_detection(pose: PoseSequence, cands: list[dict], chosen_rank: int = 0) -> Detection:
    if pose.selected_by == "tap":
        return Detection(chosen_rank, "High", ["You tapped the bowler."], "user")
    if not cands:
        return Detection(chosen_rank, "Low", ["Only one person track could be built."])
    best = cands[chosen_rank] if chosen_rank < len(cands) else cands[0]
    others = [c["score"] for i, c in enumerate(cands) if i != chosen_rank]
    margin = best["score"] - max(others) if others else best["score"]
    why = (f"Person {chosen_rank + 1} of {len(cands)}: run-up score {best['score']:.2f}/3 (travel {best['travel']:.2f}, "
           f"deceleration {best['decel']:.2f}, arm swing {best['arm']:.2f}).")
    if not others:
        conf = "High" if best["score"] >= P.context.bowler.solo_high_score else "Medium"
    else:
        conf = "Low" if margin < P.context.bowler.margin_low else ("High" if margin >= 3 * P.context.bowler.margin_low else "Medium")
        why += f" Next best {max(others):.2f}."
    if chosen_rank > 0:
        conf = "Low" if conf == "Low" else "Medium"
        why += " The higher-ranked people showed no bowling action."
    return Detection(chosen_rank, conf, [why], details={"margin": round(margin, 3)})

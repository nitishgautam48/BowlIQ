"""Stage 7: measurements. Plain geometry on smoothed keypoints at the detected events.

Returns raw measurements (value, unit, view, frame, confidence, reason). Judgement (status, targets) is applied
later in app/analysis/rules.py using benchmarks.yaml and the bowler's baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.analysis.benchmarks import rule
from app.biomech.angles import joint_angle, line_angle, line_to_line_angle, segment_angle_vs_vertical, wrap_deg
from app.pipeline.params import P
from app.pipeline.params import frames as n_frames
from app.pipeline.types import DeliveryEvents, J, Kinematics, Scale

LEVELS = ["Low", "Medium", "High"]

METHODS: dict[str, dict] = {
    "stride_length_pct_height": {
        "method": "Horizontal distance between the planted back foot (settled ankle position after back foot contact) and the "
                  "planted front foot (settled ankle position after front foot contact), divided by body height.",
        "event": "FFC", "joint": "front_ankle"},
    "front_knee_angle_ffc": {
        "method": "Angle at the front knee between the hip–knee and knee–ankle lines at front foot contact (180° = straight).",
        "event": "FFC", "joint": "front_knee"},
    "front_knee_angle_release": {
        "method": "Angle at the front knee between the hip–knee and knee–ankle lines at release (180° = straight).",
        "event": "release", "joint": "front_knee"},
    "front_knee_flexion_ffc_to_release": {
        "method": "Largest bend of the front knee between front foot contact and release, relative to its angle at front "
                  "foot contact (positive = the knee bends).",
        "event": "release", "joint": "front_knee"},
    "trunk_forward_flexion_release": {
        "method": "Angle between the mid-hip → mid-shoulder line and vertical at release, positive when leaning towards the batter.",
        "event": "release", "joint": "mid_shoulder"},
    "release_height_pct_height": {
        "method": "Height of the bowling wrist at release above the planted front ankle (ground line at FFC), divided by body height.",
        "event": "release", "joint": "bowling_wrist"},
    "runup_speed": {
        "method": "Horizontal speed of the mid-hip from a straight-line fit over the 0.3 s before back foot contact.",
        "event": "run-up", "joint": "mid_hip"},
    "hip_shoulder_separation_bfc": {
        "method": "Angle between the shoulder line and the hip line in the image plane at back foot contact, front-on clip.",
        "event": "BFC", "joint": "mid_shoulder"},
    "shoulder_counter_rotation": {
        "method": "Largest change in the shoulder-line angle between back foot contact and front foot contact, relative to "
                  "its angle at back foot contact, front-on clip.",
        "event": "FFC", "joint": "mid_shoulder"},
    "trunk_lateral_flexion_release": {
        "method": "Sideways angle of the mid-hip → mid-shoulder line from vertical at release, second clip (front-on or "
                  "behind the bowler).",
        "event": "release", "joint": "mid_shoulder"},
    # ── Phase 3: side-on additions (trend only: no research threshold) ──
    "head_ahead_of_front_foot_release": {
        "method": "Horizontal distance of the head (nose, or ears when the face is hidden) ahead of the planted front ankle "
                  "at release, divided by body height. Positive = head ahead of the front foot, towards the batter.",
        "event": "release", "joint": "nose"},
    "head_bob_runup": {
        "method": "Up-and-down range of the head (10th–90th percentile of its height) over the last 0.4 s of the run-up "
                  "before back foot contact, divided by body height.",
        "event": "run-up", "joint": "nose"},
    "back_knee_angle_bfc": {
        "method": "Angle at the back knee between the hip–knee and knee–ankle lines at back foot contact (180° = straight).",
        "event": "BFC", "joint": "back_knee"},
    "front_leg_plant_angle_ffc": {
        "method": "Angle of the front leg (hip → ankle line) from vertical at front foot contact; positive = foot planted "
                  "ahead of the hip.",
        "event": "FFC", "joint": "front_ankle"},
    "release_ahead_of_front_foot": {
        "method": "Horizontal distance of the bowling wrist ahead of the planted front ankle at release, divided by body "
                  "height.",
        "event": "release", "joint": "bowling_wrist"},
    "front_foot_landing_angle_ffc": {
        "method": "Angle of the heel → toe line of the front foot above horizontal at front foot contact; positive = toe up "
                  "(heel first), negative = toe first. Needs the heel and toe points (CPU/MediaPipe engine).",
        "event": "FFC", "joint": "front_heel"},
    "speed_kept_at_ffc": {
        "method": "Horizontal speed of the mid-hip around front foot contact (±0.03 s) as a percentage of the run-up speed "
                  "(0.3 s before back foot contact): how much run-up speed is carried into the front foot landing.",
        "event": "FFC", "joint": "mid_hip"},
    "front_arm_pulldown_ms": {
        "method": "Time from the highest point of the non-bowling (front) wrist, between back foot contact and release, to "
                  "release.",
        "event": "release", "joint": "front_wrist"},
    "bowling_elbow_extension": {
        "method": "Change of the bowling-arm elbow angle from when the upper arm passes horizontal to release, in the 2D "
                  "side-on picture (positive = the elbow straightens). A trend only: 2D video can't judge whether an "
                  "action is legal.",
        "event": "release", "joint": "bowling_elbow"},
    # ── Phase 3: second-clip additions (front-on or behind the bowler) ──
    "head_lateral_offset_release": {
        "method": "Sideways distance of the head from the planted front ankle at release, divided by body height; positive "
                  "= towards the non-bowling-arm side (head falling away from the line).",
        "event": "release", "joint": "nose"},
    "front_foot_alignment": {
        "method": "Sideways position of the planted front foot (FFC) relative to the planted back foot (BFC), divided by "
                  "body height; positive = across towards the bowling-arm side (closed), negative = open.",
        "event": "FFC", "joint": "front_ankle"},
    "bowling_arm_angle_release": {
        "method": "Angle of the bowling arm (shoulder → wrist) from vertical at release, seen from in front or behind; "
                  "positive = arm away from the head towards the bowling-arm side (lower, round-arm).",
        "event": "release", "joint": "bowling_wrist"},
    "front_knee_inward_release": {
        "method": "Sideways distance of the front knee from the hip–ankle line at release, as a percentage of leg length; "
                  "positive = knee caving inwards (towards the other leg).",
        "event": "release", "joint": "front_knee"},
    "runup_line_drift": {
        "method": "Sideways movement of the mid-hip over the last 0.5 s of the run-up before back foot contact, in body "
                  "heights (perspective-corrected by the body's size in the picture).",
        "event": "run-up", "joint": "mid_hip"},
}

SIDE_KEYS = ["stride_length_pct_height", "front_knee_angle_ffc", "front_knee_angle_release",
             "front_knee_flexion_ffc_to_release", "trunk_forward_flexion_release", "release_height_pct_height", "runup_speed",
             "head_ahead_of_front_foot_release", "head_bob_runup", "back_knee_angle_bfc", "front_leg_plant_angle_ffc",
             "release_ahead_of_front_foot", "front_foot_landing_angle_ffc", "speed_kept_at_ffc", "front_arm_pulldown_ms",
             "bowling_elbow_extension"]
FRONT_KEYS = ["hip_shoulder_separation_bfc", "shoulder_counter_rotation", "trunk_lateral_flexion_release",
              "head_lateral_offset_release", "front_foot_alignment", "bowling_arm_angle_release",
              "front_knee_inward_release", "runup_line_drift"]
# Second-clip metrics that stay meaningful when that clip is diagonal (reported at Low confidence); the rest need the
# camera on the line of the run (front-on or behind)
DIAGONAL_OK = {"trunk_lateral_flexion_release", "head_lateral_offset_release", "bowling_arm_angle_release"}


@dataclass
class Measurement:
    key: str
    value: Optional[float]
    unit: str
    view: str
    frame: Optional[int]
    event: Optional[str]
    confidence: Optional[str]
    confidence_reason: str
    measured: bool
    reason: Optional[str] = None
    joint: Optional[str] = None
    extra: dict = field(default_factory=dict)


def legs(kin: Kinematics) -> tuple[str, str]:
    """(front leg side, back leg side) from the limb roles (identity.assign_roles); before roles exist, the usual
    anatomy (front = opposite the bowling arm)."""
    return kin.side("ankle", "front").split("_")[0], kin.side("ankle", "back").split("_")[0]


def resolve_joint(kin, token: str) -> str:
    """Role token (front_knee, bowling_wrist, …) → the joint name to highlight. `kin` may be a bowling-arm string
    (older callers) or Kinematics (uses the leg roles)."""
    if isinstance(kin, str):
        arm, fl, bl = kin, ("left" if kin == "right" else "right"), kin
    else:
        arm, (fl, bl) = kin.bowling_arm, legs(kin)
    other = "left" if arm == "right" else "right"
    return {"front_ankle": f"{fl}_ankle", "front_knee": f"{fl}_knee", "bowling_wrist": f"{arm}_wrist",
            "mid_shoulder": f"{arm}_shoulder", "mid_hip": f"{bl}_hip", "back_knee": f"{bl}_knee",
            "front_heel": f"{fl}_heel", "front_wrist": f"{other}_wrist", "bowling_elbow": f"{arm}_elbow"}.get(token, token)


def _vis_level(v: float) -> str:
    return "High" if v >= 0.8 else ("Medium" if v >= 0.6 else "Low")


def combine_conf(levels: list[str], reasons: list[str], cap: Optional[str] = None, cap_reason: Optional[str] = None) -> tuple[str, str]:
    lvl = min(LEVELS.index(lv) for lv in levels)
    rs = list(dict.fromkeys(r for r in reasons if r))
    if cap is not None and lvl > LEVELS.index(cap):
        lvl = LEVELS.index(cap)
        if cap_reason:
            rs.insert(0, cap_reason)
    elif cap_reason:
        rs.insert(0, cap_reason)
    return LEVELS[lvl], " ".join(rs) if rs else "Measured cleanly."


def _not(key: str, view: str, reason: str) -> Measurement:
    r = rule(key)
    return Measurement(key=key, value=None, unit=r["unit"], view=view, frame=None, event=METHODS[key]["event"],
                       confidence=None, confidence_reason=reason, measured=False, reason=reason)


class _Ctx:
    def __init__(self, kin: Kinematics, quality_level: str, quality_reason: str):
        self.kin = kin
        self.q = quality_level
        self.qr = quality_reason

    def ok(self, joints: list[str], frames: list[int]) -> Optional[str]:
        for f in frames:
            if f < 0 or f >= self.kin.n:
                return f"Frame {f} is outside the clip."
            for jn in joints:
                if not self.kin.valid[f, J[jn]]:
                    return f"The {jn.replace('_', ' ')} is not visible at frame {f}."
        return None

    def vis(self, joints: list[str], frames: list[int]) -> float:
        return float(np.mean([self.kin.vis[f, J[jn]] for f in frames for jn in joints]))

    def conf(self, joints: list[str], frames: list[int], ev_levels: list[tuple[str, str]],
             cap: Optional[str] = None, cap_reason: Optional[str] = None, extra: Optional[str] = None) -> tuple[str, str]:
        v = self.vis(joints, frames)
        levels = [self.q, _vis_level(v)] + [lv for lv, _ in ev_levels]
        reasons = []
        if self.q != "High":
            reasons.append(f"Clip quality {self.q.lower()}: {self.qr}")
        if v < 0.8:
            reasons.append(f"Joint visibility {v:.2f} at the measured frame.")
        for lv, r in ev_levels:
            if lv != "High":
                reasons.append(r)
        if extra:
            reasons.append(extra)
            levels.append("Medium")
        if not reasons and cap_reason is None:
            reasons.append(f"{self.kin.fps:.0f} fps, joints clearly visible, sharp event detection.")
        return combine_conf(levels, reasons, cap, cap_reason)


PLANTED_DRIFT = 0.12  # body heights: max movement of the planted front ankle between FFC and release
MIN_LINE_WIDTH = 0.35  # shoulder/hip line length ÷ torso length below which the line is seen end-on
TILT_LOW_DEG = 8.0  # above this, distances along the pitch are foreshortened: Low confidence
TILT_MAX_DEG = 15.0  # above this, distances along the pitch are not reported


def ground_tilt(kin: Kinematics, ev: DeliveryEvents) -> Optional[float]:
    """Angle (deg) of the line between the planted back ankle (BFC) and front ankle (FFC) in the image.

    Filmed square-on, both foot contacts sit on one horizontal image line. A tilted line means the bowler runs
    towards or away from the camera, so every distance along the pitch is shortened by the viewing angle.
    """
    fl, bl = legs(kin)  # leg ROLES (front = planted at FFC), not raw labels
    b = kin.xy[ev.bfc.frame, J[f"{bl}_ankle"]]
    f = kin.xy[ev.ffc.frame, J[f"{fl}_ankle"]]
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(f))):
        return None
    return float(np.degrees(np.arctan2(abs(f[1] - b[1]), abs(f[0] - b[0]) + 1e-9)))


def oblique_reason(tilt: Optional[float]) -> Optional[str]:
    if tilt is None or tilt <= TILT_LOW_DEG:
        return None
    return (f"The feet land at different depths (ground line tilted {tilt:.0f}°), so the run isn't square to the camera "
            "and distances along the pitch are shortened.")


STILL_X = 0.25  # body heights / s: a planted foot's ankle moves less than this


def settled_x(x: np.ndarray, start: int, stop: int, body_h: float, fps: float) -> float:
    """Horizontal ankle position once the foot has settled after contact (first frame in [start, stop] whose speed is
    below STILL_X); falls back to the contact frame itself."""
    thr = STILL_X * body_h / fps
    for f in range(start, max(start, stop)):
        if f + 1 < len(x) and np.isfinite(x[f]) and np.isfinite(x[f + 1]) and abs(x[f + 1] - x[f]) < thr:
            return float(x[f])
    return float(x[start])


SCALE_DEPENDENT = {"stride_length_pct_height", "release_height_pct_height", "runup_speed",
                   "head_ahead_of_front_foot_release", "release_ahead_of_front_foot", "head_bob_runup"}


def scale_cap(rel: float) -> tuple[Optional[str], Optional[str]]:
    if rel > 0.20:
        return "Low", f"Body scale is uncertain (±{rel * 100:.0f}%), which affects distances and speeds."
    if rel > 0.10:
        return "Medium", f"Body scale is uncertain (±{rel * 100:.0f}%), which affects distances and speeds."
    return None, None


def side_metrics(kin: Kinematics, ev: DeliveryEvents, scale: Scale, quality_level: str, quality_reason: str,
                 camera_motion: Optional[float], tilt: Optional[float] = None) -> list[Measurement]:
    c = _Ctx(kin, quality_level, quality_reason)
    arm = kin.bowling_arm
    fl, bl = legs(kin)  # leg ROLES (front = planted at FFC), not raw labels
    hip_f, knee_f, ank_f = f"{fl}_hip", f"{fl}_knee", f"{fl}_ankle"
    ank_b = f"{bl}_ankle"
    wr = f"{arm}_wrist"
    bh = scale.body_height_px
    bfc, ffc, rel = ev.bfc.frame, ev.ffc.frame, ev.release.frame
    E = {"BFC": (ev.bfc.confidence, f"BFC: {ev.bfc.reason}"), "FFC": (ev.ffc.confidence, f"FFC: {ev.ffc.reason}"),
         "release": (ev.release.confidence, f"Release: {ev.release.reason}")}
    out: list[Measurement] = []
    xy = kin.xy

    def m(key: str, value: float, frame: int, conf: tuple[str, str], unit: Optional[str] = None, **extra) -> Measurement:
        return Measurement(key=key, value=float(value), unit=unit or rule(key)["unit"], view="side-on", frame=int(frame),
                           event=METHODS[key]["event"], confidence=conf[0], confidence_reason=conf[1], measured=True,
                           joint=resolve_joint(kin, METHODS[key]["joint"]), extra=extra)

    # stride length
    if tilt is None:
        tilt = ground_tilt(kin, ev)
    oblique = oblique_reason(tilt)
    too_oblique = tilt is not None and tilt > TILT_MAX_DEG
    k = "stride_length_pct_height"
    err = c.ok([ank_b], [bfc]) or c.ok([ank_f], [ffc])
    if err:
        out.append(_not(k, "side-on", err))
    elif too_oblique:
        out.append(_not(k, "side-on", f"{oblique} Film square to the crease to measure stride."))
    else:
        # footprint-to-footprint: the settled position of each planted foot, so the result doesn't depend on whether
        # the contact frame was detected at heel strike or a few frames later when the foot is flat
        d = abs(settled_x(xy[:, J[ank_f], 0], ffc, rel, bh, kin.fps) - settled_x(xy[:, J[ank_b], 0], bfc, ffc, bh, kin.fps))
        cf = c.conf([ank_f], [ffc], [E["BFC"], E["FFC"]], cap="Low" if oblique else None, cap_reason=oblique)
        lv = min(LEVELS.index(cf[0]), LEVELS.index(c.conf([ank_b], [bfc], [])[0]))
        out.append(m(k, d / bh * 100.0, ffc, (LEVELS[lv], cf[1]),
                     metres=(d * scale.metres_per_pixel) if scale.metres_per_pixel else None))

    knee = joint_angle(xy[:, J[hip_f]], xy[:, J[knee_f]], xy[:, J[ank_f]])
    leg = [hip_f, knee_f, ank_f]
    # the front foot is planted from FFC to release; if its tracked ankle wanders, the leg estimate is unreliable there
    seg = xy[ffc:rel + 1, J[ank_f]]
    drift = float(np.nanmax(np.linalg.norm(seg - xy[ffc, J[ank_f]], axis=1))) if np.isfinite(seg).any() else np.inf
    leg_unreliable = None
    if drift > PLANTED_DRIFT * bh:
        leg_unreliable = (f"The front ankle moves {drift / bh * 100:.0f}% of body height between front foot contact and "
                          "release while the foot is planted, so the leg tracking is unreliable there (blur or occlusion).")
    # knee at FFC
    k = "front_knee_angle_ffc"
    err = c.ok(leg, [ffc])
    out.append(_not(k, "side-on", err) if err else m(k, knee[ffc], ffc, c.conf(leg, [ffc], [E["FFC"]])))
    # knee at release
    k = "front_knee_angle_release"
    err = c.ok(leg, [rel]) or leg_unreliable
    out.append(_not(k, "side-on", err) if err else m(k, knee[rel], rel, c.conf(leg, [rel], [E["release"]])))
    # flexion FFC → release
    k = "front_knee_flexion_ffc_to_release"
    window = list(range(ffc, rel + 1))
    err = c.ok(leg, window) or leg_unreliable
    if err:
        out.append(_not(k, "side-on", err))
    else:
        flex = knee[ffc] - knee[ffc:rel + 1]
        i = int(np.argmax(flex))
        out.append(m(k, max(0.0, float(flex[i])), ffc + i, c.conf(leg, [ffc, ffc + i], [E["FFC"], E["release"]]),
                     knee_ffc=float(knee[ffc]), knee_min=float(knee[ffc + i])))
    # trunk forward flexion at release
    k = "trunk_forward_flexion_release"
    tj = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
    err = c.ok(tj, [rel])
    if err:
        out.append(_not(k, "side-on", err))
    else:
        mh = kin.mid("left_hip", "right_hip")[rel]
        ms = kin.mid("left_shoulder", "right_shoulder")[rel]
        out.append(m(k, segment_angle_vs_vertical(mh, ms), rel, c.conf(tj, [rel], [E["release"]])))
    # release height
    k = "release_height_pct_height"
    err = c.ok([wr], [rel]) or c.ok([ank_f], [ffc]) or leg_unreliable
    if err:
        out.append(_not(k, "side-on", err))
    else:
        hgt = xy[rel, J[wr], 1] - xy[ffc, J[ank_f], 1]  # ground line = planted front ankle at FFC
        out.append(m(k, hgt / bh * 100.0, rel, c.conf([wr], [rel], [E["release"]]),
                     metres=(hgt * scale.metres_per_pixel) if scale.metres_per_pixel else None))
    # run-up speed
    k = "runup_speed"
    runup_px_per_s: Optional[float] = None
    n_win = n_frames(P.metrics.runup_window_s, kin.fps)
    a = max(0, bfc - n_win)
    mh_x = kin.mid("left_hip", "right_hip")[:, 0]
    frames = np.arange(a, bfc + 1)
    good = frames[np.isfinite(mh_x[frames])]
    if camera_motion is not None and camera_motion >= P.camera.moving_wps:
        out.append(_not(k, "side-on", "The camera moves, so speeds across the frame can't be trusted. Use a tripod."))
    elif too_oblique:
        out.append(_not(k, "side-on", f"{oblique} Film square to the crease to measure run-up speed."))
    elif len(good) < max(4, n_frames(P.metrics.runup_min_s, kin.fps)):
        out.append(_not(k, "side-on", "Less than 0.15 s of run-up is visible before back foot contact."))
    else:
        slope = float(np.polyfit(good / kin.fps, mh_x[good], 1)[0])
        runup_px_per_s = abs(slope)
        extra = None if len(good) >= n_win * 0.9 else f"Only {len(good) / kin.fps:.2f} s of run-up visible before back foot contact."
        cf = c.conf(["left_hip", "right_hip"], [int(good[0]), int(good[-1])], [E["BFC"]], extra=extra,
                    cap="Low" if oblique else None, cap_reason=oblique)
        if scale.metres_per_pixel:
            out.append(m(k, abs(slope) * scale.metres_per_pixel, int(good[0]), cf, unit="m/s"))
        else:
            out.append(m(k, abs(slope) / bh * 100.0, int(good[0]), cf, unit=rule(k)["unit_relative"]))
    out += _side_phase3(kin, ev, c, E, m, bh, camera_motion, oblique, too_oblique, leg_unreliable, runup_px_per_s)
    cap, why = scale_cap(scale.rel_uncertainty)
    if cap:
        for mm in out:
            if mm.measured and mm.key in SCALE_DEPENDENT and LEVELS.index(mm.confidence) > LEVELS.index(cap):
                mm.confidence = cap
                mm.confidence_reason = f"{why} {mm.confidence_reason}"
    return out


def _head(kin: Kinematics, f: int) -> tuple[Optional[np.ndarray], list[str]]:
    """Head point: the nose, or the middle of the ears when the face is turned away."""
    if 0 <= f < kin.n:
        if kin.valid[f, J["nose"]]:
            return kin.xy[f, J["nose"]], ["nose"]
        if kin.valid[f, J["left_ear"]] and kin.valid[f, J["right_ear"]]:
            return 0.5 * (kin.xy[f, J["left_ear"]] + kin.xy[f, J["right_ear"]]), ["left_ear", "right_ear"]
    return None, []


TWO_D_ELBOW = ("2D projection: when the arm turns towards or away from the camera the elbow angle in the picture "
               "changes, so this is a personal trend only and can't judge whether an action is legal.")


def _side_phase3(kin, ev, c, E, m, bh, camera_motion, oblique, too_oblique, leg_unreliable, runup_px_per_s) -> list:
    arm = kin.bowling_arm
    fr = "left" if arm == "right" else "right"
    fl, bl = legs(kin)  # leg ROLES (front = planted at FFC), not raw labels
    xy = kin.xy
    bfc, ffc, rel = ev.bfc.frame, ev.ffc.frame, ev.release.frame
    ank_f = f"{fl}_ankle"
    out: list[Measurement] = []
    along_pitch = f"{oblique} Film square to the crease to measure this." if too_oblique else None
    obl_cap = ("Low", oblique) if oblique else (None, None)
    front_x = settled_x(xy[:, J[ank_f], 0], ffc, rel, bh, kin.fps) if kin.valid[ffc, J[ank_f]] else None

    # head ahead of the front foot at release
    k = "head_ahead_of_front_foot_release"
    hp, hj = _head(kin, rel)
    err = (None if hp is not None else f"The head is not visible at frame {rel}.") or c.ok([ank_f], [ffc]) or leg_unreliable \
        or along_pitch
    out.append(_not(k, "side-on", err) if err else
               m(k, (hp[0] - front_x) / bh * 100.0, rel, c.conf(hj, [rel], [E["release"]], *obl_cap)))

    # head bob over the last 0.4 s of the run-up
    k = "head_bob_runup"
    w0 = max(0, bfc - int(round(0.4 * kin.fps)))
    ys = np.array([p[1] for p in (_head(kin, f)[0] for f in range(w0, bfc)) if p is not None])
    if camera_motion is not None and camera_motion >= P.camera.steady_wps:
        out.append(_not(k, "side-on", "The camera moves, so the head's up-and-down movement can't be separated from it. "
                                      "Use a tripod."))
    elif len(ys) < max(4, int(0.25 * kin.fps)):
        out.append(_not(k, "side-on", "Less than 0.25 s of the run-up with the head visible before back foot contact."))
    else:
        out.append(m(k, float(np.percentile(ys, 90) - np.percentile(ys, 10)) / bh * 100.0, w0,
                     c.conf(["left_hip", "right_hip"], [w0, bfc], [E["BFC"]])))

    # back knee at back foot contact
    k = "back_knee_angle_bfc"
    legb = [f"{bl}_hip", f"{bl}_knee", f"{bl}_ankle"]
    err = c.ok(legb, [bfc])
    out.append(_not(k, "side-on", err) if err else
               m(k, float(joint_angle(*(xy[bfc, J[j]] for j in legb))), bfc, c.conf(legb, [bfc], [E["BFC"]])))

    # front leg plant angle at FFC
    k = "front_leg_plant_angle_ffc"
    err = c.ok([f"{fl}_hip", ank_f], [ffc])
    if err:
        out.append(_not(k, "side-on", err))
    else:
        h, a = xy[ffc, J[f"{fl}_hip"]], xy[ffc, J[ank_f]]
        ang = float(np.degrees(np.arctan2(a[0] - h[0], h[1] - a[1])))
        out.append(m(k, ang, ffc, c.conf([f"{fl}_hip", ank_f], [ffc], [E["FFC"]], *obl_cap)))

    # release point ahead of the front foot
    k = "release_ahead_of_front_foot"
    wr = f"{arm}_wrist"
    err = c.ok([wr], [rel]) or c.ok([ank_f], [ffc]) or leg_unreliable or along_pitch
    out.append(_not(k, "side-on", err) if err else
               m(k, (xy[rel, J[wr], 0] - front_x) / bh * 100.0, rel, c.conf([wr], [rel], [E["release"]], *obl_cap)))

    # front foot landing angle (heel/toe)
    k = "front_foot_landing_angle_ffc"
    heel, toe = f"{fl}_heel", f"{fl}_foot_index"
    if not (kin.valid[ffc, J[heel]] and kin.valid[ffc, J[toe]]):
        out.append(_not(k, "side-on", "The heel and toe of the front foot aren't visible at front foot contact (the GPU "
                                      "engine doesn't detect them; the CPU engine does when the foot is in view)."))
    else:
        d = xy[ffc, J[toe]] - xy[ffc, J[heel]]
        foot_len = float(np.hypot(*d))
        if foot_len < 0.04 * bh:
            out.append(_not(k, "side-on", "The front foot is too small or foreshortened in the picture to measure its angle."))
        else:
            out.append(m(k, float(np.degrees(np.arctan2(d[1], abs(d[0])))), ffc,
                         c.conf([heel, toe], [ffc], [E["FFC"]], "Low" if oblique else "Medium",
                                oblique or "Foot points are small in the picture; small pixel errors change this angle.")))

    # speed kept at front foot contact
    k = "speed_kept_at_ffc"
    mh_x = kin.mid("left_hip", "right_hip")[:, 0]
    dw = max(2, int(round(0.03 * kin.fps)))
    win = np.arange(max(0, ffc - dw), min(kin.n, ffc + dw + 1))
    win = win[np.isfinite(mh_x[win])]
    if camera_motion is not None and camera_motion >= P.camera.moving_wps:
        out.append(_not(k, "side-on", "The camera moves, so speeds across the frame can't be trusted. Use a tripod."))
    elif along_pitch:
        out.append(_not(k, "side-on", along_pitch))
    elif runup_px_per_s is None or runup_px_per_s <= 0:
        out.append(_not(k, "side-on", "The run-up speed couldn't be measured, so there is nothing to compare with."))
    elif len(win) < 3:
        out.append(_not(k, "side-on", "The hips aren't visible around front foot contact."))
    else:
        v = abs(float(np.polyfit(win / kin.fps, mh_x[win], 1)[0]))
        out.append(m(k, v / runup_px_per_s * 100.0, ffc,
                     c.conf(["left_hip", "right_hip"], [int(win[0]), int(win[-1])], [E["BFC"], E["FFC"]], *obl_cap)))

    # front arm pull-down timing
    k = "front_arm_pulldown_ms"
    fw = f"{fr}_wrist"
    rng = np.arange(max(0, bfc - int(round(0.15 * kin.fps))), rel + 1)
    vis = kin.valid[rng, J[fw]]
    if vis.mean() < 0.7:
        out.append(_not(k, "side-on", "The front (non-bowling) wrist is hidden for most of back foot contact → release "
                                      "(often behind the body side-on)."))
    else:
        yy = np.where(vis, xy[rng, J[fw], 1], -np.inf)
        i = int(np.argmax(yy))
        p = int(rng[i])
        if i == 0:
            out.append(_not(k, "side-on", "The front arm is already coming down when the clip's view of the gather "
                                          "starts, so its highest point isn't visible."))
        else:
            out.append(m(k, (rel - p) / kin.fps * 1000.0, p, c.conf([fw], [p], [E["release"]])))

    # bowling elbow extension (2D trend)
    k = "bowling_elbow_extension"
    sh, el, wr = f"{arm}_shoulder", f"{arm}_elbow", f"{arm}_wrist"
    rng = np.arange(ffc, rel + 1)
    ok = kin.valid[rng, J[sh]] & kin.valid[rng, J[el]] & kin.valid[rng, J[wr]]
    if not ok.all():
        out.append(_not(k, "side-on", "The bowling arm isn't visible in every frame from front foot contact to release."))
    else:
        up = xy[rng, J[el]] - xy[rng, J[sh]]
        horiz = np.abs(up[:, 1]) / np.maximum(np.linalg.norm(up, axis=1), 1e-6)  # 0 = upper arm horizontal
        i = int(np.argmin(horiz))
        if horiz[i] > 0.35:
            out.append(_not(k, "side-on", "The upper arm never passes horizontal in the picture between front foot "
                                          "contact and release."))
        else:
            ea = joint_angle(xy[rng, J[sh]], xy[rng, J[el]], xy[rng, J[wr]])
            h = int(rng[i])
            out.append(m(k, float(ea[-1] - ea[i]), rel, c.conf([sh, el, wr], [h, rel], [E["FFC"], E["release"]], "Low",
                                                              TWO_D_ELBOW), horizontal_frame=h))
    return out


def front_metrics(kin: Optional[Kinematics], ev: Optional[DeliveryEvents], quality_level: str, quality_reason: str,
                  missing_reason: str, view: str = "front-on") -> list[Measurement]:
    """Measurements from the second clip. `view` is its detected camera angle: front-on (facing the bowler), behind
    (behind the bowler) or diagonal; each metric is only reported where that angle shows it reliably."""
    view = view if view in ("front-on", "behind", "diagonal") else "front-on"
    if kin is None or ev is None:
        return [_not(k, view, missing_reason) for k in FRONT_KEYS]
    c = _Ctx(kin, quality_level, quality_reason)
    arm = kin.bowling_arm
    fl, bl = legs(kin)  # leg ROLES (front = planted at FFC), not raw labels
    xy = kin.xy
    bfc, ffc, rel = ev.bfc.frame, ev.ffc.frame, ev.release.frame
    tag = {"front-on": "Front-on", "behind": "Behind", "diagonal": "Diagonal"}[view]
    E = {"BFC": (ev.bfc.confidence, f"{tag} BFC: {ev.bfc.reason}"), "FFC": (ev.ffc.confidence, f"{tag} FFC: {ev.ffc.reason}"),
         "release": (ev.release.confidence, f"{tag} release: {ev.release.reason}")}
    rot_cap, rot_reason = "Medium", "2D projection of a 3D rotation."
    diag_cap, diag_reason = "Low", "The second clip is filmed diagonally, which distorts sideways angles and distances."
    sh = ["left_shoulder", "right_shoulder"]
    hp = ["left_hip", "right_hip"]
    out: list[Measurement] = []
    torso = np.linalg.norm(kin.mid("left_shoulder", "right_shoulder") - kin.mid("left_hip", "right_hip"), axis=1)
    body_h = _second_body_height(kin)

    def line_visible(a: str, b: str, frames: list[int]) -> Optional[str]:
        """A shoulder/hip line seen nearly end-on (the two points overlap) has no meaningful angle."""
        w = np.linalg.norm(xy[frames, J[a]] - xy[frames, J[b]], axis=1) / np.maximum(torso[frames], 1e-6)
        if np.nanmin(w) < MIN_LINE_WIDTH:
            what = "shoulders" if "shoulder" in a else "hips"
            return (f"From this camera angle the {what} overlap in the picture (seen end-on), so their line can't be "
                    "measured. Film the second clip from directly behind the stumps, facing the bowler.")
        return None

    def m(key: str, value: float, frame: int, conf: tuple[str, str], **extra) -> Measurement:
        return Measurement(key=key, value=float(value), unit=rule(key)["unit"], view=view, frame=int(frame),
                           event=METHODS[key]["event"], confidence=conf[0], confidence_reason=conf[1], measured=True,
                           joint=resolve_joint(kin, METHODS[key]["joint"]), extra=extra)

    def conf(joints: list[str], frames: list[int], evs: list, cap: Optional[str] = None, why: Optional[str] = None):
        if view == "diagonal":
            return c.conf(joints, frames, evs, diag_cap, diag_reason if why is None else f"{diag_reason} {why}")
        return c.conf(joints, frames, evs, cap, why)

    def diagonal_skip(key: str) -> Optional[str]:
        if view == "diagonal" and key not in DIAGONAL_OK:
            return ("The second clip is filmed diagonally; this needs the camera on the line of the run (behind the "
                    "stumps facing the bowler, or behind the bowler).")
        return None

    # image-x direction towards the bowler's non-bowling (front-leg) side, from the model's anatomical hip labels:
    # facing the camera the bowler's right is on the image left, from behind it is on the image right
    win = list(range(bfc, rel + 1))
    dx = xy[win, J[f"{fl}_hip"], 0] - xy[win, J[f"{bl}_hip"], 0]
    dx = dx[np.isfinite(dx)]
    side = float(np.sign(np.median(dx))) if len(dx) and abs(np.median(dx)) > 0.05 * np.nanmedian(torso[win]) else 0.0
    no_side = ("The hips overlap in the picture, so the bowler's left and right can't be told apart from this angle.")

    k = "hip_shoulder_separation_bfc"
    err = diagonal_skip(k) or c.ok(sh + hp, [bfc]) or line_visible(*sh, [bfc]) or line_visible(*hp, [bfc])
    if err:
        out.append(_not(k, view, err))
    else:
        v = line_to_line_angle(xy[bfc, J["left_shoulder"]], xy[bfc, J["right_shoulder"]],
                               xy[bfc, J["left_hip"]], xy[bfc, J["right_hip"]])
        out.append(m(k, v, bfc, conf(sh + hp, [bfc], [E["BFC"]], rot_cap, rot_reason)))

    k = "shoulder_counter_rotation"
    window = list(range(bfc, ffc + 1))
    err = diagonal_skip(k) or c.ok(sh, window) or line_visible(*sh, window)
    if err:
        out.append(_not(k, view, err))
    else:
        ang = line_angle(xy[:, J["left_shoulder"]], xy[:, J["right_shoulder"]])
        dev = np.abs(wrap_deg(ang[bfc:ffc + 1] - ang[bfc]))
        dev = np.minimum(dev, 180.0 - dev)  # a line has no direction: 170° of "rotation" is really 10°
        i = int(np.argmax(dev))
        out.append(m(k, float(dev[i]), bfc + i, conf(sh, [bfc, bfc + i], [E["BFC"], E["FFC"]], rot_cap, rot_reason)))

    k = "trunk_lateral_flexion_release"
    err = c.ok(sh + hp, [rel])
    if err:
        out.append(_not(k, view, err))
    else:
        v = abs(float(segment_angle_vs_vertical(kin.mid("left_hip", "right_hip")[rel], kin.mid("left_shoulder", "right_shoulder")[rel])))
        out.append(m(k, v, rel, conf(sh + hp, [rel], [E["release"]])))

    # head falling away
    k = "head_lateral_offset_release"
    ank = f"{fl}_ankle"
    hpnt, hj = _head(kin, rel)
    err = (None if hpnt is not None else f"The head is not visible at frame {rel}.") or c.ok([ank], [ffc]) \
        or (no_side if side == 0 else None)
    if err:
        out.append(_not(k, view, err))
    else:
        out.append(m(k, (hpnt[0] - xy[ffc, J[ank], 0]) * side / body_h[rel] * 100.0, rel,
                     conf(hj + [ank], [rel], [E["release"]])))

    # front foot alignment
    k = "front_foot_alignment"
    bank = f"{bl}_ankle"
    err = diagonal_skip(k) or c.ok([bank], [bfc]) or c.ok([ank], [ffc]) or (no_side if side == 0 else None)
    if err:
        out.append(_not(k, view, err))
    else:
        cx = kin.width / 2.0
        # perspective: sideways offset from the camera axis divided by the body's size at that moment
        lat_b = (xy[bfc, J[bank], 0] - cx) / body_h[bfc]
        lat_f = (xy[ffc, J[ank], 0] - cx) / body_h[ffc]
        out.append(m(k, (lat_f - lat_b) * -side * 100.0, ffc,
                     conf([bank, ank], [bfc, ffc], [E["BFC"], E["FFC"]], "Medium",
                          "The two feet are at different distances from the camera; their positions are compared in body "
                          "heights at each moment.")))

    # bowling arm from vertical
    k = "bowling_arm_angle_release"
    bs, bw = f"{arm}_shoulder", f"{arm}_wrist"
    err = c.ok([bs, bw], [rel]) or (no_side if side == 0 else None)
    if err:
        out.append(_not(k, view, err))
    else:
        d = xy[rel, J[bw]] - xy[rel, J[bs]]
        out.append(m(k, float(np.degrees(np.arctan2(d[0] * -side, d[1]))), rel, conf([bs, bw], [rel], [E["release"]])))

    # front knee caving inwards
    k = "front_knee_inward_release"
    fh, fk = f"{fl}_hip", f"{fl}_knee"
    err = diagonal_skip(k) or c.ok([fh, fk, ank], [rel]) or (no_side if side == 0 else None)
    if err:
        out.append(_not(k, view, err))
    else:
        h, kn, a = xy[rel, J[fh]], xy[rel, J[fk]], xy[rel, J[ank]]
        t = np.clip(np.dot(kn - h, a - h) / max(np.dot(a - h, a - h), 1e-6), 0, 1)
        off = (kn - (h + t * (a - h)))[0]
        leg = np.linalg.norm(kn - h) + np.linalg.norm(a - kn)
        out.append(m(k, off * -side / max(leg, 1e-6) * 100.0, rel, conf([fh, fk, ank], [rel], [E["release"]])))

    # run-up line drift
    k = "runup_line_drift"
    w0 = max(0, bfc - int(round(0.5 * kin.fps)))
    mh = kin.mid("left_hip", "right_hip")
    frames = [f for f in range(w0, bfc + 1) if np.all(np.isfinite(mh[f])) and np.isfinite(body_h[f])]
    err = diagonal_skip(k)
    if err:
        out.append(_not(k, view, err))
    elif len(frames) < max(4, int(0.3 * kin.fps)):
        out.append(_not(k, view, "Less than 0.3 s of the run-up is visible before back foot contact in the second clip."))
    else:
        lat = np.array([(mh[f, 0] - kin.width / 2.0) / body_h[f] for f in frames])
        out.append(m(k, abs(float(lat[-1] - lat[0])) * 100.0, frames[0],
                     conf(hp, [frames[0], frames[-1]], [E["BFC"]], "Medium",
                          "Run-up drift is estimated from the body's size in the picture.")))
    return out


def _second_body_height(kin: Kinematics) -> np.ndarray:
    """Body height in pixels per frame (shin + thigh + trunk + neck→nose), interpolated over gaps. In the second clip
    the bowler's size changes as he runs towards or away from the camera, so every frame gets its own scale."""
    def seg(a, b):
        return np.linalg.norm(kin.xy[:, J[a]] - kin.xy[:, J[b]], axis=1)
    leg = np.fmax(seg("left_hip", "left_knee") + seg("left_knee", "left_ankle"),
                  seg("right_hip", "right_knee") + seg("right_knee", "right_ankle"))
    trunk = np.linalg.norm(kin.mid("left_shoulder", "right_shoulder") - kin.mid("left_hip", "right_hip"), axis=1)
    neck = np.linalg.norm(kin.xy[:, J["nose"]] - kin.mid("left_shoulder", "right_shoulder"), axis=1)
    h = (leg + trunk + neck) * 1.07
    ok = np.isfinite(h)
    if ok.sum() < 2:
        return np.where(ok, h, np.nan)
    idx = np.arange(len(h))
    return np.interp(idx, idx[ok], h[ok])

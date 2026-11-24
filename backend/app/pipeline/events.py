"""Stage 5: rule-based delivery events: back foot contact (BFC), front foot contact (FFC) and release.

FFC  – after the last airborne phase, the first frame where the front ankle's horizontal speed drops > 60 % versus its
       approach speed (peak over the previous 0.1 s, at least 3 frames) and its vertical speed is ≈ 0 (allowing up to
       60 ms for heel-to-flat settling). The longer reference makes gradual real-world stops detectable.
BFC  – the same rule on the back ankle, before FFC.
Release – the frame of maximum bowling-wrist height within 150 ms after FFC.

Works on mirrored, Y-up kinematics (bowling towards +x). For a front-on clip (mode="front") the travel is towards the
camera, so the speed used is the full 2D image speed instead of the horizontal component.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from app.core.errors import FILMING_FIX, BowlIQError
from app.pipeline.params import P, frames
from app.pipeline.types import DeliveryEvents, EventResult, Kinematics

# every threshold comes from config/pipeline.yaml (seconds, body heights, fractions): see the comments there
C, R, ST = P.contacts, P.release, P.stride
MIN_N = P.general.min_samples
DROP_RATIO = C.drop_ratio
MOVE_THR = C.move_bh_s
STILL_VY = C.still_vy_bh_s
RELEASE_WINDOW_S = R.window_after_ffc_s
MIN_FFC_RELEASE_S = ST.ffc_release_min_s
SETTLE_S = C.settle_s
BFC_FFC_MIN_S, BFC_FFC_MAX_S = ST.bfc_ffc_min_s, ST.bfc_ffc_max_s
RELEASE_ABOVE_HEAD = R.above_head_bh
MIN_RUNUP = ST.runup_min_bh_s

EVENTS_FIX = (
    "We need the feet and bowling arm clearly visible through the delivery stride. " + FILMING_FIX
)


def _fwd_speed(p: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(p, axis=0) * fps
    d = np.vstack([d, d[-1:]]) if len(d) else np.zeros_like(p)
    return np.abs(d[:, 0]), np.abs(d[:, 1])


def find_contacts(ankle: np.ndarray, fps: float, body_h: Union[float, np.ndarray], mode: str = "side",
                  lo: int = 0, hi: int | None = None) -> list[tuple[int, float]]:
    """All foot-contact frames in [lo, hi] with their sharpness (0–1, fraction of speed lost)."""
    n = len(ankle)
    hi = n - 1 if hi is None else min(hi, n - 1)
    vx, vy = _fwd_speed(ankle, fps)
    sp = np.hypot(vx, vy) if mode == "front" else vx
    bh = np.broadcast_to(np.asarray(body_h, float), (n,))
    settle = frames(SETTLE_S, fps)
    out: list[tuple[int, float]] = []
    # approach speed = swing peak over the reference window (in seconds, so the same at every frame rate)
    # … and at least the span of one smoothed speed sample (smoothing window + the frame difference): smoothing spreads a
    # stop over that many frames (6 frames at 30 fps)
    from app.pipeline.smooth import sg_window
    ref_n = max(frames(C.ref_window_s, fps, MIN_N), sg_window(fps, P.smoothing.body_window_s * 1000) + 1)
    sup_n = frames(C.refractory_s, fps)  # two contacts of one foot can't be closer than this
    nxt_n = frames(C.next_s, fps)
    for f in range(max(lo, MIN_N), hi + 1):
        # a foot lands again only after it has swung: the approach speed is measured since its previous contact
        start = max(0, f - ref_n, (out[-1][0] + 1) if out else 0)
        prev_v = sp[start:f]
        prev_v = prev_v[np.isfinite(prev_v)]
        if len(prev_v) < MIN_N or not np.isfinite(sp[f]) or not np.isfinite(bh[f]):
            continue
        ref = float(prev_v.max())  # approach speed: the swing peak over the last ~0.1 s (robust to a gradual stop)
        if ref < MOVE_THR * bh[f]:
            continue
        if sp[f] >= DROP_RATIO * ref:
            continue
        if mode == "side":
            # heel strike → foot flat: the ankle may still sink for a few frames after it stops moving forward,
            # so vertical speed must settle to ≈ 0 within SETTLE_S of the horizontal stop
            win = vy[f:f + settle + 1]
            win = win[np.isfinite(win)]
            if not len(win) or win.min() >= STILL_VY * bh[f]:
                continue
        nxt = sp[f + 1:f + 1 + nxt_n]
        nxt = nxt[np.isfinite(nxt)]
        if len(nxt) and nxt.mean() > C.next_ratio * ref:  # still moving at nearly full speed: not a contact (pivoting is fine)
            continue
        if out and f - out[-1][0] <= sup_n:
            continue
        out.append((f, float(1.0 - sp[f] / ref)))
    return out


def _confidence(fps: float, vis: float, sharp: float, extra: str | None = None) -> tuple[str, str]:
    """Single-method confidence (refined by the multi-method consensus in consensus.py)."""
    parts = []
    q = P.quality
    if fps >= q.fps_good and vis >= 0.8 and sharp >= 0.75:
        level = "High"
    elif fps >= q.fps_ok and vis >= 0.6 and sharp >= 0.6:
        level = "Medium"
    else:
        level = "Low"
    if fps < q.fps_ok:
        level = "Low"
    if extra:
        level = "Low" if level != "High" else "Medium"
    parts.append(f"{fps:.0f} fps")
    parts.append("joint clearly visible" if vis >= 0.8 else ("joint partly visible" if vis >= 0.6 else "joint poorly visible"))
    parts.append("sharp detection" if sharp >= 0.75 else ("clear detection" if sharp >= 0.6 else "gradual change, timing approximate"))
    if extra:
        parts.append(extra)
    return level, ", ".join(parts) + "."


def head_y(kin: Kinematics, f: int, body_h: float) -> float:
    """Head height (Y up) at frame f: nose, else eyes/ears, else shoulder line + 10 % of body height.

    Some pose models (rightly) mark the face as not visible when the bowler turns away at release."""
    nose = kin.joint("nose")[f, 1]
    if np.isfinite(nose):
        return float(nose)
    face = [kin.joint(n)[f, 1] for n in ("left_eye", "right_eye", "left_ear", "right_ear")]
    face = [v for v in face if np.isfinite(v)]
    if face:
        return float(np.mean(face))
    sh = [kin.joint(n)[f, 1] for n in ("left_shoulder", "right_shoulder")]
    sh = [v for v in sh if np.isfinite(v)]
    return float(np.mean(sh) + 0.10 * body_h) if sh else float("nan")


def _check_release(kin: Kinematics, wy: np.ndarray, rel_f: int, bh_arr: np.ndarray) -> None:
    """At release the bowling hand is well above the head (≥ 10 % of body height above it)."""
    hy = head_y(kin, rel_f, float(bh_arr[rel_f]))
    if not np.isfinite(hy) or not np.isfinite(wy[rel_f]) or wy[rel_f] - hy < RELEASE_ABOVE_HEAD * bh_arr[rel_f]:
        raise BowlIQError("EVENTS_NOT_FOUND",
                          "No bowling action was found: the hand never goes clearly above the head after a front foot landing.",
                          EVENTS_FIX)


def _running_gait(kin: Kinematics, a: int, b: int, bh: float) -> bool:
    """A camera that follows the bowler keeps his hips still in the picture, but the feet still swing well forward
    and back relative to the hips while running: both feet must sweep ≥ 25 % of body height, one ≥ 40 %."""
    hip_x = kin.mid("left_hip", "right_hip")[a:b + 1, 0]
    sweeps = []
    for side in ("left", "right"):
        rel = kin.joint(f"{side}_ankle")[a:b + 1, 0] - hip_x
        rel = rel[np.isfinite(rel)]
        sweeps.append(float(np.ptp(rel)) if len(rel) >= 4 else 0.0)
    return min(sweeps) >= ST.gait_sweep_min_bh * bh and max(sweeps) >= ST.gait_sweep_max_bh * bh


def _check_runup(kin: Kinematics, ffc_f: int, fps: float, bh_arr: np.ndarray) -> None:
    """Side-on: the hips travel towards the batter before front foot contact."""
    hip_x = kin.mid("left_hip", "right_hip")[:, 0]
    a = max(0, ffc_f - frames(ST.runup_window_s, fps))
    seg = np.arange(a, ffc_f + 1)
    seg = seg[np.isfinite(hip_x[seg])]
    speed = float(np.polyfit(seg / fps, hip_x[seg], 1)[0]) if len(seg) >= max(4, MIN_N) else 0.0
    if speed < MIN_RUNUP * bh_arr[ffc_f] and not _running_gait(kin, a, ffc_f, bh_arr[ffc_f]):
        raise BowlIQError("EVENTS_NOT_FOUND",
                          "No run-up was found: the bowler isn't moving towards the batter before front foot contact.",
                          "Include the last 3–4 run-up strides in the clip. " + FILMING_FIX)


# ───────────────────────────── release-first search ─────────────────────────────
# Release is the most distinctive moment of a delivery: the bowling hand comes up from the coil to a sharp, high peak
# and drops straight into the follow-through. The bound before it can also have both hands above the head, but there
# the hands stay up (a plateau) instead of rising and falling sharply. Found on real 30 fps clips: anchoring release
# to "the highest wrist within 150 ms after a front foot contact" picked the bound (and the wrong arm) whenever the
# contact was detected a few frames early.
PEAK_HALF_S = R.peak_half_s  # a release peak is the highest wrist point within ±this
RISE_S, RISE_BH = R.rise_s, R.rise_bh  # the hand rises ≥ RISE_BH in RISE_S before release …
DROP_S, DROP_BH = R.drop_s, R.drop_bh  # … and falls ≥ DROP_BH in DROP_S after (follow-through)
FFC_BEFORE_REL_MAX_S = ST.ffc_release_max_s  # front foot contact comes 40–300 ms before release


def release_candidates(kin: Kinematics, wy: np.ndarray, bh: np.ndarray) -> list[tuple[int, float]]:
    """Sharp, high bowling-wrist peaks with a rise before and a follow-through drop after, best first."""
    fps, n = kin.fps, kin.n
    k = frames(PEAK_HALF_S, fps)
    r_n, d_n = frames(RISE_S, fps, 2), frames(DROP_S, fps, 2)
    out = []
    for r in range(1, n - 1):
        if not np.isfinite(wy[r]) or not np.isfinite(bh[r]):
            continue
        win = wy[max(0, r - k):r + k + 1]
        if wy[r] < np.nanmax(win):
            continue
        hy = head_y(kin, r, float(bh[r]))
        if not np.isfinite(hy) or wy[r] - hy < RELEASE_ABOVE_HEAD * bh[r]:
            continue
        before, after = wy[max(0, r - r_n):r], wy[r + 1:r + 1 + d_n]
        before, after = before[np.isfinite(before)], after[np.isfinite(after)]
        if len(before) < 2 or len(after) < 2:
            continue
        rise, drop = wy[r] - before.min(), wy[r] - after.min()
        if rise < RISE_BH * bh[r] or drop < DROP_BH * bh[r]:
            continue
        out.append((r, float((wy[r] - hy) / bh[r] + min(rise, drop) / bh[r])))
    return sorted(out, key=lambda c: c[1], reverse=True)


def detect_events(kin: Kinematics, body_h: Union[float, np.ndarray], mode: str = "side") -> DeliveryEvents:
    if mode == "side":
        try:
            return _detect_release_first(kin, body_h)
        except BowlIQError:
            pass  # e.g. the follow-through isn't in the clip: fall back to the contact-first search
    return _detect_contact_first(kin, body_h, mode)


REL_FIRST_BFC_FFC_MIN_S = ST.bfc_ffc_min_s  # back foot → front foot takes at least this in a real delivery stride


def _detect_release_first(kin: Kinematics, body_h: Union[float, np.ndarray]) -> DeliveryEvents:
    for ev in release_first_all(kin, body_h):
        return ev
    raise AssertionError("unreachable")  # release_first_all raises when nothing is found


def release_first_all(kin: Kinematics, body_h: Union[float, np.ndarray]):
    """Every delivery (BFC, FFC, release) that passes the release-first checks, best release peak first. Raises
    EVENTS_NOT_FOUND (with the first reason) if none does."""
    found = False
    fps, n = kin.fps, kin.n
    wrist_name = kin.side("wrist", "bowling")
    wy = kin.joint(wrist_name)[:, 1]
    bh = np.broadcast_to(np.asarray(body_h, float), (n,))
    # The pose model sometimes labels the legs the wrong way round for a whole clip, so the front foot is decided by
    # position: at release it is the foot further forward (towards the batter, +x).
    names = [kin.side("ankle", "front"), kin.side("ankle", "back")]
    pos = {nm: kin.joint(nm) for nm in names}
    contacts = {nm: find_contacts(pos[nm], fps, body_h, "side") for nm in names}
    min_gap = frames(MIN_FFC_RELEASE_S, fps)
    max_gap = max(min_gap + 1, frames(FFC_BEFORE_REL_MAX_S, fps))
    first_error: BowlIQError | None = None
    for rel_f, rel_score in release_candidates(kin, wy, bh):
        xa, xb = pos[names[0]][rel_f, 0], pos[names[1]][rel_f, 0]
        if np.isfinite(xa) and np.isfinite(xb) and xb > xa:
            front_name, back_name = names[1], names[0]
        else:
            front_name, back_name = names[0], names[1]
        ff = [(f, sh) for f, sh in contacts[front_name] if rel_f - max_gap <= f <= rel_f - min_gap]
        if not ff:
            first_error = first_error or BowlIQError("EVENTS_NOT_FOUND", "We couldn't find the front foot landing "
                                                     "just before the release.", EVENTS_FIX)
            continue
        # the front foot lands once and stays planted: its first contact in the window is the landing (later "stops"
        # are the planted foot settling or pivoting)
        ffc_f, ffc_sharp = ff[0]
        try:
            _check_runup(kin, ffc_f, fps, bh)
        except BowlIQError as exc:
            first_error = first_error or exc
            continue
        lo = max(0, ffc_f - frames(BFC_FFC_MAX_S, fps))
        hi = ffc_f - frames(REL_FIRST_BFC_FFC_MIN_S, fps)
        bb = [(f, sh) for f, sh in contacts[back_name] if lo <= f <= hi]
        if not bb:
            first_error = first_error or BowlIQError("EVENTS_NOT_FOUND", "We couldn't find the back foot landing "
                                                     "before front foot contact.", EVENTS_FIX)
            continue
        bfc_f, bfc_sharp = bb[-1]

        def vis(name: str, f: int) -> float:
            return float(kin.joint_vis(name)[max(0, f - 1):f + 2].mean())

        bl, br = _confidence(fps, vis(back_name, bfc_f), bfc_sharp)
        fl, fr = _confidence(fps, vis(front_name, ffc_f), ffc_sharp)
        rl, rr = _confidence(fps, vis(wrist_name, rel_f), float(np.clip(rel_score / 0.8, 0, 1)))
        found = True
        yield DeliveryEvents(bfc=EventResult("BFC", bfc_f, bl, br, bfc_sharp),
                             ffc=EventResult("FFC", ffc_f, fl, fr, ffc_sharp),
                             release=EventResult("release", rel_f, rl, rr, rel_score), method="release_first",
                             legs_swapped=front_name != names[0])
    if found:
        return
    raise first_error or BowlIQError("EVENTS_NOT_FOUND", "No bowling action was found: the hand never comes over the "
                                     "top and down into a follow-through.", EVENTS_FIX)


def _detect_contact_first(kin: Kinematics, body_h: Union[float, np.ndarray], mode: str = "side") -> DeliveryEvents:
    fps, n = kin.fps, kin.n
    # Foot contact is timed from the ankles. (Heel points were evaluated on real clips and didn't improve timing:
    # on compressed phone video the heel point follows the ankle.)
    front_name = kin.side("ankle", "front")
    back_name = kin.side("ankle", "back")
    wrist_name = kin.side("wrist", "bowling")
    front, back, wrist = kin.joint(front_name), kin.joint(back_name), kin.joint(wrist_name)
    wy = wrist[:, 1]
    bh_arr = np.broadcast_to(np.asarray(body_h, float), (n,))
    win = frames(RELEASE_WINDOW_S, fps, 2)

    fronts = find_contacts(front, fps, body_h, mode)
    if not fronts:
        raise BowlIQError("EVENTS_NOT_FOUND", "We couldn't find the front foot landing in this clip.", EVENTS_FIX)

    # "still rising after the landing" needs at least two samples after the contact to be seen at all
    min_gap = frames(MIN_FFC_RELEASE_S, fps, 2)

    def release_after(c: int) -> tuple[int, float]:
        seg = wy[c + 1:min(n, c + win + 1)]
        if len(seg) == 0 or not np.isfinite(seg).any():
            return -1, -np.inf
        k = int(np.nanargmax(seg))
        # the wrist must still be rising after the landing: a peak right at the start of the window means the arm
        # had already come over before this "landing", so it can't be front foot contact
        if k + 1 < min_gap:
            return -1, -np.inf
        return c + 1 + k, float(seg[k])

    cands = [(c, s, *release_after(c)) for c, s in fronts]
    cands = [c for c in cands if c[2] >= 0]
    if not cands:
        raise BowlIQError("EVENTS_NOT_FOUND", "We couldn't see the bowling arm after front foot contact.", EVENTS_FIX)
    # Try candidates from most to least plausible: highest wrist in the 150 ms window first; for the same release
    # frame, the latest landing (after the last airborne phase). A candidate that fails a plausibility gate no
    # longer ends detection: the next one is tried.
    by_release: dict[int, tuple] = {}
    for c in cands:
        if c[2] not in by_release or c[0] > by_release[c[2]][0]:
            by_release[c[2]] = c
    ordered = sorted(by_release.values(), key=lambda c: c[3], reverse=True)
    first_error: BowlIQError | None = None
    chosen = None
    for ffc_f, ffc_sharp, rel_f, _ in ordered:
        try:
            _check_release(kin, wy, rel_f, bh_arr)
            if mode == "side":
                _check_runup(kin, ffc_f, fps, bh_arr)
            lo = max(0, ffc_f - frames(BFC_FFC_MAX_S, fps))
            hi = ffc_f - frames(BFC_FFC_MIN_S, fps)
            backs = find_contacts(back, fps, body_h, mode, lo, hi)
            if not backs:
                raise BowlIQError("EVENTS_NOT_FOUND", "We couldn't find the back foot landing before front foot contact.",
                                  EVENTS_FIX)
            chosen = (ffc_f, ffc_sharp, rel_f, *backs[-1])
            break
        except BowlIQError as exc:
            first_error = first_error or exc
    if chosen is None:
        raise first_error or BowlIQError("EVENTS_NOT_FOUND", "We couldn't find the delivery stride.", EVENTS_FIX)
    ffc_f, ffc_sharp, rel_f, bfc_f, bfc_sharp = chosen

    if not (bfc_f < ffc_f < rel_f):
        raise BowlIQError("EVENTS_NOT_FOUND", "The detected key moments are out of order.", EVENTS_FIX)

    # release sharpness: how far the wrist drops either side within ±peak_half_s, vs 2 % of body height
    k = frames(PEAK_HALF_S, fps)
    left = wy[max(0, rel_f - k)]
    right = wy[min(n - 1, rel_f + k)]
    drop = wy[rel_f] - np.nanmax([left, right])
    rel_sharp = float(np.clip(drop / (0.02 * bh_arr[rel_f]), 0, 1)) if np.isfinite(drop) else 0.0
    rel_extra = None
    if rel_f == n - 1:
        rel_extra = "the clip ends at release"
    elif rel_f == ffc_f + win:
        rel_extra = "the wrist was still rising at the end of the 150 ms window"

    def vis(name: str, f: int) -> float:
        return float(kin.joint_vis(name)[max(0, f - 1):f + 2].mean())

    bl, br = _confidence(fps, vis(back_name, bfc_f), bfc_sharp)
    fl, fr = _confidence(fps, vis(front_name, ffc_f), ffc_sharp)
    rl, rr = _confidence(fps, vis(wrist_name, rel_f), rel_sharp, rel_extra)
    return DeliveryEvents(
        bfc=EventResult("BFC", bfc_f, bl, br, bfc_sharp),
        ffc=EventResult("FFC", ffc_f, fl, fr, ffc_sharp),
        release=EventResult("release", rel_f, rl, rr, rel_sharp),
    )


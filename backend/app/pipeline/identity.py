"""Limb identity: undo left/right label swaps, then give limbs ROLES.

1. Swap correction. Pose models sometimes label the two legs (or arms) the wrong way round for a few frames,
   especially side-on, where one limb hides the other. Each frame, both limbs are predicted from their last two
   corrected positions (constant velocity) and the detected limbs are assigned to the predictions with the Hungarian
   algorithm; labels are swapped only when that is clearly cheaper (a cost-ratio threshold, the hysteresis), so they
   can't flicker on near-ties. Every correction is logged (frames + reason) for the diagnostics.
   (Tried and measured on the 14 labelled clips: a whole-clip dynamic-programming version with switching penalties
   was more sensitive to its penalties and broke clips this version gets right, so it isn't used.)

2. Roles. After the key moments are found: front_leg = the leg whose foot LANDS at FFC (swinging just before, still
   just after), back_leg = the one landing at BFC, bowling_arm from the clip context. Kinematics.side() then answers "front"/"back"/"bowling" from these
   roles, so every metric uses front_knee / back_ankle / bowling_wrist rather than raw left/right labels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.pipeline.params import P, frames
from app.pipeline.types import LEG_PARTS, J

IDC = P.identity
GROUPS = {"legs": LEG_PARTS, "arms": ("elbow", "wrist")}


def _limb(kp: np.ndarray, f: int, side: str, parts: tuple[str, ...]) -> np.ndarray:
    return kp[f, [J[f"{side}_{p}"] for p in parts], :2]


def _cost(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a - b, axis=1)
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else np.nan


def _predict(hist: list[tuple[int, np.ndarray, np.ndarray]], f: int, max_gap: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not hist:
        return None
    f1, l1, r1 = hist[-1]
    if f - f1 > max_gap:
        return None
    if len(hist) >= 2 and f1 - hist[-2][0] <= max_gap:
        f0, l0, r0 = hist[-2]
        k = (f - f1) / max(1, f1 - f0)
        return l1 + (l1 - l0) * k, r1 + (r1 - r0) * k
    return l1, r1


PAIRS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "foot_index")


def suppress_duplicates(kp: np.ndarray, body_h: float) -> tuple[np.ndarray, list[dict]]:
    """A body has two wrists (knees, ankles…). When the left and right labels sit on the same point (within
    identity.duplicate_bh body heights) AND one of them got there by jumping away from where its own motion predicts
    it (by more than identity.duplicate_jump_bh), that label is wrong — typically both collapse onto the blurred
    bowling hand. It is marked unseen (then bridged from its true neighbours by the gap filling). Genuine overlaps
    (hips and shoulders side-on, legs crossing) move smoothly and are left alone."""
    kp = kp.copy()
    n = kp.shape[0]
    thr, jump = IDC.duplicate_bh * body_h, IDC.duplicate_jump_bh * body_h
    log: list[dict] = []
    for part in PAIRS:
        a, b = J[f"left_{part}"], J[f"right_{part}"]
        hist: dict[int, list[tuple[int, np.ndarray]]] = {a: [], b: []}

        def pred_err(j: int, f: int, p: np.ndarray, hist=hist) -> float:
            h = hist[j]
            if not h:
                return np.inf
            f1, p1 = h[-1]
            if len(h) >= 2:
                f0, p0 = h[-2]
                q = p1 + (p1 - p0) * (f - f1) / max(1, f1 - f0)
            else:
                q = p1
            return float(np.linalg.norm(p - q))
        for f in range(n):
            pa, pb = kp[f, a, :2], kp[f, b, :2]
            if np.isfinite(pa).all() and np.isfinite(pb).all() and np.linalg.norm(pa - pb) < thr:
                ea, eb = pred_err(a, f, pa), pred_err(b, f, pb)
                bad = a if ea > eb else b
                if max(ea, eb) > jump and np.isfinite(min(ea, eb)):
                    kp[f, bad, 2] = 0.0
                    side = "left" if bad == a else "right"
                    if log and log[-1]["joint"] == f"{side}_{part}" and log[-1]["frames"][1] == f - 1:
                        log[-1]["frames"][1] = f
                    else:
                        log.append({"joint": f"{side}_{part}", "frames": [f, f],
                                    "reason": f"Left and right {part} labels on the same point; the {side} one jumped "
                                              "there against its own motion, so it was treated as unseen."})
            for j in (a, b):
                if np.isfinite(kp[f, j, :2]).all() and kp[f, j, 2] > 0:
                    hist[j].append((f, kp[f, j, :2].copy()))
                    if len(hist[j]) > 2:
                        hist[j].pop(0)
    return kp, log


COST_PARTS = {"legs": ("knee", "ankle"), "arms": ("elbow", "wrist")}  # the steadiest points decide; all parts swap


def correct_swaps(kp: np.ndarray, fps: float, body_h: float) -> tuple[np.ndarray, list[dict]]:
    """→ (keypoints with swaps undone, log [{limb, frames: [first, last], reason, corrected}]).

    For each frame, both limbs are predicted from their last two corrected positions (constant velocity, so a fast
    bowling wrist is predicted where it is going, not where it was), and the two detected limbs are assigned to the
    two predictions with the Hungarian algorithm (cost: knee+ankle, or elbow+wrist). The labels are swapped only when
    the swapped assignment is clearly cheaper — cost ratio below identity.swap_ratio (the hysteresis: near-ties, e.g.
    legs crossing, keep the labels as they are, so they can't flicker). Whole legs (knee, ankle, heel, toe) or arms
    (elbow, wrist) swap together. Every corrected stretch is logged with its frames and the reason."""
    kp = kp.copy()
    n = kp.shape[0]
    predict_n, hold_n = frames(IDC.predict_s, fps), frames(IDC.max_gap_s, fps)
    log: list[dict] = []
    for limb, parts in GROUPS.items():
        cparts = COST_PARTS[limb]
        L = [J[f"left_{p}"] for p in parts]
        R = [J[f"right_{p}"] for p in parts]
        cl = [J[f"left_{p}"] for p in cparts]
        cr = [J[f"right_{p}"] for p in cparts]
        hist: list[tuple[int, np.ndarray, np.ndarray]] = []
        for f in range(n):
            lp, rp = kp[f, cl, :2], kp[f, cr, :2]
            if not (np.isfinite(lp).all() and np.isfinite(rp).all()):
                continue
            if hist:
                f1, l1, r1 = hist[-1]
                if len(hist) >= 2 and f - f1 <= predict_n:
                    f0, l0, r0 = hist[-2]
                    k = (f - f1) / max(1, f1 - f0)
                    pl, pr = l1 + (l1 - l0) * k, r1 + (r1 - r0) * k
                else:
                    pl, pr = l1, r1
                if f - f1 <= hold_n:
                    C = np.array([[np.linalg.norm(lp - pl, axis=1).sum(), np.linalg.norm(lp - pr, axis=1).sum()],
                                  [np.linalg.norm(rp - pl, axis=1).sum(), np.linalg.norm(rp - pr, axis=1).sum()]])
                    _rows, cols = linear_sum_assignment(C)
                    keep, swap = C[0, 0] + C[1, 1], C[0, 1] + C[1, 0]
                    if cols[0] == 1 and swap < IDC.swap_ratio * keep:
                        tmp = kp[f, L].copy()
                        kp[f, L] = kp[f, R]
                        kp[f, R] = tmp
                        lp, rp = rp, lp
                        _note(log, limb, f, keep / max(swap, 1e-9))
            hist.append((f, lp.copy(), rp.copy()))
    return kp, log


def _note(log: list[dict], limb: str, f: int, ratio: float) -> None:
    if log and log[-1]["limb"] == limb and log[-1]["frames"][1] == f - 1:
        log[-1]["frames"][1] = f
        log[-1]["fit_ratio"] = round(max(log[-1]["fit_ratio"], ratio), 2)
    else:
        log.append({"limb": limb, "frames": [f, f], "fit_ratio": round(ratio, 2), "corrected": True})
    e = log[-1]
    e["reason"] = (f"Left/right {limb} labels swapped back on frames {e['frames'][0]}–{e['frames'][1]}: the "
                   f"swapped labels fit the limbs' predicted motion {e['fit_ratio']:.1f}× better (needed: "
                   f"{1 / IDC.swap_ratio:.1f}×).")


# ───────────────────────────── roles ─────────────────────────────

def _landing(kin, side: str, f: int, fps: float) -> Optional[float]:
    """How clearly this foot LANDS at frame f: its speed just before f minus just after (body-height units per s).
    The landing foot was swinging and stops; a foot already planted (or still in the air) shows no such drop."""
    pts = [kin.joint(f"{side}_{p}") for p in ("heel", "foot_index", "ankle")]
    pts = [p for p in pts if np.isfinite(p[max(0, f - 2):f + 3]).all()]
    if not pts:
        return None
    p = np.mean(np.stack(pts), axis=0)
    w = frames(IDC.role_window_s, fps, 2)
    before, after = p[max(0, f - w):f + 1], p[f:min(kin.n, f + w + 1)]
    sp = lambda q: float(np.nanmean(np.linalg.norm(np.diff(q, axis=0), axis=1))) * fps if len(q) >= 2 else np.nan  # noqa: E731
    v = sp(before) - sp(after)
    return v if np.isfinite(v) else None


def _foot(kin, side: str) -> np.ndarray:
    """(F, 2) mean of the foot points (heel, toe, ankle) that are present; NaN where none are."""
    import warnings
    pts = np.stack([kin.joint(f"{side}_{p}") for p in ("heel", "foot_index", "ankle")])
    with warnings.catch_warnings():  # frames with no foot point at all are NaN, as intended
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(pts, axis=0)


def _speed(p: np.ndarray, fps: float) -> np.ndarray:
    d = np.diff(p, axis=0)
    v = np.linalg.norm(d, axis=1) * fps
    return np.concatenate([v, v[-1:]]) if len(v) else np.zeros(len(p))


def _vote(x: Optional[float], scale: float) -> float:
    """A cue's value → a vote in (−1, 1) (positive: the LEFT leg is the front leg); 0 when the cue is missing."""
    return 0.0 if x is None or not np.isfinite(x) else float(np.tanh(x / scale))


def _window_mean(x: np.ndarray, a: int, b: int) -> Optional[float]:
    seg = x[max(0, a):max(0, b)]
    seg = seg[np.isfinite(seg)]
    return float(np.mean(seg)) if len(seg) else None


def role_cues(kin, events, body_h: float) -> dict:
    """Label-independent evidence for which labelled leg is the FRONT leg, each as a vote (+ = left is front):

    ahead_ffc   from FFC to release the front foot is AHEAD of the back foot along the run (the run direction is
                measured from the hips, so a mis-detected direction can't flip it). The strongest cue: it holds even
                when the front leg is the far, half-hidden one (a right-arm bowler filmed from the bowling-arm side).
    planted_ffc after FFC the front foot stays put while the back foot drags or lifts.
    landing_ffc the front foot's speed drops at FFC.
    planted_bfc at BFC the BACK foot is the planted one (so the other is the front leg) — checked separately: if it
                disagrees with the FFC cues, the labels flipped between the two contacts.
    """
    fps = kin.fps
    bh = max(float(body_h), 1.0)
    ffc, bfc, rel = events.ffc.frame, events.bfc.frame, events.release.frame
    w = frames(IDC.role_window_s, fps, 2)
    fl, fr = _foot(kin, "left"), _foot(kin, "right")
    hip = kin.mid("left_hip", "right_hip")[:, 0]
    a0 = max(0, bfc - frames(0.4, fps, 3))
    run = np.diff(hip[a0:ffc + 1])
    run = run[np.isfinite(run)]
    travel = float(np.sign(np.median(run))) if len(run) and np.median(run) != 0 else 1.0
    b_end = max(ffc + w, min(rel, ffc + 3 * w)) + 1
    ahead = _window_mean(travel * (fl[:, 0] - fr[:, 0]) / bh, ffc, b_end)
    vl, vr = _speed(fl, fps) / bh, _speed(fr, fps) / bh
    planted = _window_mean(vr - vl, ffc, ffc + w + 1)  # right moving more → left planted → left is front
    sl, sr = _landing(kin, "left", ffc, fps), _landing(kin, "right", ffc, fps)
    landing = None if sl is None or sr is None else (sl - sr) / bh
    planted_b = _window_mean(vl - vr, bfc, bfc + w + 1)  # left moving more at BFC → right is back → left is front
    return {"ahead_ffc": _vote(ahead, IDC.ahead_scale_bh), "planted_ffc": _vote(planted, IDC.speed_scale_bh),
            "landing_ffc": _vote(landing, IDC.speed_scale_bh), "planted_bfc": _vote(planted_b, IDC.speed_scale_bh),
            "travel": travel}


def legs_crossing(kin, a: int, b: int) -> int:
    """The frame in [a, b) where the feet are closest along the run (the legs pass each other): where a pose model
    most often swaps their labels."""
    fl, fr = _foot(kin, "left")[:, 0], _foot(kin, "right")[:, 0]
    d = np.abs(fl[a:b] - fr[a:b])
    if not np.isfinite(d).any():
        return (a + b) // 2
    return a + int(np.nanargmin(d))


def flip_frame(kin, a: int, b: int) -> Optional[int]:
    """The frame t in (a, b] where the feet's labels most clearly swapped: each foot lands where the OTHER foot was a
    frame earlier (keeping the labels costs more motion than swapping them). None when no frame shows that, e.g.
    a swap exactly where the feet overlap."""
    fl, fr = _foot(kin, "left"), _foot(kin, "right")
    best, best_gain = None, 0.0
    for t in range(max(1, a + 1), min(kin.n, b + 1)):
        p0l, p0r, p1l, p1r = fl[t - 1], fr[t - 1], fl[t], fr[t]
        if not (np.isfinite(p0l).all() and np.isfinite(p0r).all() and np.isfinite(p1l).all() and np.isfinite(p1r).all()):
            continue
        keep = np.linalg.norm(p1l - p0l) + np.linalg.norm(p1r - p0r)
        swap = np.linalg.norm(p1l - p0r) + np.linalg.norm(p1r - p0l)
        if keep - swap > best_gain:
            best, best_gain = t, keep - swap
    return best


def assign_roles(kin, events, bowling_arm: str, body_h: float) -> dict:
    """{front_leg, back_leg, bowling_arm, confidence, reason, cues, relabel}. The front leg is decided at FFC from
    label-independent cues (role_cues), weighted ahead 2 : planted 1 : landing 1. If the BFC cue then says the
    labels at BFC point the other way, the labels flipped between the contacts: `relabel` = (start, end) frames whose
    leg labels must be swapped back (from the leg crossing before BFC to the crossing between BFC and FFC)."""
    ffc, bfc = events.ffc.frame, events.bfc.frame
    c = role_cues(kin, events, body_h)
    score = 2 * c["ahead_ffc"] + c["planted_ffc"] + c["landing_ffc"]
    default_front = "left" if bowling_arm == "right" else "right"
    if abs(score) < 1e-6:
        return {"front_leg": default_front, "back_leg": "right" if default_front == "left" else "left",
                "bowling_arm": bowling_arm, "confidence": "Low", "cues": c, "relabel": None,
                "reason": "The feet weren't visible enough at front foot contact; the front leg is assumed to be the one "
                          f"opposite the bowling arm ({default_front})."}
    front = "left" if score > 0 else "right"
    back = "right" if front == "left" else "left"
    strength = abs(score) / 4.0  # 1 = every cue fully agrees
    sign = 1 if front == "left" else -1
    agree_bfc = c["planted_bfc"] * sign  # > 0: the BFC labels agree with FFC
    relabel = None
    why_parts = []
    if c["ahead_ffc"] * sign > 0.3:
        why_parts.append(f"from front foot contact (frame {ffc}) to release the {front} foot is ahead along the run")
    if c["planted_ffc"] * sign > 0.3 or c["landing_ffc"] * sign > 0.3:
        why_parts.append(f"the {front} foot is the one that lands and stays planted at front foot contact")
    why = ("The front leg is the " + front + " leg: " + "; ".join(why_parts) + ".") if why_parts else \
        f"The {front} foot is the front foot at front foot contact (frame {ffc}), on weak evidence."
    if agree_bfc < -IDC.flip_vote and strength >= IDC.flip_strength:
        w = frames(IDC.role_window_s, kin.fps, 2)
        cross = flip_frame(kin, bfc, ffc) or legs_crossing(kin, bfc + w + 1, ffc + 1)
        a0 = max(0, bfc - frames(0.4, kin.fps, 3))
        start = flip_frame(kin, a0, bfc) or 0  # no earlier flip in view: wrong since the start of the clip
        relabel = (int(start), int(cross))
        why += (f" At back foot contact (frame {bfc}) the pose model had the legs' labels the other way round; they "
                f"were swapped back for frames {start}–{cross - 1} (between the leg crossings).")
        bfc_ok = True
    else:
        bfc_ok = agree_bfc > 0
        if bfc_ok:
            why += f" Consistent with back foot contact (frame {bfc}), where the {back} foot is the planted one."
    conf = "High" if strength >= IDC.role_strength and bfc_ok else ("Medium" if strength >= IDC.role_strength / 2 else "Low")
    if front != default_front:
        why += f" (The pose model's labels are the opposite of the usual for a {bowling_arm}-arm bowler, so the roles " \
               "follow the feet, not the labels.)"
    return {"front_leg": front, "back_leg": back, "bowling_arm": bowling_arm, "confidence": conf, "reason": why,
            "cues": {k: round(v, 3) for k, v in c.items()}, "relabel": relabel}


def swap_leg_labels(kin, start: int, end: int):
    """A copy of kin with left/right leg joints (hip, knee, ankle, heel, toe) exchanged on frames [start, end), the
    swap logged in identity_log."""
    import dataclasses
    xy, img, vis, valid = kin.xy.copy(), kin.img_xy.copy(), kin.vis.copy(), kin.valid.copy()
    f = slice(start, end)
    for part in ("hip",) + tuple(LEG_PARTS):
        a, b = J[f"left_{part}"], J[f"right_{part}"]
        for arr in (xy, img, vis, valid):
            arr[f, a], arr[f, b] = arr[f, b].copy(), arr[f, a].copy()
    entry = {"limb": "legs", "frames": [int(start), int(end - 1)], "corrected": True,
             "reason": "labels flipped between back and front foot contact; swapped back (role check)"}
    return dataclasses.replace(kin, xy=xy, img_xy=img, vis=vis, valid=valid,
                               identity_log=list(kin.identity_log) + [entry])

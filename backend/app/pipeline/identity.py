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


def assign_roles(kin, events, bowling_arm: str, body_h: float) -> dict:
    """{front_leg, back_leg, bowling_arm, confidence, reason}: the leg whose foot is planted at FFC is the front leg;
    it should also be the one NOT planted at BFC (checked)."""
    fps = kin.fps
    ffc, bfc = events.ffc.frame, events.bfc.frame
    sf = {s: _landing(kin, s, ffc, fps) for s in ("left", "right")}
    sb = {s: _landing(kin, s, bfc, fps) for s in ("left", "right")}
    default_front = "left" if bowling_arm == "right" else "right"
    if sf["left"] is None or sf["right"] is None:
        return {"front_leg": default_front, "back_leg": "right" if default_front == "left" else "left",
                "bowling_arm": bowling_arm, "confidence": "Low",
                "reason": "The feet weren't visible enough at front foot contact; the front leg is assumed to be the one "
                          f"opposite the bowling arm ({default_front})."}
    front = "left" if sf["left"] > sf["right"] else "right"  # the bigger speed drop at FFC = the landing foot
    back = "right" if front == "left" else "left"
    margin = abs(sf["left"] - sf["right"]) / max(abs(sf["left"]) + abs(sf["right"]), 1e-6)
    bfc_ok = sb["left"] is not None and sb["right"] is not None and sb[back] >= sb[front]
    conf = "High" if margin >= IDC.role_margin and bfc_ok else ("Medium" if margin >= IDC.role_margin or bfc_ok else "Low")
    why = (f"The {front} foot is the one that lands at front foot contact (frame {ffc})"
           + (f" and the {back} foot the one that lands at back foot contact (frame {bfc})." if bfc_ok else
              "; at back foot contact the feet were ambiguous."))
    if front != default_front:
        why += f" (The pose model's labels are the opposite of the usual for a {bowling_arm}-arm bowler, so the roles " \
               "follow the feet, not the labels.)"
    return {"front_leg": front, "back_leg": back, "bowling_arm": bowling_arm, "confidence": conf, "reason": why}

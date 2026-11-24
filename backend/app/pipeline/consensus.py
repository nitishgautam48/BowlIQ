"""Key moments from several independent methods each, combined by consensus.

Release  (a) highest bowling-wrist point   (b) peak hand speed + a short offset   (c) the arm passing vertical over
         the shoulder (shoulder→wrist closest to straight up, hand above the shoulder)
FFC      (a) front-ankle speed collapse (events.find_contacts)   (b) front-ankle height reaching its low point and
         staying there   (c) onset of the mid-hip deceleration (side-on only)
BFC      (a) back-ankle speed collapse   (b) back-ankle height reaching its low point and staying there

For each event the methods that agree within consensus.agree_s form a cluster; the estimate is the cluster median
(or the primary method's frame when it is in the cluster and consensus.primary_when_agreeing is set).
Confidence = w_agree × agreement + w_vis × joint visibility + w_fps × frame-rate score. Fewer than min_agree methods
agreeing → the best single estimate at Low confidence and needs_review with the reason.

The search is anchored on the strict detector in events.py (validated release peak with a front-foot landing before
it). If that finds nothing and strict=False, the anchor is the bowling hand's sharpest high peak and the result is
always flagged for review — the job never fails here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.errors import BowlIQError
from app.pipeline import events as E
from app.pipeline.params import P, frames
from app.pipeline.types import DeliveryEvents, EventResult, Kinematics

CS, ST, R, C = P.consensus, P.stride, P.release, P.contacts
MIN_N = P.general.min_samples


# ───────────────────────────── signals ─────────────────────────────

def _speed(p: np.ndarray, fps: float) -> np.ndarray:
    d = np.diff(p, axis=0) * fps
    d = np.vstack([d, d[-1:]]) if len(d) else np.zeros_like(p)
    return np.hypot(d[:, 0], d[:, 1])


def _clip(a: int, b: int, n: int) -> tuple[int, int]:
    return max(0, a), min(n - 1, b)


def _argmax_in(x: np.ndarray, a: int, b: int) -> Optional[int]:
    seg = x[a:b + 1]
    if b < a or not np.isfinite(seg).any():
        return None
    return a + int(np.nanargmax(seg))


def noise_floor(y: np.ndarray) -> float:
    """This clip's own frame-to-frame jitter of a signal (median absolute step)."""
    d = np.abs(np.diff(y))
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else 0.0


def plateau_start(y: np.ndarray, a: int, b: int, fps: float, bh: float, last: bool = False) -> Optional[int]:
    """First frame in [a, b] where the height y (Y up) has reached its low point and stays within a band of it for
    plateau_hold_s (band = the larger of plateau_band_bh and 3 × this clip's jitter). last=True: the start of the
    plateau run closest to b (the most recent landing)."""
    n = len(y)
    a, b = _clip(a, b, n)
    hold = frames(C.plateau_hold_s, fps)
    seg = y[a:min(n, b + hold + 1)]
    if b <= a or np.isfinite(seg).sum() < MIN_N:
        return None
    lo = float(np.nanmin(seg))
    band = max(C.plateau_band_bh * bh, 3.0 * noise_floor(y))
    on = np.isfinite(y) & (y <= lo + band)
    starts = [f for f in range(a, b + 1)
              if on[f] and (f == 0 or not on[f - 1] or f == a) and on[f:min(n, f + hold + 1)].all()]
    if not starts:
        return None
    return starts[-1] if last else starts[0]


def hip_decel_onset(kin: Kinematics, a: int, b: int) -> Optional[int]:
    """First frame in [a, b] where the forward hip speed has fallen below hip_decel_frac of its run-up peak."""
    hx = kin.mid("left_hip", "right_hip")[:, 0]
    v = np.gradient(hx) * kin.fps if len(hx) > 1 else hx * 0
    k = frames(P.context.bowler.speed_smooth_s, kin.fps)
    if k > 1:
        vv = np.where(np.isfinite(v), v, 0.0)
        cnt = np.convolve(np.isfinite(v).astype(float), np.ones(k), mode="same")
        v = np.where(cnt > 0, np.convolve(vv, np.ones(k), mode="same") / np.maximum(cnt, 1), np.nan)
    a, b = _clip(a, b, len(v))
    pk_a = max(0, a - frames(ST.runup_window_s, kin.fps))
    pre = v[pk_a:b + 1]
    if not np.isfinite(pre).any():
        return None
    peak = float(np.nanmax(pre))
    if peak <= 0:
        return None
    pk = pk_a + int(np.nanargmax(pre))
    for f in range(max(a, pk), b + 1):
        if np.isfinite(v[f]) and v[f] < ST.hip_decel_frac * peak:
            return f
    return None


def arm_vertical(kin: Kinematics, a: int, b: int) -> Optional[int]:
    """Frame in [a, b] where shoulder→wrist of the bowling arm is closest to vertical with the hand above the
    shoulder (the elbow passing over the shoulder line)."""
    sh = kin.joint(kin.side("shoulder", "bowling"))
    wr = kin.joint(kin.side("wrist", "bowling"))
    d = wr - sh
    a, b = _clip(a, b, len(d))
    best, bv = None, np.inf
    for f in range(a, b + 1):
        if np.all(np.isfinite(d[f])) and d[f, 1] > 0:
            ang = abs(np.degrees(np.arctan2(d[f, 0], d[f, 1])))
            if ang < bv:
                best, bv = f, ang
    return best


def foot_methods(kin: Kinematics, ankle_name: str, a: int, b: int, fps: float, body_h, bh: np.ndarray, mode: str,
                 first: bool) -> dict[str, Optional[int]]:
    """Heel/toe contact methods (engines with foot keypoints, e.g. RTMPose Halpe-26): the foot's lowest point (heel or
    toe) reaching and staying at its low point, and the heel's speed collapse (heel strike). Empty if the foot
    keypoints aren't seen in most of the window."""
    side = ankle_name.split("_")[0]
    heel, toe = kin.joint(f"{side}_heel"), kin.joint(f"{side}_foot_index")
    a, b = _clip(a, b, kin.n)
    if b <= a:
        return {}
    seen = np.isfinite(heel[a:b + 1, 1]).mean() if b >= a else 0.0
    if seen < P.contacts.foot_points_min_frac:
        return {}
    low = np.fmin(heel[:, 1], toe[:, 1])  # Y up: the lowest point of the foot
    out: dict[str, Optional[int]] = {
        "foot_height": plateau_start(low, a, b, fps, float(np.nanmedian(bh)), last=not first)}
    cs = E.find_contacts(heel, fps, body_h, mode, a, b)
    out["heel_speed"] = (cs[0][0] if first else cs[-1][0]) if cs else None
    return out


# ───────────────────────────── consensus ─────────────────────────────

# Methods that measure the same thing on neighbouring points are one kind of evidence: heel speed and ankle speed
# both time the foot's speed collapse, the foot's lowest point and the ankle's height both time its settling. Agreement
# counts independent KINDS, so two correlated methods can't confirm each other's mistake.
FAMILY = {"heel_speed": "speed", "foot_height": "height"}


def families(names) -> set[str]:
    return {FAMILY.get(k, k) for k in names}


def agree(guesses: dict[str, Optional[int]], fps: float, primary: str) -> tuple[Optional[int], list[str]]:
    """Each independent kind of evidence (FAMILY) casts ONE vote — the median of its methods (or its first method if
    they themselves disagree) — and the largest cluster of votes within agree_s wins. → (frame, agreeing method
    names). Ties go to the cluster holding the primary method."""
    tol = max(1, frames(CS.agree_s, fps))
    items = [(k, v) for k, v in guesses.items() if v is not None]
    if not items:
        return None, []
    fam: dict[str, list[tuple[str, int]]] = {}
    for k, v in items:
        fam.setdefault(FAMILY.get(k, k), []).append((k, v))
    votes: dict[str, float] = {}
    for f, mem in fam.items():
        vals = [v for _, v in mem]
        votes[f] = float(np.median(vals)) if max(vals) - min(vals) <= tol else float(mem[0][1])
    pf = FAMILY.get(primary, primary)
    best: list[str] = []
    for f0 in votes:
        cl = [f for f in votes if abs(votes[f] - votes[f0]) <= tol]
        if all(abs(votes[a] - votes[b]) <= tol for a in cl for b in cl) and \
                (len(cl) > len(best) or (len(cl) == len(best) and pf in cl and pf not in best)):
            best = cl
    names = [k for f in best for k, _ in fam[f]]
    if guesses.get(primary) is not None and best and pf not in best:
        # the definitional method (speed collapse / highest point) disagrees with the others: keep it as the best
        # estimate, reported as a disagreement (Low confidence, needs review)
        return int(guesses[primary]), [primary]
    if len(best) < CS.min_agree:
        v = guesses.get(primary)
        if v is None:
            return items[0][1], [items[0][0]]
        return v, [primary]
    if CS.estimate == "family_mean":
        return int(round(float(np.mean([votes[f] for f in best])))), names
    if pf in best and CS.primary_when_agreeing:
        # the other kinds have small systematic offsets (height plateau ≈ first heel touch, hip lags the foot): when
        # they agree with the primary method they confirm it rather than move it
        return int(guesses[primary]), names
    return int(round(float(np.median([votes[f] for f in best])))), names


def score(n_agree: int, n_methods: int, vis: float, fps: float) -> tuple[float, str]:
    agreement = n_agree / max(1, n_methods) if n_agree >= CS.min_agree else 0.0
    s = CS.w_agree * agreement + CS.w_vis * float(np.clip(vis, 0, 1)) + CS.w_fps * min(1.0, fps / CS.fps_full)
    lv = "High" if s >= CS.high else ("Medium" if s >= CS.medium else "Low")
    if n_agree < CS.min_agree:
        lv = "Low"
    return s, lv


def _vis(kin: Kinematics, name: str, f: int) -> float:
    return float(np.nanmean(kin.joint_vis(name)[max(0, f - 1):f + 2]))


def _event(name: str, guesses: dict, primary: str, fps: float, kin: Kinematics, joint: str, sharp: float,
           review: list[str], default: int) -> EventResult:
    f, agreeing = agree(guesses, fps, primary)
    if f is None:  # no method found it: typical timing, flagged
        f = int(np.clip(default, 0, kin.n - 1))
        review.append(f"{name}: no detection method found it; placed at typical timing — check it.")
        return EventResult(name, f, "Low", "Not detected: typical timing used; check it.", 0.0,
                           methods={k: (None if v is None else int(v)) for k, v in guesses.items()}, agree=0, score=0.0)
    avail = [k for k, v in guesses.items() if v is not None]
    n_agree, n_kinds = len(families(agreeing)), len(families(guesses))
    s, lv = score(n_agree, n_kinds, _vis(kin, joint, f), fps)
    ms = ", ".join(f"{k} {v if v is not None else '–'}" for k, v in guesses.items())
    if n_agree >= CS.min_agree:
        why = f"{n_agree} of {n_kinds} independent kinds of evidence agree ({ms}); {fps:.0f} fps."
    else:
        why = f"Methods disagree ({ms}): best single estimate ({primary}); check it."
        review.append(f"{name}: detection methods disagree ({ms}).")
    if len(families(avail)) < CS.min_agree and n_agree < CS.min_agree:
        why = f"Only {len(avail)} method found it ({ms}); check it."
    return EventResult(name, int(f), lv, why, sharp, methods={k: (None if v is None else int(v)) for k, v in guesses.items()},
                       agree=n_agree, score=round(s, 3))


def _front_back(kin: Kinematics, rel: int, mode: str) -> tuple[str, str, bool]:
    """Front foot = the one further towards the batter (+x) at release (side-on); else the anatomical labels."""
    a, b = kin.side("ankle", "front"), kin.side("ankle", "back")
    if mode == "side":
        xa, xb = kin.joint(a)[rel, 0], kin.joint(b)[rel, 0]
        if np.isfinite(xa) and np.isfinite(xb) and xb > xa:
            return b, a, True
    return a, b, False


def anchor_release(kin: Kinematics, bh: np.ndarray) -> int:
    wy = kin.joint(kin.side("wrist", "bowling"))[:, 1]
    cands = E.release_candidates(kin, wy, bh)
    if cands:
        return cands[0][0]
    if np.isfinite(wy).any():
        return int(np.nanargmax(wy))
    return kin.n // 2


def runup_speed(kin: Kinematics, ffc: int, bh: float) -> float:
    """Forward hip speed (body heights/s, along the run) over stride.runup_window_s before a front foot contact."""
    hx = kin.mid("left_hip", "right_hip")[:, 0]
    a = max(0, ffc - frames(ST.runup_window_s, kin.fps))
    seg = np.arange(a, ffc + 1)
    seg = seg[np.isfinite(hx[seg])]
    if len(seg) < max(4, MIN_N) or bh <= 0:
        return float("nan")
    return float(np.polyfit(seg / kin.fps, hx[seg], 1)[0]) / bh


def _fixed(name: str, f: int) -> EventResult:
    return EventResult(name, int(f), "High", "Set by hand on the video.", 1.0, methods={"user": int(f)}, agree=1, score=1.0)


def _build(kin: Kinematics, body_h, bh: np.ndarray, mode: str, anchor: Optional[DeliveryEvents], fixed: dict,
           fixed_rel_hint: Optional[int], review: list[str]) -> tuple[EventResult, EventResult, EventResult]:
    """BFC, FFC and release by consensus around one anchor delivery (or the best hand peak when there is none)."""
    fps, n = kin.fps, kin.n
    rel0 = (fixed["release"] if "release" in fixed else anchor.release.frame if anchor else
            fixed_rel_hint if fixed_rel_hint is not None else anchor_release(kin, bh))
    wname = kin.side("wrist", "bowling")
    wrist = kin.joint(wname)
    wy = wrist[:, 1]
    # ── release
    half = frames(R.search_half_s, fps)
    a, b = _clip(rel0 - half, rel0 + half, n)
    k = frames(R.peak_half_s, fps)
    sp = _speed(wrist, fps)
    g_rel = {
        "height": _argmax_in(wy, max(a, rel0 - k), min(b, rel0 + k)),
        "speed": (lambda f: None if f is None else min(n - 1, f + frames(R.speed_offset_s, fps, 0)))(_argmax_in(sp, a, b)),
        "arm_vertical": arm_vertical(kin, max(a, rel0 - k), min(b, rel0 + k)),
    }
    if anchor:
        g_rel["height"] = anchor.release.frame if g_rel["height"] is None else g_rel["height"]
    rel_sharp = anchor.release.sharpness if anchor else 0.0
    rel = _fixed("release", fixed["release"]) if "release" in fixed else \
        _event("release", g_rel, "height", fps, kin, wname, rel_sharp, review, rel0)
    # ── front foot contact
    front, back, swapped = (_front_back(kin, rel.frame, mode) if not anchor else
                            ((kin.side("ankle", "back"), kin.side("ankle", "front"), True) if anchor.legs_swapped
                             else (kin.side("ankle", "front"), kin.side("ankle", "back"), False)))
    fa, fb = _clip(rel.frame - frames(ST.ffc_release_max_s, fps), rel.frame - frames(ST.ffc_release_min_s, fps), n)
    fpos = kin.joint(front)
    if anchor and fa <= anchor.ffc.frame <= fb:
        speed_ffc = anchor.ffc.frame
    else:
        cs = E.find_contacts(fpos, fps, body_h, mode, fa, fb)
        speed_ffc = cs[0][0] if cs else None
    g_ffc = {"speed": speed_ffc, "height": plateau_start(fpos[:, 1], fa, fb, fps, float(np.nanmedian(bh)))}
    g_ffc.update(foot_methods(kin, front, fa, fb, fps, body_h, bh, mode, first=True))
    if mode == "side":
        g_ffc["hip_decel"] = hip_decel_onset(kin, fa, fb)
    ffc_sharp = anchor.ffc.sharpness if anchor else 0.0
    ffc = _fixed("FFC", fixed["ffc"]) if "ffc" in fixed else _event("FFC", g_ffc, "speed", fps, kin, front, ffc_sharp, review,
                                                                rel.frame - frames(P.priors.ffc_before_release_s, fps))
    if ffc.frame >= rel.frame and "ffc" not in fixed:  # nothing found: typical timing before release
        ffc.frame = max(0, rel.frame - frames(P.priors.ffc_before_release_s, fps))
        ffc.confidence = "Low"
        review.append("FFC: no front-foot landing found before release; estimated from typical timing.")
    # ── back foot contact
    ba, bb = _clip(ffc.frame - frames(ST.bfc_ffc_max_s, fps), ffc.frame - frames(ST.bfc_ffc_min_s, fps), n)
    bpos = kin.joint(back)
    if anchor and ba <= anchor.bfc.frame <= bb:
        speed_bfc = anchor.bfc.frame
    else:
        cs = E.find_contacts(bpos, fps, body_h, mode, ba, bb)
        speed_bfc = cs[-1][0] if cs else None
    g_bfc = {"speed": speed_bfc, "height": plateau_start(bpos[:, 1], ba, bb, fps, float(np.nanmedian(bh)), last=True)}
    g_bfc.update(foot_methods(kin, back, ba, bb, fps, body_h, bh, mode, first=False))
    bfc_sharp = anchor.bfc.sharpness if anchor else 0.0
    bfc = _fixed("BFC", fixed["bfc"]) if "bfc" in fixed else _event("BFC", g_bfc, "speed", fps, kin, back, bfc_sharp, review,
                                                                ffc.frame - frames(P.priors.bfc_before_ffc_s, fps))
    if bfc.frame >= ffc.frame and "bfc" not in fixed:
        bfc.frame = max(0, ffc.frame - frames(P.priors.bfc_before_ffc_s, fps))
        bfc.confidence = "Low"
        review.append("BFC: no back-foot landing found before front foot contact; estimated from typical timing.")
    if not anchor and not fixed:
        for e in (bfc, ffc, rel):
            e.confidence = "Low"
    if not fixed and not (0 <= bfc.frame < ffc.frame < rel.frame):  # clip edges squeezed them: spread them out
        rel.frame = int(np.clip(rel.frame, 2, n - 1))
        ffc.frame = int(np.clip(ffc.frame, 1, rel.frame - 1))
        bfc.frame = int(np.clip(bfc.frame, 0, ffc.frame - 1))
        review.append("The key moments were squeezed against the clip edge and spread out; check them.")
    if not (0 <= bfc.frame < ffc.frame < rel.frame):
        raise BowlIQError("INVALID_EVENTS", "The key moments are out of order (back foot, then front foot, then release).",
                          "Set them again in order.", 400)
    rel.legs_swapped = swapped  # carried to DeliveryEvents by the caller
    return bfc, ffc, rel


def detect_delivery(kin: Kinematics, body_h, mode: str = "side", strict: bool = True,
                    fixed: Optional[dict] = None) -> DeliveryEvents:
    """Consensus key moments. strict=True raises EVENTS_NOT_FOUND when the strict anchor search finds no delivery
    (so the caller can try another person / arm / zoom first); strict=False always returns a result.
    fixed: moments set by hand {"bfc"|"ffc"|"release": frame}; the others are searched relative to them."""
    fps, n = kin.fps, kin.n
    bh = np.broadcast_to(np.asarray(body_h, float), (n,))
    fixed = {k: int(v) for k, v in (fixed or {}).items() if v is not None}
    review: list[str] = []
    anchor: Optional[DeliveryEvents] = None
    if not fixed:
        try:
            anchor = E.detect_events(kin, body_h, mode=mode)
        except BowlIQError as exc:
            if strict:
                raise
            review.append(f"The strict search found no clear delivery ({exc.message}) — the moments below are estimates.")
    if "release" not in fixed and ("ffc" in fixed or "bfc" in fixed):
        # search release after the given foot contact
        base = fixed.get("ffc", fixed.get("bfc"))
        wy0 = kin.joint(kin.side("wrist", "bowling"))[:, 1]
        a0, b0 = _clip(base + frames(ST.ffc_release_min_s, fps), base + frames(ST.ffc_release_max_s + R.search_half_s, fps), n)
        fixed_rel_hint = _argmax_in(wy0, a0, b0)
    else:
        fixed_rel_hint = None
    candidates = [anchor]
    if anchor is not None and anchor.method == "release_first" and mode == "side":
        try:  # every delivery that passes the checks: best release peak, method agreement as tie-breaker
            candidates = list(E.release_first_all(kin, body_h))[:CS.max_candidates] or [anchor]
        except BowlIQError:
            candidates = [anchor]
    built = []
    for cand in candidates:
        rv: list[str] = list(review)
        b, f, r = _build(kin, body_h, bh, mode, cand, fixed, fixed_rel_hint, rv)
        built.append((cand.release.sharpness if cand else 0.0, runup_speed(kin, f.frame, float(np.nanmedian(bh))),
                      cand, b, f, r, rv))
    # the release peak's quality is the main evidence; between near-tied peaks the real delivery is the one at the end
    # of the fast run-up (highest hip speed before its front foot contact), not a later peak in the follow-through
    top = max(x[0] for x in built)
    tied = [x for x in built if x[0] >= top - CS.tie_margin]
    best = max(tied, key=lambda x: (x[1] if np.isfinite(x[1]) else -np.inf, x[0]))
    _, _, anchor, bfc, ffc, rel, review = best
    bfc, ffc, rel = bfc, ffc, rel
    method = "manual" if fixed else (anchor.method if anchor else "estimated")
    return DeliveryEvents(bfc, ffc, rel, method=method, legs_swapped=bool(getattr(rel, "legs_swapped", False)),
                          needs_review=bool(review), review_reasons=review)

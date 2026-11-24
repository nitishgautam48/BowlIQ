"""Phase 1: always analyse the right person.

Every detection carries its keypoints and an appearance signature (colour histograms of the torso and the legs).
The bowler is chosen from (in order of authority):
  1. the user's tap on the bowler in one frame (`BowlerSelection.tap`),
  2. otherwise every person track in the clip, ranked by the run-up pattern of a bowler (sustained travel across or
     towards/away from the camera, a sharp deceleration into the delivery stride, a big fast arm swing above the
     head) — never by size. The pipeline tries the next candidate when no delivery is found for the first.
People inside user-drawn ignore zones are dropped. The tracker follows the chosen person frame to frame using
predicted motion, body size and appearance, so it doesn't jump to someone nearby who looks different.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.core.errors import BowlIQError
from app.pipeline.params import P, frames
from app.pipeline.types import N_JOINTS, BowlerSelection, J, PoseSequence

T = P.tracking
APPEARANCE_MAX = T.appearance_max  # appearance distance above which a detection is "someone else"
_TORSO = [J["left_shoulder"], J["right_shoulder"], J["left_hip"], J["right_hip"]]


@dataclass
class Det:
    kp: np.ndarray  # (N_JOINTS, 3): x_px, y_px, visibility
    z: Optional[np.ndarray] = None  # (N_JOINTS,) relative depth, smaller = nearer the camera
    desc: Optional[np.ndarray] = None  # appearance signature
    tid: Optional[int] = None  # ByteTrack identity (RTMDet engine); None = engine without a tracker


@dataclass
class RawDetections:
    dets: list[list[Det]]
    fps: float
    width: int
    height: int
    device: str


# ───────────────────────────── appearance ─────────────────────────────

_PARTS = (("left_shoulder", "right_shoulder", "right_hip", "left_hip"), ("left_hip", "right_hip", "right_knee", "left_knee"))
_BINS = (12, 4)


def appearance(hsv: np.ndarray, kp: np.ndarray) -> Optional[np.ndarray]:
    """Hue/saturation histograms of the torso and upper-leg regions (each normalised to sum 1), concatenated."""
    out = []
    h, w = hsv.shape[:2]
    for part in _PARTS:
        pts = kp[[J[n] for n in part], :2]
        if not np.all(np.isfinite(pts)) or np.any(kp[[J[n] for n in part], 2] < 0.3):
            out.append(None)
            continue
        x0, y0 = np.clip(np.floor(pts.min(0)).astype(int), 0, [w - 1, h - 1])
        x1, y1 = np.clip(np.ceil(pts.max(0)).astype(int) + 1, 1, [w, h])
        if x1 - x0 < 3 or y1 - y0 < 3:
            out.append(None)
            continue
        roi = hsv[y0:y1, x0:x1]
        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, (pts - [x0, y0]).astype(np.int32), 255)
        hist = cv2.calcHist([roi], [0, 1], mask, list(_BINS), [0, 180, 0, 256]).ravel()
        tot = hist.sum()
        out.append(hist / tot if tot > 0 else None)
    if all(o is None for o in out):
        return None
    n = _BINS[0] * _BINS[1]
    return np.concatenate([o if o is not None else np.full(n, np.nan) for o in out])


def appearance_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Mean Bhattacharyya distance over the body parts both signatures have (None if not comparable)."""
    if a is None or b is None:
        return None
    n = _BINS[0] * _BINS[1]
    ds = []
    for k in range(len(_PARTS)):
        pa, pb = a[k * n:(k + 1) * n], b[k * n:(k + 1) * n]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        ds.append(1.0 - float(np.sum(np.sqrt(pa * pb))))
    return float(np.mean(ds)) if ds else None


# ───────────────────────────── geometry helpers ─────────────────────────────

def centroid(p: np.ndarray) -> np.ndarray:
    pts = p[_TORSO, :2]
    wts = np.clip(np.nan_to_num(p[_TORSO, 2]), 0.05, 1.0)
    ok = np.all(np.isfinite(pts), axis=1)
    if not ok.any():
        return np.array([np.nan, np.nan])
    return (pts[ok] * wts[ok, None]).sum(0) / wts[ok].sum()


def body_size(p: np.ndarray) -> float:
    sh = 0.5 * (p[J["left_shoulder"], :2] + p[J["right_shoulder"], :2])
    hp = 0.5 * (p[J["left_hip"], :2] + p[J["right_hip"], :2])
    an = 0.5 * (p[J["left_ankle"], :2] + p[J["right_ankle"], :2])
    v = np.linalg.norm(sh - hp) + np.linalg.norm(hp - an)
    return float(max(v, 1.0)) if np.isfinite(v) else 1.0


def bbox(p: np.ndarray, pad: float = 0.15) -> tuple[float, float, float, float]:
    xy = p[np.isfinite(p[:, 0]), :2]
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def _in_box(pt: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def apply_ignore_zones(dets: list[list[Det]], zones: list[tuple[float, float, float, float]]) -> list[list[Det]]:
    if not zones:
        return dets
    return [[d for d in frame if not any(_in_box(centroid(d.kp), z) for z in zones)] for frame in dets]


# ───────────────────────────── candidate seeds ─────────────────────────────

@dataclass
class Seed:
    frame: int
    c: np.ndarray
    size: float
    vel: np.ndarray
    desc: Optional[np.ndarray]
    how: str


def _build_tracks(dets: list[list[Det]], frame_range: range, fps: float) -> list[dict]:
    """Greedy tracks using position, size and appearance."""
    tracks: list[dict] = []
    link_gap = frames(T.link_gap_s, fps, P.general.min_samples)
    for f in frame_range:
        used: set[int] = set()
        for d in dets[f]:
            c, s = centroid(d.kp), body_size(d.kp)
            if not np.all(np.isfinite(c)):
                continue
            best, bd = None, np.inf
            for ti, t in enumerate(tracks):
                if ti in used or f - t["last_f"] > link_gap:
                    continue
                dist = np.linalg.norm(c - t["c"][-1])
                app = appearance_distance(d.desc, t["desc"])
                if dist >= T.link_dist_bodies * s or (app is not None and app > APPEARANCE_MAX):
                    continue
                cost = dist / s + (app or 0.0)
                if cost < bd:
                    best, bd = ti, cost
            if best is None:
                tracks.append({"f": [f], "c": [c], "s": [s], "kp": [d.kp], "last_f": f, "desc": d.desc})
                used.add(len(tracks) - 1)
            else:
                t = tracks[best]
                t["f"].append(f)
                t["c"].append(c)
                t["s"].append(s)
                t["kp"].append(d.kp)
                t["last_f"] = f
                if t["desc"] is None:
                    t["desc"] = d.desc
                used.add(best)
    return tracks


def _seed_from_track(t: dict, how: str, fps: float = 30.0) -> Seed:
    vel = np.zeros(2)
    if len(t["f"]) >= P.general.min_samples:
        k = min(frames(T.seed_velocity_s, fps), len(t["f"]) - 1)
        vel = (t["c"][k] - t["c"][0]) / max(1, t["f"][k] - t["f"][0])
    return Seed(t["f"][0], t["c"][0], float(np.median(t["s"])), vel, t["desc"], how)


def _track_bh(t: dict) -> float:
    """The track's body height (BH, pixels): nose-to-ankle × 1.07, median over its upright frames; else from the
    posture-robust body size (shoulder→hip→ankle ≈ 0.8 BH)."""
    kp = np.array(t["kp"])
    d = np.linalg.norm(kp[:, J["nose"], :2] - 0.5 * (kp[:, J["left_ankle"], :2] + kp[:, J["right_ankle"], :2]), axis=1)
    d = d[np.isfinite(d)]
    if len(d) >= P.general.min_samples:
        top = d[d >= np.percentile(d, P.scale.upright_pct)]
        return float(np.median(top) * P.scale.head_top_correction)
    return float(np.median(t["s"]) / P.context.bowler.body_size_bh)


def score_track(t: dict, fps: float) -> dict:
    """How much a person track looks like a bowler's run-up and delivery, from three scale- and fps-free cues (each
    0–1): sustained travel, a sharp deceleration, a big fast arm swing above the head. Size plays no part."""
    B = P.context.bowler
    out = {"travel": 0.0, "decel": 0.0, "arm": 0.0, "total": -1.0, "frames": len(t["f"])}
    if len(t["f"]) < max(P.general.min_samples, frames(B.min_visible_s, fps)):
        return out
    bh = max(_track_bh(t), 1.0)
    f = np.array(t["f"], float)
    c = np.array(t["c"])
    s = np.array(t["s"], float)
    # sustained travel: across the frame and towards/away from the camera (growth of apparent size)
    net_x = c[-1, 0] - c[0, 0]
    steps = np.diff(c[:, 0])
    consist_x = float(np.mean(np.sign(steps) == np.sign(net_x))) if len(steps) and net_x != 0 else 0.0
    q = max(1, len(s) // 4)
    depth = abs(np.log(max(np.median(s[-q:]), 1e-6) / max(np.median(s[:q]), 1e-6))) * P.context.view.camera_distance_bl
    lat = abs(net_x) / bh * consist_x
    out["travel"] = float(min(1.0, np.hypot(lat, depth) / B.travel_bh))
    # deceleration: forward speed (BH/s) after its peak falls to a fraction of the peak (the delivery stride)
    if len(f) >= 2 * P.general.min_samples:
        t_s = f / fps
        v = np.hypot(*np.gradient(c, t_s, axis=0).T) / bh
        k = max(1, frames(B.speed_smooth_s, fps) // max(1, int(np.median(np.diff(f)))))
        v = np.convolve(v, np.ones(k) / k, mode="same")
        pk = int(np.argmax(v[: max(1, int(B.peak_search_frac * len(v)))]))
        after = v[pk:]
        if v[pk] > 0 and len(after) > 1:
            lost = 1.0 - float(after.min()) / float(v[pk])
            out["decel"] = float(np.clip(lost / (1.0 - B.decel_frac), 0.0, 1.0)) * min(1.0, out["travel"] * 2)
    # arm swing: a hand clearly above the head, moving fast
    kp = np.array(t["kp"])
    head = kp[:, J["nose"], 1]
    above, speed = 0.0, 0.0
    for w in ("left_wrist", "right_wrist"):
        wy = kp[:, J[w], 1]
        a = (head - wy) / bh
        if np.isfinite(a).sum() >= P.general.min_samples:
            above = max(above, float(np.nanpercentile(a, 95)))
        wxy = kp[:, J[w], :2]
        dt = np.diff(f) / fps
        sp = np.linalg.norm(np.diff(wxy, axis=0), axis=1) / np.where(dt > 0, dt, np.nan) / bh
        if np.isfinite(sp).sum() >= P.general.min_samples:
            speed = max(speed, float(np.nanpercentile(sp, 95)))
    out["arm"] = 0.5 * float(np.clip(above / B.arm_above_head_bh, 0, 1)) + 0.5 * float(np.clip(speed / B.hand_speed_bh_s, 0, 1))
    out["total"] = out["travel"] + out["decel"] + out["arm"]
    return out


def _id_tracks(dets: list[list[Det]], fps: float) -> list[dict]:
    """Tracks from ByteTrack identities; an identity that ends and another that starts close by within
    tracking.link_gap_s (and looks alike) are joined, so a blurred frame or a crossing doesn't split the bowler."""
    by: dict[int, dict] = {}
    for f, frame in enumerate(dets):
        for d in frame:
            c = centroid(d.kp)
            if d.tid is None or not np.all(np.isfinite(c)):
                continue
            t = by.setdefault(d.tid, {"f": [], "c": [], "s": [], "kp": [], "desc": None, "tids": []})
            t["f"].append(f)
            t["c"].append(c)
            t["s"].append(body_size(d.kp))
            t["kp"].append(d.kp)
            t["tids"].append(d.tid)
            if t["desc"] is None:
                t["desc"] = d.desc
    tracks = sorted(by.values(), key=lambda t: t["f"][0])
    gap = frames(P.perception.relink_gap_s, fps)
    merged: list[dict] = []
    for t in tracks:
        best = None
        for m in merged:
            dt = t["f"][0] - m["f"][-1]
            if not 0 < dt <= gap:
                continue
            dist = np.linalg.norm(t["c"][0] - m["c"][-1]) / max(np.median(m["s"]), 1.0)
            app = appearance_distance(t["desc"], m["desc"])
            if dist <= T.follow_gate_bodies * (1 + 0.25 * dt) and (app is None or app <= APPEARANCE_MAX) \
                    and (best is None or dist < best[0]):
                best = (dist, m)
        if best is not None:
            m = best[1]
            for k in ("f", "c", "s", "kp", "tids"):
                m[k].extend(t[k])
        else:
            merged.append(t)
    for m in merged:
        m["last_f"] = m["f"][-1]
    return merged


def ranked_tracks(dets: list[list[Det]], fps: float) -> list[tuple[dict, dict]]:
    """Every person track in the whole clip with its bowler score, best first (deterministic: the rank is the
    candidate id the user can pick). With ByteTrack identities the tracks are those identities."""
    has_ids = any(d.tid is not None for frame in dets for d in frame)
    tracks = _id_tracks(dets, fps) if has_ids else _build_tracks(dets, range(len(dets)), fps)
    scored = [(t, score_track(t, fps)) for t in tracks]
    scored = [x for x in scored if x[1]["total"] >= 0]
    return sorted(scored, key=lambda x: (-x[1]["total"], x[0]["f"][0]))


def rank_seeds(dets: list[list[Det]], fps: float, selection: Optional[BowlerSelection] = None,
               limit: int | None = None) -> list[Seed]:
    """Candidate starting points for the bowler, most likely first. A tap gives exactly one seed."""
    n = len(dets)
    limit = limit or P.context.bowler.max_candidates
    if selection and selection.tap:
        tf, tx, ty = selection.tap
        pt = np.array([tx, ty])
        win = frames(T.tap_search_s, fps, P.general.min_samples)
        for df in sorted(range(-win, win + 1), key=abs):
            f = tf + df
            if not (0 <= f < n) or not dets[f]:
                continue
            inside = [d for d in dets[f] if _in_box(pt, bbox(d.kp))]
            pool = inside or [d for d in dets[f] if np.linalg.norm(centroid(d.kp) - pt) < 1.0 * body_size(d.kp)]
            if pool:
                d = min(pool, key=lambda d: np.linalg.norm(centroid(d.kp) - pt))
                return [Seed(f, centroid(d.kp), body_size(d.kp), np.zeros(2), d.desc, "tap")]
        raise BowlIQError("BOWLER_NOT_FOUND_AT_TAP", "No person was found where you tapped.",
                          "Tap the bowler on a frame where the whole body is visible, then analyse again.")
    if max((len(f) for f in dets), default=0) == 0:
        return []
    return [_seed_from_track(t, "auto", fps) for t, _ in ranked_tracks(dets, fps)[:limit]]


def candidates(raw: "RawDetections", selection: Optional[BowlerSelection] = None, limit: int | None = None) -> list[dict]:
    """The people the bowler can be chosen from (for 'Choose bowler'): id, score breakdown, a representative frame
    and the person's box there (fractions of the frame)."""
    limit = limit or P.context.bowler.max_candidates
    dets = apply_ignore_zones(raw.dets, selection.ignore if selection else [])
    out = []
    for i, (t, sc) in enumerate(ranked_tracks(dets, raw.fps)[:limit]):
        mid = len(t["f"]) // 2
        x0, y0, x1, y1 = bbox(t["kp"][mid])
        out.append({"id": i, "frame": int(t["f"][mid]), "first_frame": int(t["f"][0]), "last_frame": int(t["f"][-1]),
                    "box": [round(float(np.clip(x0 / raw.width, 0, 1)), 4), round(float(np.clip(y0 / raw.height, 0, 1)), 4),
                            round(float(np.clip(x1 / raw.width, 0, 1)), 4), round(float(np.clip(y1 / raw.height, 0, 1)), 4)],
                    "score": round(sc["total"], 3), "travel": round(sc["travel"], 3), "decel": round(sc["decel"], 3),
                    "arm": round(sc["arm"], 3)})
    return out


def track_from_seed(dets: list[list[Det]], fps: float, seed: Seed) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Follow the seeded person forwards and backwards. Returns kp (F, N_JOINTS, 3) and z (F, N_JOINTS) or None."""
    n = len(dets)
    out = np.full((n, N_JOINTS, 3), np.nan)
    zout = np.full((n, N_JOINTS), np.nan)
    has_z = False

    def run(frames: range, vel0: np.ndarray) -> None:
        nonlocal has_z
        last_c, last_s, gap, vel, desc = seed.c.copy(), seed.size, 0, vel0.copy(), seed.desc
        for f in frames:
            pred = last_c + vel * (gap + 1)
            gate = T.follow_gate_bodies * last_s * (1 + 0.25 * gap)
            best, bd = None, np.inf
            for d in dets[f]:
                c = centroid(d.kp)
                if not np.all(np.isfinite(c)):
                    continue
                ratio = body_size(d.kp) / max(last_s, 1e-6)
                if not (T.size_ratio[0] <= ratio <= T.size_ratio[1]):  # a different-sized person (e.g. further away)
                    continue
                app = appearance_distance(d.desc, desc)
                if app is not None and app > APPEARANCE_MAX:  # looks like someone else
                    continue
                dist = np.linalg.norm(c - pred)
                cost = dist / last_s + (app or 0.0)
                if dist < gate and cost < bd:
                    best, bd = d, cost
            if best is None:
                gap += 1
                continue
            c = centroid(best.kp)
            vel = 0.5 * vel + 0.5 * (c - last_c) / (gap + 1)
            out[f] = best.kp
            if best.z is not None:
                zout[f] = best.z
                has_z = True
            if best.desc is not None:
                desc = best.desc if desc is None else 0.9 * desc + 0.1 * best.desc
            last_c, last_s, gap = c, 0.8 * last_s + 0.2 * body_size(best.kp), 0

    run(range(seed.frame, n), seed.vel)
    run(range(seed.frame - 1, -1, -1), -seed.vel)
    return out, (zout if has_z else None)


# ───────────────────────────── finalisation ─────────────────────────────

def active_span(det: np.ndarray, fps: float) -> tuple[int, int] | None:
    """Longest stretch of frames with the bowler detected, bridging gaps of up to 0.25 s."""
    idx = np.where(det)[0]
    if len(idx) == 0:
        return None
    max_gap = frames(T.span_bridge_s, fps, P.general.min_samples)
    best, start, prev = (idx[0], idx[0]), idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap:
            start = i
        prev = i
        if prev - start > best[1] - best[0]:
            best = (start, prev)
    return int(best[0]), int(best[1])


def select(raw: RawDetections, selection: Optional[BowlerSelection] = None, rank: int = 0) -> PoseSequence:
    """Choose the bowler (candidate number `rank`), keep only the stretch where the bowler is on screen and check it's usable."""
    if not raw.dets:
        raise BowlIQError("UNREADABLE_VIDEO", "No frames could be read from the clip.", "Export the clip as MP4 (H.264) and upload again.")
    dets = apply_ignore_zones(raw.dets, selection.ignore if selection else [])
    seeds = rank_seeds(dets, raw.fps, selection)
    if rank >= len(seeds):
        raise BowlIQError("POSE_NOT_FOUND", "No (other) person in the clip moves like a bowler.",
                          "Tap the bowler in the 'Select the delivery' step and analyse again.")
    seed = seeds[rank]
    tids = None
    if any(d.tid is not None for frame in dets for d in frame):
        kp, z, tids = _follow_ids(dets, raw.fps, seed)
    else:
        kp, z = track_from_seed(dets, raw.fps, seed)
    det = np.isfinite(kp[:, 0, 0])
    span = active_span(det, raw.fps)
    rate = float(det[span[0]:span[1] + 1].mean()) if span else 0.0
    dur = (span[1] - span[0] + 1) / raw.fps if span else 0.0
    if span is None or dur < T.min_on_screen_s or rate < T.min_detection_rate:
        seen = f"{det.mean() * 100:.0f}% of frames" if span is None or dur >= T.min_on_screen_s else f"only {dur:.1f} s"
        raise BowlIQError(
            "POSE_NOT_FOUND",
            f"The bowler's skeleton was found in {seen}.",
            "Make sure the whole body is in view, well lit and not too small (fill at least a third of the frame height). "
            "Film side-on from about 10 m, on a tripod, and include the last run-up strides. If other people are in "
            "view, tap the bowler in the 'Select the delivery' step.",
        )
    # a joint pinned to the picture's edge is the model guessing a body part that is out of frame: not visible
    edge = T.edge_frac * min(raw.width, raw.height)
    at_edge = (kp[..., 0] <= edge) | (kp[..., 0] >= raw.width - 1 - edge) | (kp[..., 1] <= edge) | \
              (kp[..., 1] >= raw.height - 1 - edge)
    kp[at_edge & np.isfinite(kp[..., 0]), 2] = 0.0
    kp[: span[0]] = np.nan  # outside the bowler's window: other people or noise
    kp[span[1] + 1:] = np.nan
    if z is not None:
        z[: span[0]] = np.nan
        z[span[1] + 1:] = np.nan
    seq = PoseSequence(fps=raw.fps, width=raw.width, height=raw.height, kp=kp, device=raw.device,
                       people_detected_max=max(len(f) for f in raw.dets), detection_rate=rate, active_span=span,
                       z=z, raw=raw, selected_by=seed.how if rank == 0 else "auto-retry")
    if tids is not None:
        tids[: span[0]] = [None] * span[0]
        tids[span[1] + 1:] = [None] * (len(tids) - span[1] - 1)
        seq.tids = tids
    return seq


def _follow_ids(dets: list[list[Det]], fps: float, seed: Seed) -> tuple[np.ndarray, None, list]:
    """The seeded person's keypoints through their (re-linked) ByteTrack identities."""
    n = len(dets)
    start = None
    for d in dets[seed.frame]:
        if np.allclose(centroid(d.kp), seed.c, equal_nan=True):
            start = d.tid
            break
    out = np.full((n, N_JOINTS, 3), np.nan)
    tids: list = [None] * n
    for t in _id_tracks(dets, fps):
        if start in t["tids"]:
            for f, kp, tid in zip(t["f"], t["kp"], t["tids"], strict=True):
                out[f], tids[f] = kp, tid
            break
    else:  # seed from a tap on someone the tracker never confirmed: fall back to the frame-to-frame follower
        kp, _ = track_from_seed(dets, fps, seed)
        return kp, None, tids
    return out, None, tids


def candidate_count(raw: RawDetections, selection: Optional[BowlerSelection] = None) -> int:
    if selection and selection.tap:
        return 1
    return len(rank_seeds(apply_ignore_zones(raw.dets, selection.ignore if selection else []), raw.fps, None))

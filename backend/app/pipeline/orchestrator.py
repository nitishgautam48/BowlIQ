"""Runs every stage for one job and assembles the AnalysisResult.

Two halves:
  prepare()  probe → preprocess → pose (with the rotation check) → scene cuts → camera path. The result (keypoints of
             every person, audio, metadata) is cached in the analysis folder (cache.pkl).
  analyse()  clip context → key moments → second clip → quality → metrics → ball speed → report → video/PDF →
             diagnostics. Reads only the cache, so /recompute with user corrections takes seconds.

Graceful degradation: only an unreadable video, no person at all, or a clip under 1 s fail the job. Every other stage
returns its result with a confidence and warnings; if a stage breaks, what doesn't depend on it is still produced, the
rest is not_measured with the reason, and result.needs_review is set.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import logging
import time
import pickle  # nosec B403 (only our own cache files, written by this process into the server's storage folder)
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from scipy.ndimage import median_filter

from app.analysis.baseline import compute_baseline
from app.analysis.rules import judge
from app.core.config import settings
from app.core.errors import BowlIQError
from app.core.storage import AnalysisFiles
from app.models.schemas import AnalysisResult
from app.pipeline import clip_context as CX
from app.pipeline import gemini_video as GV
from app.pipeline import quality as Q
from app.pipeline.ball_speed import BallSpeedResult, measure_ball_speed
from app.pipeline.consensus import detect_delivery
from app.pipeline.metrics import FRONT_KEYS, LEVELS, SIDE_KEYS, Measurement, front_metrics
from app.pipeline.metrics import _not as _not_measured, ground_tilt, oblique_reason, side_metrics
from app.pipeline.params import P, frames
from app.pipeline.pose import PoseEstimator, get_estimator
from app.pipeline.preprocess import preprocess, target_size
from app.pipeline.scale import HEAD_TOP_CORRECTION, body_height_series, estimate_scale
from app.pipeline.smooth import legs_consistent, smooth_pose, swap_legs
from app.pipeline.sync import (AudioTrack, SyncResult, decide, extract_audio, motion_matcher, motion_search,
                               motion_signals)
from app.pipeline.three_d import counter_rotation, reconstruct, separation_at
from app.pipeline.tracking import candidate_count, candidates as bowler_candidates, select
from app.pipeline.types import ALL_JOINTS, BowlerSelection, DeliveryEvents, EventResult, J, Kinematics, QualityReport
from app.report.generator import build_payload, generate_report
from app.report.pdf import build_pdf
from app.report.render_video import SLOWMO_FACTOR, OverlayData, encode_seekable, frame_jpeg, render_videos

log = logging.getLogger("bowliq.pipeline")
StageFn = Callable[[str, float], None]

FATAL = {"UNREADABLE_VIDEO", "POSE_NOT_FOUND", "CLIP_TOO_SHORT", "CLIP_TOO_LONG", "INVALID_TRIM", "NO_SIDE_VIDEO",
         "POSE_INIT_FAILED", "MODEL_INVALID", "MODEL_DOWNLOAD_FAILED", "BOWLER_NOT_FOUND_AT_TAP", "DISK_FULL",
         "INVALID_EVENTS"}
NOT_SIDE = ("front-on", "behind")  # analysed with the front/behind measurements
SECOND_VIEWS = ("front-on", "behind")  # what the second clip should be
LEGACY_VIEW = {"oblique": "diagonal"}  # quality/metrics/report call an oblique view "diagonal"
MAX_SYNC_DISAGREEMENT_S = 0.1  # sound-synced vs front-clip release: beyond this the front-clip events are wrong
MAX_SIDE_ON_TILT = 35.0  # degrees of footprint-line tilt: the footprints lie along the line of sight (not side-on)
SLOW_SENSITIVE = {"runup_speed", "front_arm_pulldown_ms"}  # need real time; ratios and angles don't
THREE_D_KEYS = ("hip_shoulder_separation_bfc", "shoulder_counter_rotation")
CACHE_VERSION = 1


def _legacy(view: str) -> str:
    return LEGACY_VIEW.get(view, view)


# ───────────────────────────── small helpers ─────────────────────────────

def _event_dict(e: EventResult, fps: float) -> dict:
    return {"name": e.name, "frame": e.frame, "time_s": round(e.frame / fps, 4), "confidence": e.confidence,
            "confidence_reason": e.reason, "methods": dict(e.methods or {}), "agree": int(e.agree or 0)}


def _quality_dict(q: QualityReport) -> dict:
    d = asdict(q)
    d["camera_motion"] = None if q.camera_motion is None else round(q.camera_motion, 4)
    return d


def _keypoints_json(kin: Kinematics, video_fps: Optional[float] = None) -> dict:
    """video_fps: the frame rate of the served video. With slow motion the kinematics run at real time (fps × factor),
    but the skeleton must be drawn on the video's own frames, so the playback rate is what the browser needs."""
    w, h = kin.width, kin.height
    out_frames = []
    for f in range(kin.n):
        row = []
        for j in range(len(ALL_JOINTS)):
            if kin.valid[f, j]:
                x, y = kin.img_xy[f, j]
                row.append([round(float(x) / w, 5), round(float(y) / h, 5), round(max(float(kin.vis[f, j]), 0.5), 3)])
            else:
                row.append([0.0, 0.0, 0.0])
        out_frames.append(row)
    return {"fps": video_fps or kin.fps, "width": w, "height": h, "joints": ALL_JOINTS, "frames": out_frames}


def _front_body_height(kin: Kinematics) -> np.ndarray:
    d = body_height_series(kin) * HEAD_TOP_CORRECTION
    ok = np.isfinite(d)
    if ok.sum() < P.general.min_samples:
        return d
    idx = np.arange(len(d))
    d = np.interp(idx, idx[ok], d[ok])
    return median_filter(d, size=max(3, frames(0.1, kin.fps) | 1), mode="nearest")


def _release_arm_score(kin: Kinematics, ev, body_h: float) -> float:
    """How far the bowling wrist is above its own shoulder at release, in body heights (Y up)."""
    w = kin.joint(kin.side("wrist", "bowling"))[ev.release.frame]
    sh = kin.joint(kin.side("shoulder", "bowling"))[ev.release.frame]
    return float((w[1] - sh[1]) / body_h) if np.all(np.isfinite(w)) and np.all(np.isfinite(sh)) else -np.inf


def _selection(tap, ignore, trim_start: float, fps: float, w: int, h: int) -> Optional[BowlerSelection]:
    """User guidance (normalised 0–1 coordinates, tap time in seconds of the uploaded clip) → processed-video pixels."""
    sel = BowlerSelection()
    if tap:
        t, x, y = tap
        sel.tap = (max(0, int(round((t - trim_start) * fps))), x * w, y * h)
    for x0, y0, x1, y1 in ignore or []:
        sel.ignore.append((min(x0, x1) * w, min(y0, y1) * h, max(x0, x1) * w, max(y0, y1) * h))
    return sel if (sel.tap or sel.ignore) else None


def apply_3d(meas: list, kin: Kinematics, front_kin: Optional[Kinematics], events: DeliveryEvents,
             sync: Optional[SyncResult], f_view: Optional[str], side_q: str, front_q: str) -> list:
    """Phase 4: replace the flattened single-camera trunk rotations with true 3D ones from both synced clips."""
    if front_kin is None or sync is None or sync.offset_s is None or sync.method == "none":
        return meas
    if f_view not in ("front-on", "behind"):
        return meas
    rot = reconstruct(kin, front_kin, sync.offset_s, events)
    by = {m.key: m for m in meas}
    if rot.reason:
        for k in THREE_D_KEYS:
            m = by.get(k)
            if m is not None and m.measured:
                m.confidence_reason = f"{m.confidence_reason} 3D from both clips wasn't possible: {rot.reason}"
        return meas
    how = {"audio": "sound", "motion": "body movement", "release": "release moment"}[sync.method]
    levels = [sync.confidence, side_q, front_q, events.bfc.confidence]
    base = (f"True 3D rotation from both clips (side-on + {f_view}), synced on the {how}; differences of angles, so "
            "how square the second camera is cancels out.")

    def conf(extra_levels: list[str]) -> tuple[str, str]:
        lv = LEVELS[min(LEVELS.index(x) for x in levels + extra_levels)]
        weak = [f"sync {sync.confidence.lower()}" if sync.confidence != "High" else "",
                f"side-on clip quality {side_q.lower()}" if side_q != "High" else "",
                f"second clip quality {front_q.lower()}" if front_q != "High" else ""]
        weak = [w for w in weak if w]
        return lv, base + (f" Limited by: {', '.join(weak)}." if weak else "")

    out = []
    for m in meas:
        if m.key == "hip_shoulder_separation_bfc":
            v = separation_at(rot, events.bfc.frame)
            if v is not None:
                lv, why = conf([])
                m = Measurement(key=m.key, value=v, unit=m.unit, view="3d", frame=events.bfc.frame, event="BFC",
                                confidence=lv, confidence_reason=why, measured=True, joint=m.joint,
                                extra={"single_camera_value": m.value})
        elif m.key == "shoulder_counter_rotation":
            r = counter_rotation(rot, events.bfc.frame, events.ffc.frame)
            if r is not None:
                lv, why = conf([events.ffc.confidence])
                m = Measurement(key=m.key, value=r[0], unit=m.unit, view="3d", frame=r[1], event="FFC",
                                confidence=lv, confidence_reason=why, measured=True, joint=m.joint,
                                extra={"single_camera_value": m.value})
        out.append(m)
    return out


def map_events(ev: DeliveryEvents, side_fps: float, offset: Optional[float], f_fps: float, n: int, sync_conf: str,
               method: str) -> Optional[DeliveryEvents]:
    """Side-on key moments carried into the second clip through the sync (front_time = side_time + offset)."""
    if offset is None:
        return None
    out = []
    for e in (ev.bfc, ev.ffc, ev.release):
        f = int(round((e.frame / side_fps + offset) * f_fps))
        if not 0 <= f < n:
            return None
        conf = LEVELS[min(LEVELS.index(e.confidence), LEVELS.index(sync_conf))]
        out.append(EventResult(e.name, f, conf, f"{e.reason.rstrip('.')} (taken from the side-on clip via the {method} sync).",
                               e.sharpness))
    return DeliveryEvents(*out)


def events_agree(a: DeliveryEvents, b: DeliveryEvents, fps: float) -> bool:
    tol = max(2, frames(P.consensus.agree_s, fps))
    return all(abs(x.frame - y.frame) <= tol for x, y in ((a.bfc, b.bfc), (a.ffc, b.ffc), (a.release, b.release)))


def confirm_events(ev: DeliveryEvents, why: str) -> DeliveryEvents:
    """Two independent views agreeing on a key moment: one confidence level up."""
    def up(e: EventResult) -> EventResult:
        lv = LEVELS[min(2, LEVELS.index(e.confidence) + 1)]
        return EventResult(e.name, e.frame, lv, f"{e.reason}; {why}", e.sharpness, dict(e.methods or {}), e.agree, e.score)
    return DeliveryEvents(up(ev.bfc), up(ev.ffc), up(ev.release), ev.method, ev.legs_swapped, ev.needs_review,
                          list(ev.review_reasons))


def _correction_helps(kin: Kinematics, ev: DeliveryEvents) -> bool:
    """A planted front foot can't move (front foot contact → release). The camera correction is only kept if the
    foot is at least as still in corrected coordinates as in the raw picture."""
    a = J[kin.side("ankle", "front")]
    seg = slice(ev.ffc.frame, ev.release.frame + 1)

    def drift(xy: np.ndarray) -> float:
        p = xy[seg, a]
        p = p[np.all(np.isfinite(p), axis=1)]
        return float(np.max(np.linalg.norm(p - p[0], axis=1))) if len(p) >= 2 else np.nan
    w, r = drift(kin.xy), drift(kin.img_xy)
    nose = kin.img_xy[:, J["nose"]]
    ank = 0.5 * (kin.img_xy[:, J["left_ankle"]] + kin.img_xy[:, J["right_ankle"]])
    d = np.linalg.norm(nose - ank, axis=1)
    bh = float(np.nanmedian(d)) * HEAD_TOP_CORRECTION if np.isfinite(d).any() else 0.0
    return not (np.isfinite(w) and np.isfinite(r) and w > r * 1.05 + P.camera.planted_slack_bh * bh)


def bowler_px(pose) -> Optional[float]:
    from app.pipeline.adapt import _body_px
    if pose.active_span is None:
        return None
    h = _body_px(pose.kp[pose.active_span[0]:pose.active_span[1] + 1])
    h = h[np.isfinite(h)]
    return float(np.median(h)) if len(h) >= 5 else None


def _camera(video, pose, notes: list[str], prefix: str):
    """Phase 1c: estimate how the camera moved; return the path if it moved and could be followed."""
    from app.pipeline.adapt import estimate_camera_path, people_boxes

    if not settings.adapt_video:
        return None
    span = pose.active_span or (0, pose.kp.shape[0] - 1)
    raw_motion = Q.camera_motion(video, Q.person_boxes(pose))
    if raw_motion is None or raw_motion < P.camera.steady_wps:
        return None  # steady camera: chaining small fit errors would only add drift
    try:
        margin = frames(0.5, pose.fps)
        path = estimate_camera_path(video, people_boxes(pose.raw, pose.kp.shape[0]), (span[0] + span[1]) // 2,
                                    (span[0] - margin, span[1] + margin))
    except Exception:  # never let the optional correction break an analysis
        log.exception("camera path estimation failed")
        return None
    if not path.moved:
        return None
    if not path.ok:
        notes.append(f"{prefix}The camera moves and the background couldn't be followed in enough frames to correct "
                     "it, so speeds and distances are unreliable. Use a tripod.")
        return None
    notes.append(f"{prefix}Camera movement corrected (the view shifts up to {path.max_shift_px:.0f} px and turns up to "
                 f"{path.max_turn_deg:.1f}°); measurements use one fixed view.")
    return path


def hand_speed(pose) -> Optional[float]:
    """Highest 99th-percentile wrist speed of either arm, body heights per second (at the clip's frame rate)."""
    best = None
    for arm in ("left", "right"):
        kin = smooth_pose(pose, bowling_arm=arm)
        try:
            bh = estimate_scale(kin, None).body_height_px
        except BowlIQError:
            return None
        w = kin.joint(f"{arm}_wrist")
        sp = np.linalg.norm(np.diff(w, axis=0), axis=1) * kin.fps / max(bh, 1.0)
        if np.isfinite(sp).sum() >= 10:
            v = float(np.nanpercentile(sp, 99))
            best = v if best is None else max(best, v)
    return best


# ───────────────────────────── stage runner ─────────────────────────────

class Stages:
    """Runs each stage so that only FATAL errors end the job; anything else is logged, reported and degraded."""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self.failed: list[str] = []

    def run(self, name: str, fn: Callable[[], Any], fallback: Callable[[], Any] | Any = None) -> Any:
        t0 = time.perf_counter()
        try:
            out = fn()
            self.items.append({"name": name, "ok": True, "detail": None, "seconds": round(time.perf_counter() - t0, 2)})
            return out
        except BowlIQError as exc:
            if exc.code in FATAL:
                raise
            log.info("Stage %s degraded: %s %s", name, exc.code, exc.message)
            self.items.append({"name": name, "ok": False, "detail": exc.message})
            self.failed.append(f"{name}: {exc.message}")
        except Exception as exc:  # a bug in one stage must not lose the whole analysis
            log.exception("Stage %s crashed", name)
            self.items.append({"name": name, "ok": False, "detail": f"internal error ({type(exc).__name__})"})
            self.failed.append(f"{name}: an internal error stopped this step ({type(exc).__name__}).")
        return fallback() if callable(fallback) else fallback


# ───────────────────────────── prepare (cached) ─────────────────────────────

@dataclass
class Prepared:
    version: int
    meta: Any
    start: float
    end: float
    checks: list
    reasons: list
    level: str
    prep: Any
    pose: Any
    side_audio: Optional[AudioTrack] = None
    front_audio: Optional[AudioTrack] = None
    front_prep: Any = None
    front_pose: Any = None
    f_checks: list = field(default_factory=list)
    f_reasons: list = field(default_factory=list)
    f_level: str = "Low"
    front_fail: Optional[str] = None
    has_front_input: bool = False
    rotation: Optional[CX.Detection] = None
    adjustments: list = field(default_factory=list)
    swapped: bool = False
    used_fallback: bool = False
    timings: dict = field(default_factory=dict)  # seconds per preparation step (pose etc.)
    gemini: Any = None  # gemini_video.ClipBrief when the optional Gemini video check ran
    gemini_error: Optional[str] = None


def _pose_with_rotation(est, src, dst, meta, trim, sel, on_progress, adjustments: list[str], prefix: str = ""):
    """Pose of the bowler, verifying the picture's orientation: if the bowler is sideways/upside down (or nobody is
    found at all), the clip is re-processed rotated and the better orientation kept. Returns (prep, pose, rotation)."""
    start, end = trim
    ts = start if start > 0 else None
    prep = preprocess(src, dst, meta, ts, end)
    try:
        pose = est.estimate(dst, prep.fps, on_progress, selection=sel)
    except BowlIQError as exc:
        if exc.code != "POSE_NOT_FOUND":
            raise
        for fix in ("cw90", "ccw90"):
            try:
                prep2 = preprocess(src, dst, meta, ts, end, rotate=fix)
                pose2 = est.estimate(dst, prep2.fps, on_progress, selection=None)
            except BowlIQError:
                continue
            rot = CX.check_rotation(pose2.kp, meta.rotation)
            if rot.value == "none":
                adjustments.append(f"{prefix}Nobody was found in the picture as recorded; it was turned "
                                   f"{'clockwise' if fix == 'cw90' else 'anticlockwise'} 90° and the bowler found.")
                return prep2, pose2, CX.Detection(fix, "Medium", rot.reasons + ["Turned 90° (nobody found upright)."])
        preprocess(src, dst, meta, ts, end)  # leave the original orientation in place
        raise
    rot = CX.check_rotation(pose.kp, meta.rotation)
    if rot.value != "none":
        tmp = dst.with_name("rot_" + dst.name)
        try:
            prep2 = preprocess(src, tmp, meta, ts, end, rotate=rot.value)
            pose2 = est.estimate(tmp, prep2.fps, on_progress, selection=None)
            if CX.check_rotation(pose2.kp, 0).value == "none":
                tmp.replace(dst)
                adjustments.append(f"{prefix}The video was stored {'upside down' if rot.value == '180' else 'sideways'}"
                                   " (wrong or missing rotation tag), so it was turned upright before analysis.")
                return prep2, pose2, CX.Detection(rot.value, "High", rot.reasons + ["Turned upright: verified."])
        except BowlIQError:
            pass
        finally:
            tmp.unlink(missing_ok=True)
        rot.reasons.append("Turning the picture didn't make the bowler upright, so it was analysed as recorded.")
        rot.value, rot.confidence = "none", "Low"
    return prep, pose, rot


def prepare(files: AnalysisFiles, params: dict, on_stage: StageFn, est: PoseEstimator) -> Prepared:
    side_in = files.find_input("side")
    front_in = files.find_input("front")
    if side_in is None:
        raise BowlIQError("NO_SIDE_VIDEO", "The video is missing.", "Upload a clip and try again.", 400)
    trim_s, trim_e = params.get("trim_start_s"), params.get("trim_end_s")
    on_stage("Checking quality", 1)
    meta = Q.probe_video(side_in)
    start, end = Q.resolve_trim(meta.duration_s, trim_s, trim_e)
    checks, reasons, level = Q.metadata_checks(meta, end - start)
    side_audio = extract_audio(side_in, start, (end - start) if trim_e is not None else None) if front_in else None
    adjustments: list[str] = []
    tw, th = target_size(meta.width, meta.height, settings.max_long_side)
    side_sel = _selection(params.get("side_tap"), params.get("side_ignore"), start, meta.process_fps or meta.fps, tw, th)
    on_stage("Checking quality", 4)

    front_prep = front_pose = front_audio = front_meta = None
    f_checks: list = []
    f_reasons: list = []
    f_level = "Low"
    front_fail = None
    if front_in is not None:
        try:  # a bad second clip never blocks the main analysis
            front_meta = Q.probe_video(front_in)
            f_checks, f_reasons, f_level = Q.metadata_checks(front_meta, front_meta.duration_s)
            front_audio = extract_audio(front_in)
        except BowlIQError as exc:
            front_fail = f"Second clip: {exc.message}"
            front_meta = None
    on_stage("Checking quality", 8)

    span = (8, 55) if front_meta else (8, 70)
    t_pose = time.perf_counter()
    prep, pose, rot = _pose_with_rotation(
        est, side_in, files.side_prep, meta, (start, end if trim_e is not None else None), side_sel,
        lambda p: on_stage("Detecting skeleton", span[0] + p * (span[1] - span[0])), adjustments)
    if prep.duration_s < P.general.min_clip_s:
        raise BowlIQError("CLIP_TOO_SHORT", f"The clip is {prep.duration_s:.2f} s long; a delivery needs at least "
                          f"{P.general.min_clip_s:g} s.", "Choose a longer section with the run-up, stride and release.")
    used_fallback = bool(getattr(est, "used_fallback", False))
    pose_s = round(time.perf_counter() - t_pose, 1)
    side_in.unlink(missing_ok=True)  # side_prep.mp4 is the canonical copy from here on
    if front_meta is not None:
        try:
            ftw, fth = target_size(front_meta.width, front_meta.height, settings.max_long_side)
            front_sel = _selection(params.get("front_tap"), params.get("front_ignore"), 0.0,
                                   front_meta.process_fps or front_meta.fps, ftw, fth)
            front_prep, front_pose, _ = _pose_with_rotation(
                est, front_in, files.front_prep, front_meta, (0.0, None), front_sel,
                lambda p: on_stage("Detecting skeleton", 55 + p * 15), adjustments, "Second clip: ")
        except BowlIQError as exc:
            front_fail = f"Second clip: {exc.message}"
            front_pose = None
            if files.front_prep.exists():
                try:
                    front_prep = Q.probe_video(files.front_prep)
                except BowlIQError:
                    front_prep = None
    if front_in is not None:
        front_in.unlink(missing_ok=True)

    cuts = Q.scene_cuts(files.side_prep)
    if cuts:
        a, b = Q.keep_one_shot(pose, cuts)
        adjustments.append(f"The clip contains {len(cuts)} cut{'s' if len(cuts) > 1 else ''} (a new shot or a replay "
                           f"starting again), so only the shot with the bowler ({a / prep.fps:.1f}–{b / prep.fps:.1f} s) "
                           "was analysed.")
    for pz, pre in ((pose, ""), (front_pose, "Second clip: ")):
        if pz is not None and pz.tiled:
            adjustments.append(f"{pre}The bowler was too small to find in the full picture, so the picture was searched "
                               "again in enlarged sections.")
        if pz is not None and pz.zoomed:
            adjustments.append(f"{pre}The bowler is small in the picture (about {pz.zoomed[0]} px tall), so the skeleton "
                               f"was found again on a {pz.zoomed[1]:.1f}× zoomed-in view for more precise joints.")
    pose.cam = _camera(files.side_prep, pose, adjustments, "")
    if front_pose is not None:
        front_pose.cam = _camera(files.front_prep, front_pose, adjustments, "Second clip: ")

    swapped = False
    if front_pose is not None:
        v_main = CX.classify_view(Q.world_kp(pose))
        v_sec = CX.classify_view(Q.world_kp(front_pose))
        if v_main.value in NOT_SIDE and v_main.confidence != "Low" and v_sec.value in ("side-on", "oblique"):
            # the clips were uploaded the wrong way round: use the second one as the main (side-on) view
            tmp = files.dir / "swap_prep.mp4"
            files.side_prep.replace(tmp)
            files.front_prep.replace(files.side_prep)
            tmp.replace(files.front_prep)
            pose, front_pose = front_pose, pose
            prep, front_prep = front_prep, prep
            side_audio, front_audio = front_audio, side_audio
            checks, f_checks = f_checks, checks
            reasons, f_reasons = f_reasons, reasons
            level, f_level = f_level, level
            start, end = 0.0, prep.duration_s
            swapped = True
    on_stage("Finding key moments", 66)
    st = Prepared(CACHE_VERSION, meta, start, end, checks, reasons, level, prep, pose, side_audio, front_audio,
                    front_prep, front_pose, f_checks, f_reasons, f_level, front_fail, front_in is not None, rot,
                    adjustments, swapped, used_fallback)
    st.timings = {"pose_s": pose_s, "frames": int(pose.kp.shape[0]), "engine": getattr(est, "name", ""),
                  "pose_ms_per_frame": round(1000 * pose_s / max(1, pose.kp.shape[0]), 1)}
    attach_gemini(files, st)
    on_stage("Finding key moments", 70)
    return st


def attach_gemini(files: AnalysisFiles, st: Prepared) -> None:
    """Optional Gemini first pass on the processed main clip (cached with the keypoints, so /recompute reuses it)."""
    if not GV.enabled():
        return
    st.gemini, st.gemini_error = GV.brief(files.side_prep, st.prep.duration_s)


def save_cache(files: AnalysisFiles, st: Prepared) -> None:
    try:
        with open(files.cache, "wb") as fh:
            pickle.dump(st, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # recompute just becomes unavailable
        log.exception("Could not write the keypoint cache")


def load_cache(files: AnalysisFiles) -> Optional[Prepared]:
    if not files.cache.exists():
        return None
    try:
        with open(files.cache, "rb") as fh:
            st = pickle.load(fh)  # noqa: S301  # nosec B301 (our own file in the server's storage folder)
    except Exception:
        log.exception("Could not read the keypoint cache")
        return None
    return st if getattr(st, "version", None) == CACHE_VERSION else None


# ───────────────────────────── key moments ─────────────────────────────

def _arm_order(ctx_arm: CX.Detection) -> list[str]:
    if ctx_arm.source == "user":
        return [ctx_arm.value]
    other = "left" if ctx_arm.value == "right" else "right"
    return [ctx_arm.value, other]


def _resolve_arm_and_events(pose, height_cm, arms: list[str], direction: str, fixed: Optional[dict] = None):
    """Try the bowling-arm hypotheses (the context's choice first) and keep the one whose release is validated by a
    sharp peak and follow-through (or, failing that, the wrist highest above its own shoulder at release)."""
    best, first_error = None, None
    for arm in arms:
        kin = smooth_pose(pose, bowling_arm=arm, direction=direction)
        scale = estimate_scale(kin, height_cm)
        try:
            ev = detect_delivery(kin, scale.body_height_px, mode="side", strict=True, fixed=fixed)
        except BowlIQError as exc:
            if exc.code == "INVALID_EVENTS":
                raise
            first_error = first_error or exc
            continue
        if ev.legs_swapped:  # the pose model labelled the legs the wrong way round: make the labels match
            kin = swap_legs(kin)
        validated = ev.method == "release_first"
        score = (1, ev.release.sharpness) if validated else (0, _release_arm_score(kin, ev, scale.body_height_px))
        log.info("Arm hypothesis %s: release frame %d (%s), score %.2f", arm, ev.release.frame, ev.method, score[1])
        if best is None or score[0] > best[0][0] or (score[0] == best[0][0] and score[1] > best[0][1] + 0.02):
            best = (score, kin, scale, ev)
    if best is None:
        raise first_error
    return best[1], best[2], best[3]


def _choose_bowler(pose, height_cm, selection, video_path, arms, direction, fixed, fixed_person: bool):
    """Find the delivery for the selected person; without a tap or a chosen person, try the next most likely
    candidates when no bowling action is found, so a bystander can't be analysed by mistake."""
    first_error = None
    raw = pose.raw
    n = 1 if (raw is None or fixed_person) else min(P.context.bowler.max_candidates, max(1, candidate_count(raw, selection)))
    for rank in range(n):
        try:
            cand = pose if rank == 0 else select(raw, selection, rank)
        except BowlIQError:
            continue
        if rank > 0:  # a different person was picked: refine that one on a zoomed crop too
            from app.pipeline.adapt import zoom_refine
            cand.fps = pose.fps
            cand = zoom_refine(cand, video_path, cand.fps) if video_path else cand
        cand.cam, cand.fps = pose.cam, pose.fps
        try:
            kin, scale, events = _resolve_arm_and_events(cand, height_cm, arms, direction, fixed)
        except BowlIQError as exc:
            if exc.code != "EVENTS_NOT_FOUND":
                raise
            first_error = first_error or exc
            continue
        return cand, kin, scale, events, rank
    raise first_error or BowlIQError("EVENTS_NOT_FOUND", "No delivery was found.", "")


def _find_delivery(pose, height_cm, selection, video, arms, direction, fixed, fixed_person, adjustments):
    """Key moments: as filmed → on a zoomed-in view (rescue). Raises EVENTS_NOT_FOUND if neither finds a delivery."""
    try:
        return _choose_bowler(pose, height_cm, selection, video, arms, direction, fixed, fixed_person)
    except BowlIQError as exc:
        first = exc
    if first.code != "EVENTS_NOT_FOUND" or pose.zoomed or not settings.adapt_video:
        raise first
    from app.pipeline.adapt import zoom_refine
    zoomed = zoom_refine(pose, video, pose.fps, force=True)
    if not zoomed.zoomed:
        raise first
    zoomed.cam, zoomed.fps = pose.cam, pose.fps
    out = _choose_bowler(zoomed, height_cm, selection, video, arms, direction, fixed, fixed_person)
    adjustments.append(f"The delivery couldn't be found at first, so the skeleton was found again on a "
                       f"{out[0].zoomed[1]:.1f}× zoomed-in view of the bowler.")
    return out


def _release_arm(pose, view: str) -> Optional[CX.Detection]:
    """Front-on/behind main clip: the bowling arm is the wrist label with the best validated release peak (sharp, high,
    with a rise before and a follow-through drop after). Cross-checked with the picture: seen from behind, a right arm
    comes over on the image right; facing the camera, on the image left. Agreement = Medium, disagreement = Low."""
    from app.pipeline import events as E
    best = None
    for arm in ("right", "left"):
        kin = smooth_pose(pose, bowling_arm=arm, direction="left_to_right")
        try:
            bh = estimate_scale(kin, None).body_height_px
        except BowlIQError:
            return None
        c = E.release_candidates(kin, kin.joint(f"{arm}_wrist")[:, 1], np.full(kin.n, bh))
        if c and (best is None or c[0][1] > best[2]):
            best = (arm, c[0][0], c[0][1], kin)
    if best is None:
        return None
    arm, r, _score, kin = best
    im = kin.img_xy
    off = im[r, J[f"{arm}_wrist"], 0] - 0.5 * (im[r, J["left_shoulder"], 0] + im[r, J["right_shoulder"], 0])
    if not np.isfinite(off):
        return CX.Detection(arm, "Low", [f"The clearest release peak is the {arm} hand's (frame {r})."],
                            details={"release_frame": int(r)})
    shoulder_w = abs(im[r, J["left_shoulder"], 0] - im[r, J["right_shoulder"], 0])
    if not np.isfinite(shoulder_w) or abs(off) < P.context.arm.side_min_frac * max(shoulder_w, 1.0):
        return CX.Detection(arm, "Low", [f"The clearest release peak is the {arm} hand's (frame {r}), but the hand "
                                         "comes over almost straight above the head, so which side it is on can't "
                                         "confirm the arm."], details={"release_frame": int(r)})
    image_side = "right" if off > 0 else "left"
    expected = image_side if view == "behind" else ("left" if image_side == "right" else "right")
    agree = expected == arm
    why = (f"The clearest release peak is the {arm} hand's (frame {r}); seen {('from behind' if view == 'behind' else 'from the front')}, "
           f"that hand comes over on the image {image_side}, which {'fits' if agree else 'suggests the other arm — the pose model may have swapped left and right'}.")
    return CX.Detection(arm, "Medium" if agree else "Low", [why], details={"release_frame": int(r), "image_side": image_side})


def _second_view_events(pose, height_cm, arm: str, fixed: Optional[dict]):
    """The main clip is filmed from in front of or behind the bowler: key moments from the front-view signals."""
    kin = smooth_pose(pose, bowling_arm=arm, direction="left_to_right")
    scale = estimate_scale(kin, height_cm)
    ev = detect_delivery(kin, _front_body_height(kin), mode="front", strict=False, fixed=fixed)
    return kin, scale, ev


def _fixed_from(ov: dict) -> Optional[dict]:
    f = {k: ov.get(f"{k}_frame") for k in ("bfc", "ffc", "release")}
    f = {k: int(v) for k, v in f.items() if v is not None}
    return f or None


def _overrides(params: dict) -> dict:
    """User corrections: /recompute overrides plus the older upload/reanalyse fields."""
    ov = dict(params.get("overrides") or {})
    if params.get("camera_view") in ("side-on", "front-on", "behind") and "view" not in ov:
        ov["view"] = params["camera_view"]
    if params.get("bowling_arm") in ("right", "left") and "bowling_arm" not in ov:
        ov["bowling_arm"] = params["bowling_arm"]
    fe = params.get("forced_events")
    if fe:
        for k in ("bfc", "ffc", "release"):
            ov.setdefault(f"{k}_frame", fe.get(k))
    if ov.get("view") == "diagonal":
        ov["view"] = "oblique"
    return {k: v for k, v in ov.items() if v is not None}


# ───────────────────────────── analyse ─────────────────────────────

def analyse(files: AnalysisFiles, st: Prepared, params: dict, history: list[dict], on_stage: StageFn,
            render: bool = True) -> dict:
    S = Stages()
    ov = _overrides(params)
    height_cm = ov.get("height_cm", params.get("bowler_height_cm"))
    warnings: list[str] = []
    if st.swapped:
        warnings.append("The first clip looks filmed from in front of or behind the bowler and the second from the "
                        "side, so they were swapped: the second clip is used as the main view.")
    adjustments = list(st.adjustments)
    review: list[str] = []
    pose, prep, front_pose, front_prep = st.pose, st.prep, st.front_pose, st.front_prep
    checks, reasons, level = list(st.checks), list(st.reasons), st.level
    side_sel = _selection(params.get("side_tap"), params.get("side_ignore"), st.start, prep.fps, prep.width, prep.height)
    if (st.rotation is not None and st.rotation.value != "none") or st.swapped:
        side_sel = None  # taps were given on the picture as uploaded

    gem = getattr(st, "gemini", None)
    gem_out: dict[str, str] = {}

    # ── Gemini's bowler (a hint: checked against the run-up scoring, never used over a tap or a chosen person)
    if gem is not None and gem.bowler is not None and pose.raw is not None and pose.selected_by != "tap" \
            and ov.get("bowler_track_id") is None:
        def gemini_bowler():
            b, m = gem.bowler, P.gemini_video.bowler_box_margin
            f = int(round(b.time_s * prep.fps))
            cx, cy = 0.5 * (b.x0 + b.x1) * prep.width, 0.5 * (b.y0 + b.y1) * prep.height
            p2 = select(pose.raw, BowlerSelection(tap=(f, cx, cy), ignore=side_sel.ignore if side_sel else []))
            box = ((b.x0 - m) * prep.width, (b.y0 - m) * prep.height, (b.x1 + m) * prep.width, (b.y1 + m) * prep.height)
            return p2, box
        got_g = S.run("Gemini bowler", gemini_bowler, None)
        if got_g is not None:
            p2, _box = got_g
            both = np.isfinite(pose.kp[:, J["left_hip"], 0]) & np.isfinite(p2.kp[:, J["left_hip"], 0])
            same = both.sum() >= P.general.min_samples and float(np.median(np.linalg.norm(
                pose.kp[both, J["left_hip"], :2] - p2.kp[both, J["left_hip"], :2], axis=1))) < 0.25 * (bowler_px(pose) or 1e9)
            if same:
                gem_out["bowler"] = "agree"
            else:
                from app.pipeline.adapt import zoom_refine
                p2 = zoom_refine(p2, files.side_prep, p2.fps)
                p2.cam, p2.selected_by = pose.cam, "gemini"
                pose = p2
                gem_out["bowler"] = "gemini"

    # ── who is the bowler
    cands = S.run("candidates", lambda: bowler_candidates(pose.raw, side_sel) if pose.raw is not None else [], [])
    rank = 0
    fixed_person = pose.selected_by == "tap"
    if ov.get("bowler_track_id") is not None and pose.raw is not None:
        def pick():
            from app.pipeline.adapt import zoom_refine
            p2 = select(pose.raw, BowlerSelection(ignore=side_sel.ignore if side_sel else []), int(ov["bowler_track_id"]))
            p2 = zoom_refine(p2, files.side_prep, p2.fps)
            p2.cam = pose.cam
            p2.selected_by = "user"
            return p2
        picked = S.run("choose bowler", pick, None)
        if picked is not None:
            pose, rank, fixed_person = picked, int(ov["bowler_track_id"]), True

    # ── slow motion
    slow_param = params.get("slow_motion")
    slow_guess: Optional[float] = None
    if slow_param is not None and float(slow_param) > 1:
        pose.fps = prep.fps * float(slow_param)
    elif slow_param is None and not _fixed_from(ov) and gem is not None and gem.slow_motion == "slow_motion" \
            and gem.slow_motion_confidence != "low" and gem.slow_motion_factor and gem.slow_motion_factor > 1:
        slow_guess = float(min(P.slow_motion.choices, key=lambda k: abs(k - gem.slow_motion_factor)))
        pose.fps = prep.fps * slow_guess
        gem_out["slow_motion"] = "gemini"
        adjustments.append(f"Gemini sees a slow-motion replay (about ×{gem.slow_motion_factor:g}), so it was analysed "
                           f"as ×{slow_guess:g} slow motion. Choose the exact speed on upload for reliable speeds.")
    elif slow_param is None and not _fixed_from(ov):
        peak = S.run("slow-motion check", lambda: hand_speed(pose), None)
        gem_real = gem is not None and gem.slow_motion == "real_time" and gem.slow_motion_confidence != "low"
        if gem_real and peak is not None and peak < P.slow_motion.hand_speed_bh_s:
            gem_out["slow_motion"] = "gemini"  # Gemini sees real-time footage: don't slow the clip down on a guess
            peak = None
        if peak is not None and peak < P.slow_motion.hand_speed_bh_s:
            slow_guess = float(min(P.slow_motion.choices,
                                   key=lambda k: abs(k - P.slow_motion.real_hand_speed_bh_s / max(peak, 1e-3))))
            pose.fps = prep.fps * slow_guess
            adjustments.append(f"This looks like a slow-motion clip, so it was analysed as about ×{slow_guess:g} slow "
                               "motion. Choose the exact speed on upload for reliable timings and speeds; check the key "
                               "moments.")
    fps_eff = pose.fps

    # ── clip context (arm, direction, view), each with confidence and reasons; corrections override
    def context():
        wkp = Q.world_kp(pose)
        arm = CX.detect_arm(wkp, fps_eff)
        hint = arm.details.get("peak_frame")
        return arm, CX.detect_direction(wkp, hint), CX.classify_view(wkp, hint)
    arm_d, dir_d, view_d = S.run("clip context", context, lambda: (
        CX.Detection("right", "Low", ["Couldn't be detected; right arm assumed."]),
        CX.Detection("left_to_right", "Low", ["Couldn't be detected."]),
        CX.Detection("side-on", "Low", ["Couldn't be detected; side-on assumed."])))
    if gem is not None:
        view_d, gem_out["view"] = GV.combine(view_d, gem.view, gem.view_confidence, "camera view")
        gd = gem.run_direction if gem.run_direction in ("left_to_right", "right_to_left") else None
        dir_d, gem_out["direction"] = GV.combine(dir_d, gd, gem.run_direction_confidence, "run-up direction")
    if "bowling_arm" in ov:
        arm_d = CX.user(ov["bowling_arm"], arm_d)
    if "direction" in ov:
        dir_d = CX.user(ov["direction"], dir_d)
    if "view" in ov:
        view_d = CX.user(ov["view"], view_d)
    main_view = view_d.value
    if main_view in NOT_SIDE and arm_d.source == "auto":
        rel_arm = S.run("bowling arm (front/behind view)", lambda: _release_arm(pose, main_view), None)
        if rel_arm is not None:
            arm_d = CX.Detection(rel_arm.value, rel_arm.confidence, rel_arm.reasons, details={**arm_d.details, **rel_arm.details})
        else:
            arm_d.confidence = "Low"
            arm_d.reasons.append("No clear release peak was found for either hand.")
    if gem is not None:
        arm_d, gem_out["bowling_arm"] = GV.combine(arm_d, gem.bowling_arm, gem.bowling_arm_confidence, "bowling arm")
    fixed = _fixed_from(ov)
    if fixed:
        names = {"bfc": "back foot", "ffc": "front foot", "release": "release"}
        warnings.append("Key moments set by hand (" + ", ".join(f"{names[k]} frame {v}" for k, v in fixed.items())
                        + "); every measurement uses them.")
    if fixed and not all(0 <= v < pose.kp.shape[0] for v in fixed.values()):
        raise BowlIQError("INVALID_EVENTS", "A key moment set by hand is outside the clip.", "Set it again.", 400)

    # ── key moments
    estimated = False

    def fallback_events(p):
        k = smooth_pose(p, bowling_arm=arm_d.value, direction=dir_d.value)
        sc = estimate_scale(k, height_cm)
        return k, sc, detect_delivery(k, sc.body_height_px, mode="side", strict=False, fixed=fixed)

    def side_events():
        nonlocal pose, rank, estimated, main_view, view_d
        try:
            p2, k, sc, ev, r = _find_delivery(pose, height_cm, side_sel, files.side_prep, _arm_order(arm_d),
                                              dir_d.value, fixed, fixed_person, adjustments)
            if r > 0:
                p2.selected_by = "auto-retry"
            pose, rank = p2, rank + r
        except BowlIQError as exc:
            if exc.code == "INVALID_EVENTS":
                raise
            estimated = True
            k, sc, ev = fallback_events(pose)
        tilt = ground_tilt(k, ev)
        tilt_meaningful = pose.cam is not None or view_d.value != "side-on" or (view_d.details.get("travel_bl") or 0) > 0.6
        if (tilt is not None and tilt > MAX_SIDE_ON_TILT and tilt_meaningful and view_d.source != "user"
                and view_d.confidence != "High" and not fixed):
            # the two footprints lie along the camera's line of sight: filmed from behind/in front, not side-on
            sr = view_d.details.get("size_ratio") or 1.0
            main_view = "behind" if sr < 1 else "front-on"
            view_d = CX.Detection(main_view, "Medium", view_d.reasons + [
                f"The two footprints lie along the line of sight (ground line tilted {tilt:.0f}°), so the clip is "
                f"analysed as filmed {'from behind' if main_view == 'behind' else 'front-on'}."], details=view_d.details)
            return None
        return k, sc, ev

    kin = scale = events = None
    if main_view not in NOT_SIDE:
        got = S.run("key moments", side_events, None)
        if got is None and main_view not in NOT_SIDE:  # the stage broke: best estimate
            got = S.run("key moments (estimate)", lambda: fallback_events(pose), None)
            estimated = True
        if got is not None:
            kin, scale, events = got
    if main_view in NOT_SIDE:
        got = S.run("key moments", lambda: _second_view_events(pose, height_cm, arm_d.value, fixed), None)
        if got is not None:
            kin, scale, events = got
            warnings.append(f"This clip is filmed {'from behind the bowler' if main_view == 'behind' else 'from in front of the bowler'}"
                            ", so it was analysed as a front/behind view: side bend, arm angle, head position, front-foot "
                            "alignment and run-up line. Stride, knee and trunk-lean angles need a side-on clip.")
    if kin is None:
        raise BowlIQError("POSE_NOT_FOUND", "The bowler's body couldn't be measured in this clip.",
                          "Keep the whole body, head to feet, in view and try again.")
    second_main = main_view in NOT_SIDE
    estimated = estimated or events.method == "estimated"

    if not second_main and pose.cam is not None and not fixed and not estimated and not _correction_helps(kin, events):
        def without_correction():
            nonlocal pose
            cam = pose.cam
            pose.cam = None  # the planted foot is stiller without the correction: don't use it
            try:
                p2, k2, s2, e2, _ = _choose_bowler(pose, height_cm, side_sel, files.side_prep, _arm_order(arm_d),
                                                   dir_d.value, fixed, True)
            except BowlIQError:
                pose.cam = cam
                raise
            adjustments[:] = [a for a in adjustments if not a.startswith("Camera movement corrected")]
            adjustments.append("The camera moves, and correcting it didn't hold the planted foot still, so it wasn't "
                               "used: speeds and distances are less reliable. Use a tripod.")
            return p2, k2, s2, e2
        got = S.run("camera correction check", without_correction, None)
        if got is not None:
            pose, kin, scale, events = got

    if not second_main and not estimated and legs_consistent(kin, events.ffc.frame) is False:
        # the leg labels are mirrored for the whole clip (front foot behind the back foot at FFC): swap and retry
        swapped_kin = swap_legs(kin)
        if fixed:
            if legs_consistent(swapped_kin, events.ffc.frame) is not False:
                kin = swapped_kin
                warnings.append("Left and right legs were mislabelled by the pose model and have been corrected.")
        else:
            try:
                ev2 = detect_delivery(swapped_kin, scale.body_height_px, mode="side", strict=True)
                if legs_consistent(swapped_kin, ev2.ffc.frame) is not False:
                    kin, events = swapped_kin, ev2
                    warnings.append("Left and right legs were mislabelled by the pose model and have been corrected.")
            except BowlIQError:
                warnings.append("The front and back legs may be mislabelled; leg metrics have lower reliability.")

    # roles: front/back leg from the foot contacts (not the raw labels), bowling arm from the context
    from app.pipeline.identity import assign_roles
    roles = S.run("limb roles", lambda: assign_roles(kin, events, kin.bowling_arm, scale.body_height_px), None)
    if roles is not None:
        kin = dataclasses.replace(kin, roles=roles)
        if roles["confidence"] == "Low":
            review.append(f"Front/back leg: {roles['reason']}")

    # final arm as used (the release check can overrule the context's first choice)
    if arm_d.source == "auto" and kin.bowling_arm != arm_d.value:
        arm_d = CX.Detection(kin.bowling_arm, "Medium", arm_d.reasons + [
            f"The {kin.bowling_arm} arm gives the clearer release (high, sharp peak and follow-through), so it was used."],
            details=arm_d.details)
    elif arm_d.source == "auto" and arm_d.confidence == "Low" and events.method == "release_first":
        arm_d.confidence = "Medium"
        arm_d.reasons.append("Confirmed by a clear release peak with this arm.")
    if estimated and arm_d.source == "auto" and arm_d.confidence != "Low":
        arm_d.confidence = "Low"
        arm_d.reasons.append("No clear release was found to confirm the bowling arm.")
    bowler_d = CX.bowler_detection(pose, cands, rank)
    if pose.selected_by == "user":
        bowler_d = CX.user(rank, bowler_d)
    elif gem_out.get("bowler") == "agree":
        bowler_d.confidence = "High" if bowler_d.confidence != "Low" else "Medium"
        bowler_d.reasons.append("Gemini, watching the clip, points to the same person.")
    elif gem_out.get("bowler") == "gemini":
        bowler_d = CX.Detection(bowler_d.value, "Medium", ["Gemini, watching the clip, pointed to this person as the "
                                                           "bowler (the run-up scoring preferred someone else)."])
    if gem is not None and gem.release_time_s is not None and not fixed:
        dt = events.release.frame / prep.fps - gem.release_time_s
        if abs(dt) > P.gemini_video.release_tolerance_s:
            review.append(f"Release: measured at {events.release.frame / prep.fps:.2f} s but Gemini, watching the "
                          f"clip, places it near {gem.release_time_s:.2f} s — check the key moments.")
            gem_out["release"] = "conflict"
        else:
            gem_out["release"] = "agree"
    if gem is not None and not gem.is_bowling_delivery:
        review.append("Gemini doesn't see a bowling delivery in this clip — check that the right clip was uploaded.")
    ctx = CX.ClipContext(st.rotation or CX.Detection("none", "Low"), dir_d, arm_d, view_d, bowler_d, cands,
                         roles=kin.roles, identity_log=list(kin.identity_log), fusion=getattr(pose, "fusion", None),
                         engine=getattr(pose, "engine", "") or getattr(pose.raw, "device", ""))
    on_stage("Finding key moments", 74)
    if estimated:
        warnings.append("The key moments couldn't be detected reliably, so they were estimated from the bowling arm. "
                        "Check them with \"Check detection\" if they're off: every measurement depends on them.")
    if events.needs_review:
        review.extend(events.review_reasons)

    # ── second clip
    front = S.run("second clip", lambda: _second_clip(files, st, prep, kin, events, front_pose, front_prep, warnings),
                  lambda: _no_second(st))
    if front["events"] is not None:
        events = front["events"]
    if front["fail"] and front["fail"] not in warnings and front["q"] is None and st.has_front_input:
        warnings.append(front["fail"])

    # ── quality
    boxes = Q.person_boxes(pose)
    raw_motion = S.run("camera motion", lambda: Q.camera_motion(files.side_prep, boxes), None)
    motion = raw_motion
    if pose.cam is not None:
        from app.pipeline.adapt import residual_motion
        motion = S.run("camera motion (corrected)", lambda: residual_motion(files.side_prep, boxes, pose.cam), raw_motion)
    small = bowler_px(pose)
    if small is not None and small < P.quality.bowler_small_model_frac * P.pose_model.input_px:
        level = "Low"
        reasons.append(f"The bowler is only about {small:.0f} px tall, so joint positions and timing are approximate.")
        warnings.append(f"The bowler is small in the picture (about {small:.0f} px tall): results are approximate. Film "
                        "closer or zoom in so the bowler fills at least a third of the frame height.")
    warnings.extend(adjustments)
    qview = _legacy(main_view)
    quality = Q.finalize(prep, st.end - st.start, checks, reasons, level, motion, qview,
                         (qview,) if second_main else ("side-on",), pose.detection_rate)
    if pose.active_span and (pose.active_span[1] - pose.active_span[0] + 1) < 0.9 * pose.kp.shape[0]:
        a, b = pose.active_span
        warnings.append(f"The bowler is on screen from {a / prep.fps:.1f} s to {b / prep.fps:.1f} s; the rest of the "
                        "clip was ignored. Trimming to the delivery on upload makes analysis faster.")
    if st.used_fallback:
        warnings.append("The GPU pose engine couldn't find the bowler in this clip, so the CPU engine (MediaPipe) was used.")
    if pose.selected_by == "tap":
        warnings.append("You picked the bowler with a tap; other people in the video were ignored.")
    elif pose.selected_by == "user":
        warnings.append("You chose the bowler; other people in the video were ignored.")
    elif pose.selected_by == "gemini":
        warnings.append("The bowler was picked with Gemini's help (it watched the clip). Use \"Choose bowler\" if "
                        "it's the wrong person.")
    elif pose.selected_by == "auto-retry":
        warnings.append("The first person found didn't show a bowling action, so another person was analysed. Use "
                        "\"Choose bowler\" to be sure.")
    elif pose.people_detected_max > 1:
        warnings.append(f"{pose.people_detected_max} people were visible; the bowler was picked automatically from "
                        "their run-up and arm action. Use \"Choose bowler\" if it's the wrong person.")
    on_stage("Measuring", 76)

    # ── measurements
    front_q, front_kin, front_ev, f_view, sync = front["q"], front["kin"], front["ev"], front["view"], front["sync"]
    missing = "Add a front-on clip" if not st.has_front_input else (front["fail"] or "The second clip could not be analysed.")

    def measure():
        if second_main:
            need = f"Needs a side-on clip: this one is filmed {'from behind the bowler' if main_view == 'behind' else 'front-on'}."
            m = [_not_measured(k, "side-on", need) for k in SIDE_KEYS]
            m += front_metrics(kin, events, quality.level, " ".join(quality.reasons), "", view=main_view)
            return m, None
        tilt_ = ground_tilt(kin, events)
        m = side_metrics(kin, events, scale, quality.level, " ".join(quality.reasons), motion, tilt_)
        if oblique_reason(tilt_):
            warnings.append(oblique_reason(tilt_))
        m += front_metrics(front_kin if front_ev else None, front_ev, front_q.level if front_q else "Low",
                           " ".join(front_q.reasons) if front_q else "", missing, view=_legacy(f_view or "front-on"))
        m = apply_3d(m, kin, front_kin, events, sync, f_view, quality.level, front_q.level if front_q else "Low")
        return m, tilt_
    why_fail = "This measurement couldn't be computed (an internal step failed); the rest of the analysis is unaffected."
    meas, tilt = S.run("measurements", measure, lambda: (
        [_not_measured(k, "side-on", why_fail) for k in SIDE_KEYS] + [_not_measured(k, "front-on", why_fail) for k in FRONT_KEYS],
        None))
    # 3D lifting (MotionBERT) and the view-aware choice between 2D and 3D per angle
    from app.pipeline import lift3d
    angle_report: dict = {}
    pose3d: dict = {}
    if lift3d.available():
        def lifted():
            L = lift3d.lift(kin)
            if L is None:
                return meas, {}
            with contextlib.suppress(Exception):  # the viewer's copy is a help, never a reason to lose the 3D angles
                pose3d.update(lift3d.export(L, kin, events))
            return lift3d.apply_sources(meas, lift3d.angles(L, kin, events), events, main_view)
        meas, angle_report = S.run("3D lifting", lifted, lambda: (meas, {}))
    if estimated:  # every measurement hangs on the estimated key moments: say so and cap the confidence
        for mm in meas:
            if mm.measured:
                mm.confidence = "Low"
                mm.confidence_reason = "Key moments were estimated, not detected. " + mm.confidence_reason
    if slow_guess:
        why = (f"The slow-motion speed was estimated (about ×{slow_guess:g}), so real speeds and times can't be trusted. "
               "Choose the clip's slow-motion speed when uploading.")
        for mm in meas:
            if mm.key in SLOW_SENSITIVE and mm.measured:
                mm.value, mm.measured, mm.confidence, mm.reason, mm.confidence_reason = None, False, None, why, why
    on_stage("Measuring", 80)

    if second_main or slow_guess or estimated:
        ball = BallSpeedResult(None, None, "Ball speed needs a side-on clip, real-time (or a stated slow-motion speed) "
                               "and detected key moments.", 0, [])
    else:
        ball = S.run("ball speed", lambda: measure_ball_speed(
            str(files.side_prep), kin.fps, events.release.frame, kin.img_xy[:, ALL_JOINTS.index(f"{kin.bowling_arm}_wrist")],
            boxes, scale.metres_per_pixel, kin.direction, raw_motion, tilt, scale.rel_uncertainty),
            lambda: BallSpeedResult(None, None, "Ball speed couldn't be measured (an internal step failed).", 0, []))
    ball_d = {"value_kmh": ball.value_kmh, "plus_minus": ball.plus_minus,
              "status": "measured" if ball.value_kmh is not None else "not_measured", "reason": ball.reason,
              "frames_tracked": ball.frames_tracked}
    on_stage("Writing report", 84)

    # ── report
    baseline = compute_baseline(history)
    quality_d = _quality_dict(quality)

    def report(ms):
        judged_ = judge(ms, baseline)
        payload = build_payload(judged_.findings, judged_.strengths, judged_.injury_flags, quality_d, ball_d,
                                judged_.metrics, front_ev is not None)
        return judged_, generate_report(payload, prep.fps, front_ev is not None)
    judged, rep = S.run("report", lambda: report(meas), lambda: report(
        [_not_measured(m.key, m.view if m.view in ("side-on", "front-on") else "side-on", why_fail) for m in meas]))
    texts = {t["rank"]: t for t in rep.pop("findings")}
    findings = [{**f, "why": texts.get(f["rank"], {}).get("why", ""), "cue": texts.get(f["rank"], {}).get("cue", ""),
                 "drill": texts.get(f["rank"], {}).get("drill", "")} for f in judged.findings]
    on_stage("Writing report", 87)

    # ── outputs
    files.keypoints.write_text(json.dumps(_keypoints_json(kin, prep.fps), separators=(",", ":")))
    ov_data = OverlayData.from_kinematics(kin, {"BFC": events.bfc.frame, "FFC": events.ffc.frame,
                                                "release": events.release.frame})
    # the served videos (and the browser's time ↔ frame mapping) run at the clip's own frame rate, never at the
    # slow-motion-corrected real-time rate: otherwise every drawn skeleton belongs to another frame
    ov_data.fps = prep.fps
    if second_main:
        ov_data.view = "second"
    ov_data.save(files.overlay)
    s0, s1 = 0, max(0, kin.n - 1)
    if render:
        got = S.run("video", lambda: render_videos(files.side_prep, ov_data, files.annotated, files.clean,
                                                   lambda p: on_stage("Writing report", 87 + p * 9)), None)
        if got is not None:
            s0, s1 = got
    key_frames: dict[str, bytes] = {}

    def keyframes():
        for name, ev in (("bfc", events.bfc), ("ffc", events.ffc), ("release", events.release)):
            data = frame_jpeg(files.side_prep, ev.frame, ov_data)
            if data:
                files.key_frame(name).write_bytes(data)
                key_frames[name] = data
    S.run("key frames", keyframes)
    for p in files.dir.joinpath("frames").glob("f*_*.jpg"):  # cached scrubber frames show old overlays
        p.unlink(missing_ok=True)
    front_video = S.run("second video", lambda: _front_outputs(files, front_prep, front_kin, render), None)
    on_stage("Writing report", 97)

    fe = None
    if st.has_front_input:
        fe = {"bfc": _event_dict(front_ev.bfc, front_kin.fps) if front_ev else None,
              "ffc": _event_dict(front_ev.ffc, front_kin.fps) if front_ev else None,
              "release": _event_dict(front_ev.release, front_kin.fps) if front_ev else None,
              "reason": None if front_ev else missing}
    if estimated:
        review.append("The key moments were estimated, not detected.")
    for name in ctx.low():
        d = {"direction": ctx.direction, "bowling_arm": ctx.arm, "view": ctx.view, "bowler": ctx.bowler}[name]
        review.append(f"{name.replace('_', ' ').capitalize()} detected with low confidence: {' '.join(d.reasons)}")
    if slow_guess:
        review.append(f"Slow-motion speed estimated (×{slow_guess:g}).")
    review.extend(S.failed)
    result = {
        "id": files.id,
        "created_at": params.get("_created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bowler_name": params.get("bowler_name") or "Me",
        "bowler_height_cm": height_cm,
        "bowling_arm": kin.bowling_arm,
        "direction": dir_d.value if second_main else kin.direction,
        "has_front": st.has_front_input,
        "video": {"fps": prep.fps, "width": prep.width, "height": prep.height, "frame_count": kin.n,
                  "duration_s": round(kin.n / prep.fps, 4),
                  "slowmo": {"start_frame": s0, "end_frame": s1, "factor": SLOWMO_FACTOR}},
        "quality": quality_d,
        "front_quality": _quality_dict(front_q) if front_q else None,
        "events": {"bfc": _event_dict(events.bfc, prep.fps), "ffc": _event_dict(events.ffc, prep.fps),
                   "release": _event_dict(events.release, prep.fps)},
        "front_events": fe,
        "key_moments_estimated": estimated,
        "front_video": front_video,
        "sync": sync.to_dict() if sync else None,
        "scale": {"method": scale.method, "metres_per_pixel": scale.metres_per_pixel,
                  "rel_uncertainty": scale.rel_uncertainty,
                  "body_height_px": round(scale.body_height_px, 2), "frame": scale.frame},
        "metrics": judged.metrics,
        "ball_speed": ball_d,
        "headline_metric": judged.headline_metric,
        "findings": findings,
        "strengths": judged.strengths,
        "kinetic_chain": judged.kinetic_chain,
        "injury_flags": judged.injury_flags,
        "baseline": ({"active": False, "deliveries": 0, "deliveries_until_baseline": 0,
                      "based_on": "synthetic sample: samples never build a personal baseline"}
                     if params.get("bowler_name") == settings.sample_bowler else
                     {"active": baseline.active, "deliveries": baseline.deliveries,
                      "deliveries_until_baseline": baseline.deliveries_until_baseline, "based_on": baseline.based_on}),
        "report": rep,
        "feedback": {},
        "warnings": [w for w in dict.fromkeys(warnings) if w],
        "needs_review": bool(review),
        "review_reasons": list(dict.fromkeys(review)),
        "context": {**ctx.to_dict(), "angle_sources": angle_report, "timings": getattr(st, "timings", {})},
        "stages": S.items,
        "overrides": ov,
        "can_recompute": files.cache.exists(),
        "diagnostics": False,
        "recomputed_at": params.get("_recomputed_at"),
        "gemini": None if gem is None and not getattr(st, "gemini_error", None) else {
            "used": gem is not None, "error": getattr(st, "gemini_error", None),
            "brief": gem.model_dump() if gem is not None else None, "outcomes": gem_out},
    }
    if gem is not None:
        for g_issue in gem.issues[:3]:
            result["warnings"].append(f"Gemini noted: {g_issue}")
    from app.pipeline.diagnostics import write_diagnostics
    result["diagnostics"] = bool(S.run("diagnostics", lambda: write_diagnostics(
        files.id, kin, events, ctx, key_frames, result, scale.body_height_px,
        video=files.side_prep if render else None, pose3d=pose3d or None), False))
    result["stages"] = S.items
    result = AnalysisResult.model_validate(result).model_dump(mode="json")
    if render:
        S.run("pdf", lambda: build_pdf(result, key_frames, files.pdf))
    on_stage("Writing report", 99)
    return result


def _no_second(st: Prepared) -> dict:
    return {"events": None, "q": None, "kin": None, "ev": None, "view": None, "sync": None,
            "fail": st.front_fail or ("The second clip could not be analysed." if st.has_front_input else None)}


def _second_clip(files, st: Prepared, prep, kin, events, front_pose, front_prep, warnings: list[str]) -> dict:
    """The optional second (front-on/behind) clip: its own key moments, the sync and its quality."""
    out = _no_second(st)
    if front_pose is None or front_prep is None:
        if st.front_fail:
            warnings.append(st.front_fail)
        return out
    front_fail = st.front_fail
    front_note = None
    front_kin = smooth_pose(front_pose, bowling_arm=kin.bowling_arm, direction="left_to_right")
    front_ev = None
    front_release_raw = None
    try:
        front_ev = detect_delivery(front_kin, _front_body_height(front_kin), mode="front", strict=True)
        front_release_raw = front_ev.release.frame / front_kin.fps
        # cross-check: both clips show the same delivery, so BFC→FFC must take about as long in each
        t_side = (events.ffc.frame - events.bfc.frame) / prep.fps
        t_front = (front_ev.ffc.frame - front_ev.bfc.frame) / front_kin.fps
        if abs(t_side - t_front) > max(0.05, 0.4 * t_side):
            front_ev = None
            front_fail = (f"Second clip: its key moments don't match the main clip (back foot to front foot "
                          f"{t_front:.2f} s vs {t_side:.2f} s), so they couldn't be found reliably.")
    except BowlIQError as exc:
        front_fail = f"Second clip: {exc.message}"
    lo = events.bfc.frame - frames(0.4, prep.fps)
    hi = events.release.frame + frames(0.3, prep.fps)
    body_match = motion_matcher(motion_signals(kin), prep.fps, motion_signals(front_kin), front_kin.fps, lo, hi)
    span = max(prep.duration_s, front_prep.duration_s)
    motion_best = motion_search(body_match, max_offset=min(30.0, span))
    sync = decide(st.side_audio, st.front_audio, events.release.frame / prep.fps,
                  front_ev.release.frame / front_kin.fps if front_ev else None, prep.fps, front_release_raw,
                  motion=body_match, motion_best=motion_best)
    if sync.method == "none":
        warnings.append(f"The two clips couldn't be lined up in time. {sync.detail}")
    main_events = None
    own = front_ev
    mapped = (map_events(events, prep.fps, sync.offset_s, front_kin.fps, front_kin.n, sync.confidence,
                         "sound" if sync.method == "audio" else "body-motion")
              if sync.method in ("audio", "motion") else None)
    if mapped is not None:
        # the side-on clip times the key moments best; the sync carries them into the second clip exactly
        front_ev, front_fail = mapped, None
        if own is not None and events_agree(own, mapped, front_kin.fps):
            main_events = confirm_events(events, "confirmed in the second clip")
            front_ev = confirm_events(mapped, "confirmed by both clips")
            front_note = "Key moments found independently in both clips agree."
        elif own is not None:
            front_note = ("The second clip's own key-moment detection disagreed with the synced main-clip moments; "
                          "the main-clip moments were used.")
        if sync.release_offset_s is not None and abs(sync.release_offset_s - sync.offset_s) > MAX_SYNC_DISAGREEMENT_S:
            sync.detail = sync.detail.replace("so its own key moments may be off",
                                              "so the side-on key moments were carried over through the sound sync")
    f_boxes = Q.person_boxes(front_pose)
    if front_pose.cam is not None:
        from app.pipeline.adapt import residual_motion
        f_motion = residual_motion(files.front_prep, f_boxes, front_pose.cam)
    else:
        f_motion = Q.camera_motion(files.front_prep, f_boxes)
    f_view = CX.classify_view(Q.world_kp(front_pose), front_ev.bfc.frame if front_ev else None).value
    front_q = Q.finalize(front_prep, front_prep.duration_s, list(st.f_checks), list(st.f_reasons), st.f_level, f_motion,
                         _legacy(f_view), SECOND_VIEWS, front_pose.detection_rate)
    if f_view == "side-on":
        front_fail = ("Second clip: it is also filmed side-on, so it adds no new view. Film the second clip from "
                      "behind the stumps (facing the bowler) or from behind the bowler.")
        front_ev = None
    if front_fail:
        warnings.append(front_fail)
    if front_note:
        warnings.append(front_note)
    return {"events": main_events, "q": front_q, "kin": front_kin, "ev": front_ev, "view": f_view, "sync": sync,
            "fail": front_fail}


def _front_outputs(files: AnalysisFiles, front_prep, front_kin, render: bool) -> Optional[dict]:
    if front_prep is None or not files.front_prep.exists():
        return None
    if render or not files.front_clean.exists():
        encode_seekable(files.front_prep, files.front_clean)
    fn = front_kin.n if front_kin is not None else front_prep.frame_count
    out = {"fps": front_prep.fps, "width": front_prep.width, "height": front_prep.height, "frame_count": fn,
           "duration_s": round(fn / front_prep.fps, 4), "slowmo": {"start_frame": 0, "end_frame": max(0, fn - 1), "factor": 1}}
    if front_kin is not None:
        files.front_keypoints.write_text(json.dumps(_keypoints_json(front_kin, front_prep.fps), separators=(",", ":")))
        fov = OverlayData.from_kinematics(front_kin, {})  # side-on arcs mean nothing here: second-view angles
        fov.fps = front_prep.fps
        fov.trunk_deg = np.full_like(fov.trunk_deg, np.nan)
        fov.view = "second"
        fov.save(files.front_overlay)
    return out


# ───────────────────────────── entry points ─────────────────────────────

def run_analysis(analysis_id: str, params: dict, history: list[dict], on_stage: StageFn,
                 estimator: Optional[PoseEstimator] = None) -> dict:
    files = AnalysisFiles(analysis_id).ensure()
    est = estimator or get_estimator()
    st = prepare(files, params, on_stage, est)
    save_cache(files, st)
    return analyse(files, copy.deepcopy(st), params, history, on_stage)


def recompute(analysis_id: str, params: dict, history: list[dict], render: bool = True) -> dict:
    """Re-run everything after the pose stage from the cached keypoints, with the user's corrections in
    params["overrides"]. Takes seconds (no pose model)."""
    files = AnalysisFiles(analysis_id)
    st = load_cache(files)
    if st is None:
        raise BowlIQError("NO_CACHE", "This delivery was analysed before quick corrections were available, so it can't "
                          "be recomputed.", "Use \"Correct them\" to re-run it, or upload the clip again.", 409)
    if not files.side_prep.exists():
        raise BowlIQError("NOT_FOUND", "The processed video for this delivery is missing.", "Upload the clip again.", 404)
    return analyse(files, st, params, history, lambda s, p: None, render=render)

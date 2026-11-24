"""Per-analysis diagnostics in storage/debug/{id}/ for checking (and tuning, via config/pipeline.yaml) the detection:

  signals.png   ankle speeds, wrist height, mid-hip speed (all in body heights), with the final key moments and every
                detection method's guess marked
  context.json  rotation / direction / arm / view / bowler with confidences and reasons, the events with each method's
                frame, review reasons and stage results
  bfc.jpg, ffc.jpg, release.jpg   the key frames with the skeleton overlay
  pose3d.json   the 3D lifted pose (MotionBERT) for the coach view's 3D viewer
  roles.mp4     the skeleton coloured by role (front leg red, back leg blue, bowling arm green), with every frame
                where left/right labels were swapped back marked
"""
from __future__ import annotations

import json
import logging

import numpy as np

from app.core.storage import AnalysisFiles
from app.pipeline.types import DeliveryEvents, Kinematics

log = logging.getLogger("bowliq.diagnostics")
NAMES = ("signals.png", "context.json", "bfc.jpg", "ffc.jpg", "release.jpg", "roles.mp4", "pose3d.json")
COLORS = {"BFC": "#1f77b4", "FFC": "#d62728", "release": "#2ca02c"}


def _speed(p: np.ndarray, fps: float) -> np.ndarray:
    d = np.diff(p, axis=0) * fps
    d = np.vstack([d, d[-1:]]) if len(d) else np.zeros_like(p)
    return np.hypot(d[:, 0], d[:, 1])


def signals(kin: Kinematics, bh: float) -> dict[str, np.ndarray]:
    fps = kin.fps
    out = {
        "front ankle speed": _speed(kin.joint(kin.side("ankle", "front")), fps) / bh,
        "back ankle speed": _speed(kin.joint(kin.side("ankle", "back")), fps) / bh,
        "mid-hip speed": np.abs(np.gradient(kin.mid("left_hip", "right_hip")[:, 0]) * fps) / bh,
    }
    ank = kin.mid("left_ankle", "right_ankle")[:, 1]
    floor = np.nanpercentile(ank, 5) if np.isfinite(ank).any() else 0.0
    out["bowling wrist height"] = (kin.joint(kin.side("wrist", "bowling"))[:, 1] - floor) / bh
    return out


def plot_signals(path, kin: Kinematics, events: DeliveryEvents, bh: float) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib comes with MediaPipe; without it, skip the picture
        log.info("matplotlib not available: signals.png skipped")
        return False
    sig = signals(kin, bh)
    t = np.arange(kin.n) / kin.fps
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    keys = [("front ankle speed", "body heights / s"), ("back ankle speed", "body heights / s"),
            ("mid-hip speed", "body heights / s"), ("bowling wrist height", "body heights above the feet")]
    evs = (events.bfc, events.ffc, events.release)
    for ax, (k, unit) in zip(axes, keys, strict=True):
        ax.plot(t, sig[k], color="#333", lw=1.2)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_title(k, fontsize=9, loc="left")
        for e in evs:
            ax.axvline(e.frame / kin.fps, color=COLORS[e.name], lw=1.6, alpha=0.9)
            for m, f in (e.methods or {}).items():
                if f is not None and m != "user":
                    ax.axvline(f / kin.fps, color=COLORS[e.name], lw=0.8, ls=":", alpha=0.7)
        ax.grid(alpha=0.25)
    lines = [f"{e.name}: frame {e.frame} ({e.confidence}) — " +
             ", ".join(f"{m} {f if f is not None else '–'}" for m, f in (e.methods or {}).items()) for e in evs]
    axes[0].text(0.01, 0.97, "\n".join(lines), transform=axes[0].transAxes, fontsize=7, va="top",
                 bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#ccc"})
    axes[-1].set_xlabel(f"time (s) at {kin.fps:g} fps — solid: final key moment, dotted: each method's guess")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return True


ROLE_COLORS = {"front": (60, 60, 240), "back": (240, 150, 40), "bowling": (60, 220, 60), "other": (190, 190, 190)}


def roles_video(video, kin: Kinematics, out) -> bool:
    """Debug video: the skeleton coloured by ROLE (front leg red, back leg blue, bowling arm green, rest grey), frames
    where left/right labels were swapped back marked, and the key moments named."""
    import cv2

    from app.pipeline.types import J
    from app.report.render_video import _Encoder
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    swapped = {f for e in kin.identity_log if e.get("corrected", True) for f in range(e["frames"][0], e["frames"][1] + 1)}
    roles = kin.roles or {}
    fl, bl = roles.get("front_leg"), roles.get("back_leg")
    arm = kin.bowling_arm
    other_arm = "left" if arm == "right" else "right"
    segs = []
    for side, role in ((fl, "front"), (bl, "back")):
        if side:
            segs += [((f"{side}_hip", f"{side}_knee"), role), ((f"{side}_knee", f"{side}_ankle"), role),
                     ((f"{side}_ankle", f"{side}_heel"), role), ((f"{side}_heel", f"{side}_foot_index"), role),
                     ((f"{side}_ankle", f"{side}_foot_index"), role)]
    segs += [((f"{arm}_shoulder", f"{arm}_elbow"), "bowling"), ((f"{arm}_elbow", f"{arm}_wrist"), "bowling"),
             ((f"{other_arm}_shoulder", f"{other_arm}_elbow"), "other"), ((f"{other_arm}_elbow", f"{other_arm}_wrist"), "other"),
             (("left_shoulder", "right_shoulder"), "other"), (("left_hip", "right_hip"), "other"),
             (("left_shoulder", "left_hip"), "other"), (("right_shoulder", "right_hip"), "other")]
    enc = _Encoder(out, w - w % 2, h - h % 2, kin.fps)
    ok_all = True
    try:
        for f in range(kin.n):
            ok, frame = cap.read()
            if not ok:
                break
            frame = frame[: h - h % 2, : w - w % 2]
            P2 = kin.img_xy[f]
            for (a, b), role in segs:
                pa, pb = P2[J[a]], P2[J[b]]
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    cv2.line(frame, tuple(int(v) for v in pa), tuple(int(v) for v in pb), ROLE_COLORS[role], 3, cv2.LINE_AA)
            cv2.putText(frame, f"F{f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if f in swapped:
                cv2.putText(frame, "L/R swap corrected", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            enc.write(frame)
    except Exception:
        ok_all = False
        raise
    finally:
        cap.release()
        enc.close(ok_all)
    return True


def write_diagnostics(analysis_id: str, kin: Kinematics, events: DeliveryEvents, ctx, key_frames: dict[str, bytes],
                      result: dict, bh: float, video=None, pose3d: dict | None = None) -> bool:
    d = AnalysisFiles(analysis_id).debug_dir
    d.mkdir(parents=True, exist_ok=True)
    info = {"id": analysis_id, "fps": kin.fps, "frames": kin.n, "body_height_px": round(float(bh), 2),
            "bowling_arm": kin.bowling_arm, "direction": kin.direction, "context": ctx.to_dict(),
            "events": result.get("events"), "method": events.method, "needs_review": result.get("needs_review"),
            "review_reasons": result.get("review_reasons"), "stages": result.get("stages"),
            "overrides": result.get("overrides"), "warnings": result.get("warnings")}
    (d / "context.json").write_text(json.dumps(info, indent=2, default=str))
    for name, data in key_frames.items():
        (d / f"{name}.jpg").write_bytes(data)
    plot_signals(d / "signals.png", kin, events, max(float(bh), 1.0))
    p3 = d / "pose3d.json"
    if pose3d:
        p3.write_text(json.dumps(pose3d, separators=(",", ":")))
    elif p3.exists():  # recomputed without 3D: don't leave the old pose behind
        p3.unlink()
    if video is not None:
        try:
            roles_video(video, kin, d / "roles.mp4")
        except Exception:  # a debug aid: never fails the analysis
            log.exception("roles video failed")
    return True

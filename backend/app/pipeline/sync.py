"""Phase 2: line up the side-on and front-on clips in time.

The two phones start recording at different moments, so frame N of one clip is not frame N of the other. Two methods:

1. **Sound** (preferred). Both phones hear the same sounds: a clap at the start, footsteps, the ball hitting the
   ground or the bat. Two independent matchers look for the time shift between the soundtracks:
   (a) GCC-PHAT on the waveforms (sample-accurate) and (b) cross-correlation of *onset envelopes* (where the sound
   suddenly gets louder, 1 ms resolution). The sound sync is accepted only when both land on the same shift
   (within 4 ms), and the rest of the soundtracks must still line up at that shift with the matched sound blanked
   out (so two unrelated clips that each contain one clap can't be paired). Calibration: on unrelated clips the two always disagreed by 0.2–0.9 s; on simulated two-phone
   recordings (different mic, room echo, background noise, AAC re-encode) they agreed in 31 of 36 cases, and the
   rest fell back to method 2. Sound travels ~3 ms per metre, so clap about equally far from both phones.
2. **Release** (backup, about ±1–2 frames). The release moment found in each clip is taken to be the same instant.

Convention everywhere: ``front_time = side_time + offset_s``, both measured from the first frame of the processed
(trimmed) videos.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app.core.ffmpeg import SAFE_INPUT, ffmpeg_path, ffprobe_path

log = logging.getLogger("bowliq.sync")

SR = 16000  # audio sample rate used for matching
HOP = 16  # 1 ms analysis hop at 16 kHz
AGREE_S = 0.004  # the two sound matchers must agree within this
STRONG_RATIO = 3.0  # GCC-PHAT peak this much above any other shift = an unmistakable shared sound (e.g. a clap)
MIN_RMS = 1e-4  # quieter than this = no usable sound (muted / silent track)
AGREE_FRAMES = 3  # sound and release offsets agreeing within this many frames confirm each other


@dataclass
class AudioTrack:
    samples: np.ndarray  # mono float32 at SR; sample 0 ≙ first frame of the processed video
    duration_s: float


@dataclass
class SyncResult:
    method: str  # "audio" | "motion" | "release" | "none"
    offset_s: Optional[float]  # front_time = side_time + offset_s
    confidence: str  # "High" | "Medium" | "Low"
    detail: str
    audio_offset_s: Optional[float] = None
    audio_score: Optional[float] = None
    audio_peak_ratio: Optional[float] = None
    anchor_side_s: Optional[float] = None  # side-clip time of the sharpest shared sound (usually the clap)
    release_offset_s: Optional[float] = None
    motion_corr: Optional[float] = None  # how well the up-and-down body motion matches at offset_s (1 = perfectly)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


# ───────────────────────────── audio extraction ─────────────────────────────

def _stream_starts(path: Path) -> tuple[Optional[float], Optional[float]]:
    """(video start_time, audio start_time) in seconds, None when the stream is missing."""
    try:
        out = subprocess.run([ffprobe_path(), "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json",
                              str(path)], capture_output=True, timeout=60, check=False)
        streams = json.loads(out.stdout or b"{}").get("streams", [])
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None, None
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def st(s):
        try:
            return float(s.get("start_time", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
    return (st(v) if v else None), (st(a) if a else None)


def extract_audio(path: Path, start: float = 0.0, duration: Optional[float] = None) -> Optional[AudioTrack]:
    """Mono 16 kHz soundtrack aligned to the processed video (sample 0 = first video frame after trimming).
    None when the clip has no sound or it is silent."""
    v0, a0 = _stream_starts(path)
    if a0 is None or v0 is None:
        return None
    cmd = [ffmpeg_path(), "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += [*SAFE_INPUT, "-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    # first_pts=0 pads a late-starting soundtrack with silence so timestamps stay aligned with the video
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-af", "aresample=async=1:first_pts=0", "-f", "f32le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    x = np.frombuffer(out.stdout, dtype="<f4").astype(np.float32)
    if start <= 0:
        # ffmpeg measures from the earliest stream; the processed video starts at the first video frame
        lead = max(0.0, v0 - min(v0, a0))
        x = x[int(round(lead * SR)):]
    if len(x) < SR // 2 or float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) < MIN_RMS:
        return None
    return AudioTrack(x, len(x) / SR)


# ───────────────────────────── matching ─────────────────────────────

def onset_envelope(x: np.ndarray) -> np.ndarray:
    """1 ms-resolution 'sudden loudness increase' curve, emphasising sharp sounds (claps, ball impacts)."""
    hp = np.diff(x, prepend=x[:1])  # first difference ≈ high-pass: removes wind rumble and hum
    n = len(hp) // HOP
    if n < 4:
        return np.zeros(max(n, 1))
    e = np.sqrt(np.mean(hp[: n * HOP].reshape(n, HOP).astype(np.float64) ** 2, axis=1))
    loud = 20 * np.log10(e + 1e-6)
    k = 5
    smooth = np.convolve(loud, np.ones(k) / k, mode="same")
    on = np.maximum(0.0, smooth - np.concatenate([np.full(k, smooth[0]), smooth[:-k]]))
    on -= np.median(on)
    on = np.maximum(on, 0.0)
    sd = on.std()
    return on / sd if sd > 0 else on


def _xcorr(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation c[lag] = Σ a[t] · b[t + lag] for every lag, with the lag values (in hops)."""
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.conj(np.fft.rfft(a, size)) * np.fft.rfft(b, size), size)
    c = np.concatenate([c[-(len(a) - 1):], c[: len(b)]]) if len(a) > 1 else c[: len(b)]
    lags = np.arange(-(len(a) - 1), len(b))
    return c, lags


def match_audio(side: AudioTrack, front: AudioTrack, max_offset_s: Optional[float] = None
                ) -> tuple[Optional[float], float, float, Optional[float]]:
    """Best shift (front_time = side_time + offset) → (offset_s, score, peak_ratio, anchor_side_s)."""
    a, b = onset_envelope(side.samples), onset_envelope(front.samples)
    if not a.any() or not b.any():
        return None, 0.0, 0.0, None
    c, lags = _xcorr(a, b)
    # normalise by the overlap length so short overlaps at extreme shifts can't win by luck
    overlap = np.minimum(len(a), np.minimum(len(b), np.minimum(lags + len(a), len(b) - lags)))
    valid = overlap >= min(len(a), len(b)) * 0.25
    if max_offset_s is not None:
        valid &= np.abs(lags) <= max_offset_s * 1000
    if not valid.any():
        return None, 0.0, 0.0, None
    norm = np.where(valid, c / np.sqrt(np.maximum(overlap, 1)), -np.inf)
    i = int(np.argmax(norm))
    lag = int(lags[i])
    far = valid & (np.abs(lags - lag) > 30)
    second = float(np.max(norm[far])) if far.any() else 0.0
    ratio = float(norm[i] / second) if second > 0 else float("inf")
    # normalised correlation coefficient over the overlap at the best shift
    if lag >= 0:
        sa, sb = a[: len(b) - lag], b[lag: lag + len(a)]
    else:
        sa, sb = a[-lag: -lag + len(b)], b[: len(a) + lag]
    m = min(len(sa), len(sb))
    sa, sb = sa[:m], sb[:m]
    den = float(np.linalg.norm(sa) * np.linalg.norm(sb))
    score = float(np.dot(sa, sb) / den) if den > 0 else 0.0
    prod = sa * sb
    anchor = (int(np.argmax(prod)) + (0 if lag >= 0 else -lag)) / 1000.0 if prod.any() else None
    return lag / 1000.0, score, ratio, anchor


def gcc_phat(side: AudioTrack, front: AudioTrack, max_offset_s: Optional[float] = None
             ) -> tuple[Optional[float], float]:
    """Waveform match with phase-only weighting (GCC-PHAT, 300 Hz–6 kHz) → (offset_s, peak_ratio).

    Phase-only weighting makes the peak razor-sharp when both mics heard the same sounds, regardless of the
    microphones' different tone, gain or echo. peak_ratio = best peak / best peak more than 5 ms away."""
    a = np.diff(side.samples.astype(np.float64))
    b = np.diff(front.samples.astype(np.float64))
    if len(a) < 2 or len(b) < 2:
        return None, 0.0
    n = len(a) + len(b)
    size = 1 << (n - 1).bit_length()
    x = np.conj(np.fft.rfft(a, size)) * np.fft.rfft(b, size)
    f = np.fft.rfftfreq(size, 1 / SR)
    x = np.where((f > 300) & (f < 6000), x / np.maximum(np.abs(x), 1e-12), 0)
    c = np.fft.irfft(x, size)
    c = np.abs(np.concatenate([c[-(len(a) - 1):], c[: len(b)]]))
    lags = np.arange(-(len(a) - 1), len(b))
    overlap = np.minimum(len(a), np.minimum(len(b), np.minimum(lags + len(a), len(b) - lags)))
    valid = overlap >= min(len(a), len(b)) * 0.25
    if max_offset_s is not None:
        valid &= np.abs(lags) <= max_offset_s * SR
    if not valid.any():
        return None, 0.0
    cv = np.where(valid, c, -np.inf)
    i = int(np.argmax(cv))
    far = valid & (np.abs(lags - lags[i]) > int(0.005 * SR))
    second = float(np.max(c[far])) if far.any() else 0.0
    return float(lags[i]) / SR, (float(c[i] / second) if second > 0 else float("inf"))


ANCHOR_MASK_S = 0.15


def _confirmed_without_anchor(side: AudioTrack, front: AudioTrack, offset: float, anchor: Optional[float]) -> bool:
    """Blank the matched sound (e.g. the clap) in both tracks and check the rest of the soundtracks (footsteps,
    ambient noise) still line up at the same shift. Two unrelated clips that each contain one clap fail this."""
    if anchor is None:
        return False
    a, b = side.samples.copy(), front.samples.copy()
    rng = np.random.default_rng(12345)
    for x, t in ((a, anchor), (b, anchor + offset)):
        i0, i1 = int(max(0.0, t - ANCHOR_MASK_S) * SR), int(max(0.0, t + ANCHOR_MASK_S) * SR)
        i1 = min(i1, len(x))
        if i1 <= i0:
            continue
        # replace with independent noise at the surrounding loudness: silence would create a matching "edge"
        ctx = np.concatenate([x[max(0, i0 - SR // 4):i0], x[i1:i1 + SR // 4]])
        level = float(np.median(np.abs(ctx))) * 1.25 if len(ctx) else 0.0
        x[i0:i1] = (level * rng.standard_normal(i1 - i0)).astype(x.dtype)
    ma, mb = AudioTrack(a, side.duration_s), AudioTrack(b, front.duration_s)
    g, _ = gcc_phat(ma, mb)
    if g is not None and abs(g - offset) <= AGREE_S:
        return True
    o, *_ = match_audio(ma, mb)
    return o is not None and abs(o - offset) <= AGREE_S


# ───────────────────────────── body-motion check ─────────────────────────────
# Vertical movement looks the same from any horizontal viewing direction (side-on, front-on, behind), so the
# up-and-down paths of the bowling wrist and the head, scaled by body size, must match between the clips at the right
# offset. Measured: the same delivery 0.96 with a clear single peak; two different deliveries 0.52 at the offset the
# sound and release happened to agree on, with no clear peak anywhere (best 0.87 vs 0.83 elsewhere).
MOTION_CONFIRM = 0.7  # needed to accept a sync that isn't a clean sound match
MOTION_SYNC = 0.85  # a motion-only sync needs this peak ...
MOTION_MARGIN = 0.1  # ... beating every offset more than 0.15 s away by this much


def motion_signals(kin) -> np.ndarray:
    """(n, 2): bowling-wrist height and head height above the mid-hip, in body heights (Y-up kinematics)."""
    from app.pipeline.types import J

    def seg(a, b):
        return np.linalg.norm(kin.xy[:, J[a]] - kin.xy[:, J[b]], axis=1)
    leg = np.fmax(seg("left_hip", "left_knee") + seg("left_knee", "left_ankle"),
                  seg("right_hip", "right_knee") + seg("right_knee", "right_ankle"))
    bh = leg + np.linalg.norm(kin.mid("left_shoulder", "right_shoulder") - kin.mid("left_hip", "right_hip"), axis=1)
    ok = np.isfinite(bh)
    if ok.sum() < 3:
        return np.full((kin.n, 2), np.nan)
    idx = np.arange(kin.n)
    bh = np.interp(idx, idx[ok], bh[ok])
    k = max(3, int(0.15 * kin.fps) | 1)  # running median: body size changes slowly
    pad = np.pad(bh, k // 2, mode="edge")
    bh = np.array([np.median(pad[i:i + k]) for i in range(kin.n)])
    mh = kin.mid("left_hip", "right_hip")[:, 1]
    wr = kin.xy[:, J[f"{kin.bowling_arm}_wrist"], 1]
    nose = kin.xy[:, J["nose"], 1]
    return np.stack([(wr - mh) / bh, (nose - mh) / bh], axis=1)


def motion_matcher(side_sig: np.ndarray, side_fps: float, front_sig: np.ndarray, front_fps: float, lo: int, hi: int
                   ) -> Callable[[float], Optional[float]]:
    """corr(offset): mean correlation of the two signals over side frames [lo, hi) mapped into the second clip."""
    lo, hi = max(0, lo), min(len(side_sig), hi)

    def corr(offset: float) -> Optional[float]:
        t = np.arange(lo, hi) / side_fps
        j = np.round((t + offset) * front_fps).astype(int)
        inside = (j >= 0) & (j < len(front_sig))
        x, y = side_sig[lo:hi][inside], front_sig[j[inside]]
        m = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
        if m.sum() < max(10, 0.6 * (hi - lo)):
            return None
        cs = []
        for c in range(x.shape[1]):
            a, b = x[m, c], y[m, c]
            if a.std() < 1e-6 or b.std() < 1e-6:
                return None
            cs.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(cs))
    return corr


def motion_search(corr: Callable[[float], Optional[float]], max_offset: float = 30.0, step: float = 0.005
                  ) -> Optional[tuple[float, float, float]]:
    """Best offset by body motion → (offset, corr, margin over the best offset > 0.15 s away)."""
    offs = np.arange(-max_offset, max_offset, step)
    cs = np.array([np.nan if (c := corr(o)) is None else c for o in offs])
    if not np.isfinite(cs).any():
        return None
    i = int(np.nanargmax(cs))
    far = np.abs(offs - offs[i]) > 0.15
    second = float(np.nanmax(cs[far])) if np.isfinite(cs[far]).any() else -1.0
    return float(offs[i]), float(cs[i]), float(cs[i] - second)


# ───────────────────────────── decision ─────────────────────────────

def decide(side_audio: Optional[AudioTrack], front_audio: Optional[AudioTrack], side_release_s: Optional[float],
           front_release_s: Optional[float], fps: float, front_release_raw_s: Optional[float] = None,
           motion: Optional[Callable[[float], Optional[float]]] = None,
           motion_best: Optional[tuple[float, float, float]] = None) -> SyncResult:
    """Choose the sync method and say honestly how good it is.

    front_release_s: release found in the second clip *and* validated against the side-on timing (usable on its own);
    front_release_raw_s: release found in the second clip before that check (only used to confirm the sound)."""
    rel = (front_release_s - side_release_s) if (side_release_s is not None and front_release_s is not None) else None
    raw = front_release_raw_s if front_release_raw_s is not None else front_release_s
    rel_raw = (raw - side_release_s) if (side_release_s is not None and raw is not None) else None
    frame = 1.0 / max(fps, 1.0)
    a_off = a_score = a_ratio = anchor = None
    o_off = None
    why_not_audio = ""
    if side_audio is None or front_audio is None:
        which = "either clip" if side_audio is None and front_audio is None else (
            "the side-on clip" if side_audio is None else "the second clip")
        why_not_audio = f"No usable sound in {which}."
    else:
        g_off, a_ratio = gcc_phat(side_audio, front_audio)
        o_off, a_score, _, anchor = match_audio(side_audio, front_audio)
        if g_off is not None and o_off is not None and abs(g_off - o_off) <= AGREE_S and \
                _confirmed_without_anchor(side_audio, front_audio, g_off, anchor):
            a_off = g_off
        else:
            why_not_audio = ("The two soundtracks don't share a clear sound (clap once, loudly, in view of both phones "
                             "when you start recording).")
    def mcorr(off: Optional[float]) -> Optional[float]:
        return None if (motion is None or off is None) else motion(off)

    confirmed_by_motion = False
    if why_not_audio and o_off is not None and rel_raw is not None and abs(rel_raw - o_off) <= AGREE_FRAMES * frame \
            and (mcorr(o_off) or -1.0) >= MOTION_CONFIRM:
        # The waveforms don't match (phones process sound differently: noise suppression, compression), but the
        # sound onsets line up at the same shift as the release moment seen independently in the two videos.
        # Two unrelated signals agreeing within a few frames, plus the body moving the same way in both videos at that
        # shift, is strong evidence they show the same delivery (without the motion check, two different deliveries
        # were once paired this way by coincidence).
        a_off, why_not_audio, confirmed_by_motion = o_off, "", True
    audio_ok = a_off is not None and not why_not_audio
    common = {"audio_offset_s": a_off, "audio_score": a_score,
              "audio_peak_ratio": None if a_ratio is None or not np.isfinite(a_ratio) else a_ratio,
              "anchor_side_s": anchor if audio_ok else None, "release_offset_s": rel,
              "motion_corr": mcorr(a_off) if audio_ok else None}
    if audio_ok and confirmed_by_motion:
        common["release_offset_s"] = rel_raw
        return SyncResult("audio", a_off, "Medium",
                          f"Synced on the sound onsets (claps, footsteps, the ball), confirmed by the release moment seen "
                          f"in both videos ({abs(rel_raw - a_off) * 1000:.0f} ms apart). The exact waveforms differ "
                          "between the phones, so this is accurate to about a frame.", **common)
    if audio_ok:
        where = f" at {anchor:.2f} s of the side-on clip" if anchor is not None else ""
        if rel is not None and abs(rel - a_off) <= AGREE_FRAMES * frame:
            return SyncResult("audio", a_off, "High",
                              f"Synced on a shared sound{where}, confirmed by the release moment in both clips "
                              f"({abs(rel - a_off) * 1000:.0f} ms apart).", **common)
        if rel is not None:
            return SyncResult("audio", a_off, "Medium",
                              f"Synced on a shared sound{where}. The release moment found in the second clip is "
                              f"{abs(rel - a_off):.2f} s away from the synced moment, so its own key moments may be off.",
                              **common)
        return SyncResult("audio", a_off, "High" if a_ratio >= STRONG_RATIO else "Medium",
                          f"Synced on a shared sound{where}.", **common)
    mrel = mcorr(rel)
    if rel is not None and (motion is None or (mrel is not None and mrel >= MOTION_CONFIRM)):
        common["motion_corr"] = mrel
        return SyncResult("release", rel, "Medium",
                          f"{why_not_audio} Synced on the release moment instead (accurate to about 1–2 frames).", **common)
    if motion_best is not None and motion_best[1] >= MOTION_SYNC and motion_best[2] >= MOTION_MARGIN:
        off, c, _ = motion_best
        common["motion_corr"] = c
        return SyncResult("motion", off, "Medium",
                          f"{why_not_audio} Synced on the body's up-and-down movement, which matches in both videos "
                          f"(match {c:.2f}); accurate to about 1–2 frames.", **common)
    if rel is not None:
        why_not_audio += (f" The release moments suggest a {rel:+.2f} s shift, but the body's movement doesn't match at "
                          "that shift, so the clips may show different deliveries.")
    extra = "" if rel is not None else " The release moment couldn't be confirmed in the second clip either."
    return SyncResult("none", None, "Low",
                      f"{why_not_audio}{extra} The clips aren't synced; line them up by hand in Linked playback.",
                      **common)

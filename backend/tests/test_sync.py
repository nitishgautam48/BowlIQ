"""Phase 2: lining up the side-on and front-on clips (sound first, release moment as backup, manual override)."""
import json
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.ffmpeg import ffmpeg_path
from app.models.db import Analysis, session
from app.pipeline.sync import SR, AudioTrack, decide, extract_audio, gcc_phat, match_audio
from tests.apiclient import API, ensure_sample, headers

rng = np.random.default_rng(7)


def _scene(seconds=5.0, claps=(2.0,), events=12) -> np.ndarray:
    """Background noise, a few footstep-like thuds and loud claps."""
    n = int(seconds * SR)
    x = 0.01 * rng.standard_normal(n)
    for t in rng.uniform(0.2, seconds - 0.2, events):
        i = int(t * SR)
        x[i:i + 800] += 0.08 * rng.standard_normal(800) * np.exp(-np.arange(800) / 150)
    for t in claps:
        i = int(t * SR)
        x[i:i + 600] += 0.6 * rng.standard_normal(600) * np.exp(-np.arange(600) / 60)
    return x


def _second_phone(x: np.ndarray, shift_s: float, snr_db: float = 10.0) -> np.ndarray:
    """Same sounds heard by another phone: echo, duller mic, own noise, starts recording at a different moment."""
    rir = np.zeros(int(0.2 * SR))
    rir[0] = 1.0
    rir[1:] = 0.15 * rng.standard_normal(len(rir) - 1) * np.exp(-np.arange(1, len(rir)) / (0.04 * SR))
    y = np.convolve(x, rir)[: len(x)]
    y = np.convolve(y, np.ones(4) / 4, mode="same") * 0.5
    y += np.std(y) / 10 ** (snr_db / 20) * rng.standard_normal(len(y))
    k = int(round(shift_s * SR))
    return np.concatenate([0.01 * rng.standard_normal(k), y]) if k > 0 else y[-k:]


def _t(x: np.ndarray) -> AudioTrack:
    return AudioTrack(x.astype(np.float32), len(x) / SR)


@pytest.mark.parametrize("shift", [0.75, -1.2, 0.0])
def test_sound_sync_recovers_the_offset_to_a_millisecond(shift):
    side = _scene()
    front = _second_phone(side, shift)
    g, _ratio = gcc_phat(_t(side), _t(front))
    o, *_ = match_audio(_t(side), _t(front))
    assert g == pytest.approx(shift, abs=0.001) and o == pytest.approx(shift, abs=0.002)
    r = decide(_t(side), _t(front), side_release_s=3.0, front_release_s=3.0 + shift + 0.01, fps=60)
    assert r.method == "audio" and r.confidence == "High" and r.offset_s == pytest.approx(shift, abs=0.001)
    assert r.anchor_side_s == pytest.approx(2.0, abs=0.05)  # the clap


def test_unrelated_soundtracks_are_not_trusted_and_release_is_used():
    a, b = _scene(claps=(0.5,)), _scene(claps=(1.7,))
    r = decide(_t(a), _t(b), side_release_s=3.0, front_release_s=2.5, fps=60)
    assert r.method == "release" and r.offset_s == pytest.approx(-0.5) and r.confidence == "Medium"
    assert "don't share a clear sound" in r.detail and r.anchor_side_s is None
    none = decide(_t(a), _t(b), side_release_s=3.0, front_release_s=None, fps=60)
    assert none.method == "none" and none.offset_s is None and "by hand" in none.detail


def test_sound_and_release_disagreeing_lowers_confidence():
    side = _scene()
    front = _second_phone(side, 0.4)
    r = decide(_t(side), _t(front), side_release_s=3.0, front_release_s=3.0 + 0.4 + 0.2, fps=60)
    assert r.method == "audio" and r.confidence == "Medium" and "0.20 s away" in r.detail


def test_missing_sound_is_explained():
    r = decide(None, _t(_scene()), 3.0, 3.1, 60)
    assert r.method == "release" and "No usable sound in the side-on clip" in r.detail


def test_extract_audio_is_aligned_with_the_trimmed_video(tmp_path):
    """A click at 1.50 s in the file must appear at 1.00 s when the clip is trimmed to start at 0.50 s."""
    clip = tmp_path / "click.mp4"
    subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=30:d=3",
                    "-f", "lavfi", "-i", "aevalsrc='0.9*sin(2*PI*3000*t)*between(t,1.5,1.52)+0.002*random(0)':s=44100:d=3",
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(clip)], check=True)
    for start, expect in ((0.0, 1.5), (0.5, 1.0)):
        tr = extract_audio(clip, start)
        assert tr is not None
        peak = int(np.argmax(np.abs(tr.samples))) / SR
        assert peak == pytest.approx(expect, abs=0.03)
    silent = tmp_path / "silent.mp4"
    subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=30:d=2",
                    "-c:v", "libx264", str(silent)], check=True)
    assert extract_audio(silent) is None


@pytest.fixture(scope="module")
def client():
    ensure_sample()
    from app.main import app

    with TestClient(app) as c:
        yield c


def _row(aid: str, sync):
    with session() as s:
        s.add(Analysis(id=aid, owner_id="tester", bowler_name="Sync test", result_json=json.dumps({"sync": sync})))
        s.commit()


def test_manual_alignment_saved_and_reset(client):
    _row("syncrow1", {"method": "audio", "offset_s": -0.8, "confidence": "High", "detail": "x"})
    h = {**headers(), "Content-Type": "application/json"}
    r = client.put(f"{API}/analysis/syncrow1/sync", content=json.dumps({"offset_s": -0.8167}), headers=h)
    assert r.status_code == 200 and r.json()["manual_offset_s"] == -0.8167 and r.json()["offset_s"] == -0.8
    assert client.put(f"{API}/analysis/syncrow1/sync", content=json.dumps({"offset_s": None}), headers=h).json()["manual_offset_s"] is None
    assert client.put(f"{API}/analysis/syncrow1/sync", content='{"offset_s": 999}', headers=h).status_code == 422
    assert client.put(f"{API}/analysis/syncrow1/sync", content='{"offset_s": NaN}', headers=h).status_code in (400, 422)
    _row("syncrow2", None)
    assert client.put(f"{API}/analysis/syncrow2/sync", content='{"offset_s": 0}', headers=h).json()["code"] == "NO_FRONT_CLIP"
    other = {**headers("other-token-0123456789abcdef"), "Content-Type": "application/json"}
    assert client.put(f"{API}/analysis/syncrow1/sync", content='{"offset_s": 0}', headers=other).status_code == 404


# ───────────────────── body-motion check (found on a real pair of different deliveries) ─────────────────────
from app.pipeline.sync import motion_matcher, motion_search  # noqa: E402


def _motion(n=300, fps=60.0, seed=1):
    r = np.random.default_rng(seed)
    t = np.arange(n) / fps
    wrist = np.exp(-((t - 3.0) / 0.12) ** 2) * 1.2 + 0.05 * r.standard_normal(n)  # arm over the top at 3.0 s
    head = 0.6 - 0.5 * (t > 3.2) * np.minimum(1, (t - 3.2) / 0.3) + 0.03 * r.standard_normal(n)  # bends over after
    return np.stack([wrist, head], 1)


def test_motion_matcher_finds_the_shift_and_rejects_a_different_delivery():
    side = _motion()
    front = np.roll(_motion(seed=2), -60, axis=0)  # same movement, second phone started 1.0 s later
    corr = motion_matcher(side, 60.0, front, 60.0, 150, 220)
    off, c, margin = motion_search(corr, max_offset=4.0)
    assert off == pytest.approx(-1.0, abs=0.02) and c > 0.85 and margin > 0.1
    other = _motion(seed=3)
    other[:, 1] = 0.6 + 0.03 * np.random.default_rng(4).standard_normal(len(other))  # no bend-over: another ball
    assert motion_matcher(side, 60.0, np.roll(other, -60, 0), 60.0, 150, 220)(-1.0) < 0.7


def test_sound_onsets_plus_release_need_matching_motion():
    """Real case: onsets (−1.970 s) and the release moments (−1.94 s) agreed by coincidence on two different
    deliveries; the body movement didn't match there (0.52), so they must not be paired."""
    a, b = _scene(claps=(0.5,)), _scene(claps=(1.7,))  # waveforms don't match
    o_off, *_ = match_audio(_t(a), _t(b))
    common = dict(side_release_s=3.0, front_release_s=None, fps=60, front_release_raw_s=3.0 + o_off)
    assert decide(_t(a), _t(b), **common, motion=lambda off: 0.52).method == "none"
    ok = decide(_t(a), _t(b), **common, motion=lambda off: 0.9)
    assert ok.method == "audio" and ok.confidence == "Medium" and ok.motion_corr == 0.9


def test_release_sync_needs_matching_motion_and_motion_sync_fallback():
    assert decide(None, None, 3.0, 2.5, 60, motion=lambda off: 0.3).method == "none"
    assert decide(None, None, 3.0, 2.5, 60, motion=lambda off: 0.8).method == "release"
    m = decide(None, None, 3.0, None, 60, motion=lambda off: None, motion_best=(-0.82, 0.96, 0.4))
    assert m.method == "motion" and m.offset_s == -0.82 and "movement" in m.detail
    assert decide(None, None, 3.0, None, 60, motion_best=(-1.6, 0.87, 0.04)).method == "none"  # no clear peak

"""Manual assist and graceful degradation: corrections recompute from the cached keypoints in place, are logged, and
write diagnostics; a broken stage never loses the whole analysis."""
import json

import pytest
from fastapi.testclient import TestClient

from tests.apiclient import API, ensure_sample, headers, poll, upload


@pytest.fixture(scope="module")
def client():
    ensure_sample()
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def analysed(client):
    job = poll(client, upload(client, bowler_name="Assist", bowler_height_cm="180").json()["job_id"])
    assert job["status"] == "done", job
    return job["result"]


def _recompute(client, aid, body):
    return client.post(f"{API}/analysis/{aid}/recompute", content=json.dumps(body),
                       headers={**headers(), "Content-Type": "application/json"})


def test_result_reports_context_with_confidence(analysed):
    ctx = analysed["context"]
    for k in ("rotation", "direction", "bowling_arm", "view", "bowler"):
        assert ctx[k]["confidence"] in ("Low", "Medium", "High") and ctx[k]["reasons"]
    assert ctx["view"]["value"] == "side-on" and ctx["bowling_arm"]["value"] == "right"
    assert analysed["can_recompute"] is True
    assert all(s["ok"] for s in analysed["stages"]), analysed["stages"]
    ev = analysed["events"]["release"]
    assert ev["methods"] and ev["agree"] >= 1


def test_recompute_in_place_with_overrides_and_log(client, analysed):
    from app.core.config import settings

    aid = analysed["id"]
    ev = {k: analysed["events"][k]["frame"] for k in ("bfc", "ffc", "release")}
    r = _recompute(client, aid, {"release_frame": ev["release"] + 2, "bowling_arm": "right"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["id"] == aid  # same delivery, updated in place
    assert res["events"]["release"]["frame"] == ev["release"] + 2
    assert res["events"]["release"]["methods"] == {"user": ev["release"] + 2}
    assert res["context"]["bowling_arm"]["source"] == "user"
    assert res["overrides"]["release_frame"] == ev["release"] + 2
    assert res["recomputed_at"]
    log = [json.loads(line) for line in (settings.storage_dir / "corrections.jsonl").read_text().splitlines()]
    mine = [x for x in log if x["clip_id"] == aid]
    assert {"field": "release_frame", "auto": ev["release"], "corrected": ev["release"] + 2} == \
        {k: next(x for x in mine if x["field"] == "release_frame")[k] for k in ("field", "auto", "corrected")}
    # stored: a reload returns the corrected result; undo restores the automatic moments
    assert client.get(f"{API}/analysis/{aid}", headers=headers()).json()["events"]["release"]["frame"] == ev["release"] + 2
    back = _recompute(client, aid, {"clear": True}).json()
    assert back["events"]["release"]["frame"] == ev["release"] and back["overrides"] == {}


def test_recompute_rejects_bad_frames(client, analysed):
    aid = analysed["id"]
    n = analysed["video"]["frame_count"]
    assert _recompute(client, aid, {"release_frame": n + 5}).status_code == 400
    ev = analysed["events"]
    r = _recompute(client, aid, {"ffc_frame": ev["release"]["frame"] + 1})
    assert r.status_code == 400 and r.json()["code"] == "INVALID_EVENTS"


def test_diagnostics_written_and_served(client, analysed):
    from app.core.storage import AnalysisFiles

    aid = analysed["id"]
    d = AnalysisFiles(aid).debug_dir
    assert (d / "context.json").exists() and (d / "signals.png").exists() and (d / "release.jpg").exists()
    r = client.get(f"{API}/analysis/{aid}/debug/context.json", headers=headers())
    assert r.status_code == 200 and r.json()["context"]["view"]["value"] == "side-on"
    assert client.get(f"{API}/analysis/{aid}/debug/signals.png", headers=headers()).headers["content-type"] == "image/png"
    assert client.get(f"{API}/analysis/{aid}/debug/..%2Fsecret", headers=headers()).status_code == 404


def test_recompute_other_owner_is_404(client, analysed):
    from tests.conftest import OTHER
    r = client.post(f"{API}/analysis/{analysed['id']}/recompute", content="{}",
                    headers={**headers(OTHER), "Content-Type": "application/json"})
    assert r.status_code == 404


def test_a_broken_stage_degrades_instead_of_failing(client, analysed, monkeypatch):
    """Ball speed and the report crash: the analysis still completes, those parts say why, needs_review is set."""
    from app.pipeline import orchestrator as O

    def boom(*a, **k):
        raise RuntimeError("simulated bug")
    from app.pipeline import lift3d

    monkeypatch.setattr(O, "measure_ball_speed", boom)
    monkeypatch.setattr(O, "side_metrics", boom)
    monkeypatch.setattr(lift3d, "available", lambda: False)  # no independent 3D source in this check
    res = O.recompute(analysed["id"], {"bowler_name": "Assist", "bowler_height_cm": 180}, [], render=False)
    assert res["needs_review"] is True
    failed = {s["name"] for s in res["stages"] if not s["ok"]}
    assert {"ball speed", "measurements"} <= failed
    assert res["ball_speed"]["status"] == "not_measured"
    assert all(m["status"] == "not_measured" for m in res["metrics"])
    assert res["events"]["release"]["frame"] == analysed["events"]["release"]["frame"]  # unaffected stages still ran


def _clean_index(f: int, v: dict) -> int:
    """Position of source frame f in the served (overlay-free) video, as the browser maps it (frontend/lib/frames.ts)."""
    s0, s1, k = v["slowmo"]["start_frame"], v["slowmo"]["end_frame"], v["slowmo"]["factor"]
    if f < s0:
        return f
    if f <= s1:
        return s0 + (f - s0) * k
    return s0 + (s1 - s0 + 1) * k + (f - s1 - 1)


@pytest.mark.parametrize("slow", [None, 4.0])
def test_skeleton_and_video_share_one_time_base(client, analysed, slow):
    """Found on a user clip: with slow motion (stated or guessed) the served video was encoded at fps × factor while
    the page mapped time → frame at the clip's fps, so every skeleton was drawn on the wrong frame."""
    import cv2

    from app.core.storage import AnalysisFiles
    from app.pipeline import orchestrator as O

    res = O.recompute(analysed["id"], {"bowler_name": "Assist", "bowler_height_cm": 180, "slow_motion": slow}, [])
    f = AnalysisFiles(analysed["id"])
    v = res["video"]
    cap = cv2.VideoCapture(str(f.clean))
    assert abs(cap.get(cv2.CAP_PROP_FPS) - v["fps"]) < 0.01  # served video runs at the reported rate
    assert abs(json.loads(f.keypoints.read_text())["fps"] - v["fps"]) < 0.01
    prep = cv2.VideoCapture(str(f.side_prep))
    for src in (res["events"]["bfc"]["frame"], res["events"]["release"]["frame"], v["frame_count"] - 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, _clean_index(src, v))
        prep.set(cv2.CAP_PROP_POS_FRAMES, src)
        ok1, a = cap.read()
        ok2, b = prep.read()
        assert ok1 and ok2
        assert float(abs(a.astype(int) - b.astype(int)).mean()) < 4.0, f"frame {src} doesn't match the served video"
    cap.release()
    prep.release()


def test_non_square_pixels_are_made_square(tmp_path):
    """A clip with 2:1 pixels is processed in display proportions, so the browser and the skeleton agree."""
    import subprocess

    from app.core.ffmpeg import ffmpeg_path
    from app.pipeline.preprocess import preprocess
    from app.pipeline.quality import probe, probe_video
    from tests.apiclient import SAMPLE

    src = tmp_path / "anamorphic.mp4"
    subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-i", str(SAMPLE), "-t", "1.2", "-vf", "scale=640:720,setsar=2/1",
                    "-c:v", "libx264", "-crf", "20", str(src)], check=True)
    meta = probe_video(src)
    assert (meta.width, meta.height) == (1280, 720)
    out = preprocess(src, tmp_path / "prep.mp4", meta, None, None)
    vs = next(x for x in probe(tmp_path / "prep.mp4")["streams"] if x.get("codec_type") == "video")
    assert (out.width, out.height) == (1280, 720) and vs.get("sample_aspect_ratio", "1:1") in ("1:1", "N/A", "0:1")


def test_3d_lifting_still_measures_angles_when_2d_measurement_breaks(client, analysed, monkeypatch):
    from app.pipeline import lift3d
    from app.pipeline import orchestrator as O
    if not lift3d.available():
        pytest.skip("PyTorch not installed: no 3D lifting")

    def boom(*a, **k):
        raise RuntimeError("simulated bug")
    monkeypatch.setattr(O, "side_metrics", boom)
    res = O.recompute(analysed["id"], {"bowler_name": "Assist", "bowler_height_cm": 180}, [], render=False)
    by = {m["key"]: m for m in res["metrics"]}
    assert by["trunk_forward_flexion_release"]["status"] != "not_measured" and by["trunk_forward_flexion_release"]["view"] == "3d"
    assert by["stride_length_pct_height"]["status"] == "not_measured"  # a distance: no 3D source

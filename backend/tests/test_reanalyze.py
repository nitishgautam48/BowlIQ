"""Key moments corrected by hand: the delivery is re-analysed with exactly those frames."""
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


def _post(client, aid, body):
    return client.post(f"{API}/analysis/{aid}/reanalyze", content=json.dumps(body),
                       headers={**headers(), "Content-Type": "application/json"})


def test_reanalyze_with_hand_set_moments(client):
    job = poll(client, upload(client, bowler_name="Fixer", bowler_height_cm="180").json()["job_id"])
    assert job["status"] == "done"
    old = job["result"]
    ev = {k: old["events"][k]["frame"] for k in ("bfc", "ffc", "release")}
    new_ev = {"bfc": ev["bfc"] - 3, "ffc": ev["ffc"] + 2, "release": ev["release"] + 1}
    r = _post(client, old["id"], new_ev)
    assert r.status_code == 200, r.text
    j = poll(client, r.json()["job_id"])
    assert j["status"] == "done", j
    res = j["result"]
    assert {k: res["events"][k]["frame"] for k in new_ev} == new_ev
    assert res["events"]["ffc"]["confidence_reason"] == "Set by hand on the video."
    assert res["video"]["frame_count"] == old["video"]["frame_count"]  # same frame numbering as the original
    assert res["bowling_arm"] == old["bowling_arm"] and res["bowler_name"] == "Fixer"
    assert any("set by hand" in w for w in res["warnings"])
    assert res["id"] != old["id"]


def test_reanalyze_rejects_impossible_moments(client):
    job = poll(client, upload(client, bowler_name="Fixer2").json()["job_id"])
    aid = job["result"]["id"]
    assert _post(client, aid, {"bfc": 50, "ffc": 40, "release": 60}).json()["code"] == "INVALID_EVENTS"
    assert _post(client, aid, {"bfc": 0, "ffc": 1, "release": 140}).json()["code"] == "INVALID_EVENTS"  # far apart
    assert _post(client, "nope00", {"bfc": 1, "ffc": 20, "release": 30}).status_code == 404


def _slow_sample(tmp_path):
    import subprocess

    from app.core.ffmpeg import ffmpeg_path
    from tests.apiclient import SAMPLE

    slow = tmp_path / "slow.mp4"  # 120 fps sample played 4× slower at 30 fps: same frames, each 4× longer
    subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-i", str(SAMPLE), "-vf", "setpts=4*PTS", "-r", "30",
                    "-an", "-c:v", "libx264", "-crf", "18", str(slow)], check=True)
    return slow


def _upload_file(client, path, **data):
    with open(path, "rb") as f:
        return client.post(f"{API}/analyze", files={"side_video": (path.name, f, "video/mp4")}, data=data,
                           headers=headers())


def test_slow_motion_clip_is_never_rejected(client, tmp_path):
    """Not sure about slow motion: analysed anyway as slow motion, speeds/times withheld. With ×4 given: the key
    moments match the original sample's (70 / 92 / 104)."""
    slow = _slow_sample(tmp_path)
    j = poll(client, _upload_file(client, slow, bowler_name="SlowMo").json()["job_id"])
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    assert any("slow-motion" in w for w in res["warnings"])
    assert res["ball_speed"]["value_kmh"] is None
    assert next(m for m in res["metrics"] if m["key"] == "runup_speed")["value"] is None
    j4 = poll(client, _upload_file(client, slow, bowler_name="SlowMo", slow_motion="4").json()["job_id"])
    ev = {k: j4["result"]["events"][k]["frame"] for k in ("bfc", "ffc", "release")}
    assert abs(ev["bfc"] - 70) <= 2 and abs(ev["ffc"] - 92) <= 2 and abs(ev["release"] - 104) <= 2, ev


def test_failed_job_can_be_finished_by_marking(client, tmp_path):
    """Any failure the user can fix by marking (kept 24 h) finishes with exactly the marked frames."""
    import shutil

    from app.core.storage import AnalysisFiles
    from app.pipeline.preprocess import preprocess
    from app.pipeline.quality import probe_video
    from tests.apiclient import SAMPLE

    jid = "markfail0000001"
    files = AnalysisFiles(jid).ensure()
    preprocess(SAMPLE, files.side_prep, probe_video(SAMPLE), None, None)
    from app.models.db import Job, session
    with session() as s:  # inserted already failed, so the background worker never picks it up
        s.add(Job(id=jid, owner_id="tester", status="failed", error_code="EVENTS_NOT_FOUND", error_message="x",
                  error_fix="y", params_json=json.dumps({"bowler_name": "Marked"})))
        s.commit()
    st = client.get(f"{API}/jobs/{jid}", headers=headers()).json()
    assert st["can_mark"] and st["clip"]["frame_count"] > 100
    q = client.get(f"{API}/jobs/{jid}/links", headers=headers()).json()["query"]
    assert client.get(f"{API}/jobs/{jid}/video?{q}", headers={"Range": "bytes=0-99"}).status_code == 206
    h = {**headers(), "Content-Type": "application/json"}
    assert client.post(f"{API}/jobs/{jid}/mark", content=json.dumps({"bfc": 90, "ffc": 80, "release": 104}),
                       headers=h).json()["code"] == "INVALID_EVENTS"
    m = client.post(f"{API}/jobs/{jid}/mark", content=json.dumps({"bfc": 70, "ffc": 92, "release": 104}), headers=h)
    done = poll(client, m.json()["job_id"])
    assert done["status"] == "done", done
    assert {k: done["result"]["events"][k]["frame"] for k in ("bfc", "ffc", "release")} == {"bfc": 70, "ffc": 92,
                                                                                            "release": 104}
    shutil.rmtree(files.dir, ignore_errors=True)

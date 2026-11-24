"""Upload the sample clip, poll to done, then fetch every output through the authenticated v1 API."""
from urllib.parse import parse_qs

import pytest
from fastapi.testclient import TestClient

from app.pipeline.metrics import FRONT_KEYS, SIDE_KEYS
from tests.apiclient import API, OTHER, ensure_sample, headers, poll, upload


@pytest.fixture(scope="module")
def client():
    ensure_sample()
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_is_public_and_has_no_paths(client):
    h = client.get(f"{API}/health").json()
    assert h["ffmpeg"] is True and h["model_file"] is True and h["database"] is True
    assert h["pose_device"] in ("gpu", "cpu")
    assert not any("/" in v or "\\" in v for v in h["details"].values())


def test_rejects_bad_input(client):
    r = client.post(f"{API}/analyze", files={"side_video": ("clip.avi", b"xxxx", "video/x-msvideo")}, headers=headers())
    assert r.status_code == 415 and set(r.json()) == {"code", "message", "fix"}
    r = client.post(f"{API}/analyze", files={"side_video": ("clip.mp4", b"not a video at all", "video/mp4")}, headers=headers())
    assert r.json()["code"] == "UNREADABLE_VIDEO"
    assert upload(client, bowler_height_cm="300").json()["code"] == "INVALID_HEIGHT"
    assert upload(client, trim_start_s="0.2", trim_end_s="0.7").json()["code"] == "CLIP_TOO_SHORT"
    assert client.get(f"{API}/analysis/nope", headers=headers()).json()["code"] == "NOT_FOUND"


def test_full_flow(client):
    r = upload(client, bowler_name="E2E", bowler_height_cm="180")
    assert r.status_code == 200, r.text
    job = poll(client, r.json()["job_id"])
    assert job["status"] == "done", f"the sample clip must analyse cleanly: {job['error']}"
    res = job["result"]
    aid = res["id"]
    assert len(res["metrics"]) == len(SIDE_KEYS) + len(FRONT_KEYS) == 24
    for m in res["metrics"]:
        assert m["status"] in ("in_range", "outside", "not_measured", "info")
        if m["status"] == "not_measured":
            assert m["value"] is None and m["reason"]
        else:
            assert m["confidence"] in ("High", "Medium", "Low") and m["confidence_reason"]
    ev = res["events"]
    assert ev["bfc"]["frame"] < ev["ffc"]["frame"] < ev["release"]["frame"]
    assert 0.0 < res["scale"]["rel_uncertainty"] < 0.5
    assert res["report"]["generated_by"] == "template"
    # ball speed: truth ~122 km/h; the ± must cover it now that scale uncertainty is included
    bs = res["ball_speed"]
    if bs["value_kmh"] is not None:
        assert abs(bs["value_kmh"] - 121.6) <= bs["plus_minus"] + 1.0

    # media needs a signed link (no Authorization header on <video>)
    assert client.get(f"{API}/analysis/{aid}/video").status_code == 401
    q = client.get(f"{API}/analysis/{aid}/links", headers=headers()).json()["query"]
    v = client.get(f"{API}/analysis/{aid}/video?{q}")
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4" and len(v.content) > 1000
    part = client.get(f"{API}/analysis/{aid}/video?{q}", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    bad = dict(parse_qs(q))
    tampered = f"exp={bad['exp'][0]}&sig={'0' * 64}&u={bad['u'][0]}"
    assert client.get(f"{API}/analysis/{aid}/video?{tampered}").status_code == 401
    kp = client.get(f"{API}/analysis/{aid}/keypoints", headers=headers()).json()
    assert len(kp["joints"]) == 21 and kp["joints"][17:] == ["left_heel", "right_heel", "left_foot_index", "right_foot_index"]
    assert len(kp["frames"]) == res["video"]["frame_count"]
    fr = client.get(f"{API}/analysis/{aid}/frame/{ev['ffc']['frame']}?{q}")
    assert fr.status_code == 200 and fr.content[:2] == b"\xff\xd8"
    assert client.get(f"{API}/analysis/{aid}/frame/99999?{q}").json()["code"] == "FRAME_OUT_OF_RANGE"
    pdf = client.get(f"{API}/analysis/{aid}/pdf?{q}")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    # owner isolation: another user can't see, list, fetch media or delete it
    assert client.get(f"{API}/analysis/{aid}", headers=headers(OTHER)).status_code == 404
    assert all(d["id"] != aid for d in client.get(f"{API}/deliveries", headers=headers(OTHER)).json())
    assert client.delete(f"{API}/analysis/{aid}", headers=headers(OTHER)).status_code == 404

    assert any(b["name"] == "E2E" for b in client.get(f"{API}/bowlers", headers=headers()).json())
    hist = client.get(f"{API}/bowlers/E2E/history", headers=headers()).json()
    assert len(hist["deliveries"]) == 1 and hist["deliveries_until_baseline"] == 9
    pt = hist["deliveries"][0]["metrics"]["runup_speed"]
    assert set(pt) == {"value", "unit"} and pt["unit"] == "m/s"
    cmp = client.get(f"{API}/compare?a={aid}&b={aid}", headers=headers()).json()
    assert all(d["verdict"] in ("same", "not_comparable", "neutral") for d in cmp["diffs"])
    if res["findings"]:
        assert client.post(f"{API}/analysis/{aid}/feedback", json={"finding_rank": 1, "helpful": True}, headers=headers()).json() == {"ok": True}
        client.post(f"{API}/analysis/{aid}/feedback", json={"finding_rank": 1, "helpful": False}, headers=headers())
        assert client.get(f"{API}/analysis/{aid}", headers=headers()).json()["feedback"] == {"1": False}
    assert client.delete(f"{API}/analysis/{aid}", headers=headers()).json() == {"deleted": aid}
    assert client.get(f"{API}/analysis/{aid}", headers=headers()).status_code == 404


def test_sample_bowler_never_gets_a_baseline(client):
    from app.core.config import settings
    from app.core.jobs import history

    assert history("tester", settings.sample_bowler, "x") == []

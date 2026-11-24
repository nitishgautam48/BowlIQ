"""Regression tests for the security/robustness fixes from AUDIT_REPORT.md."""
import json
from datetime import timedelta

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.api.routes import _range_response, parse_float, verdict
from app.core import jobs
from app.core.errors import BowlIQError
from app.core.security import RateLimiter, sign_media, verify_media
from app.models.db import Job, session, utcnow
from app.pipeline.quality import metadata_checks, resolve_trim
from app.pipeline.types import VideoMeta
from tests.apiclient import API, ensure_sample, headers, upload


@pytest.fixture(scope="module")
def client():
    ensure_sample()
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_auth_required(client):
    assert client.get(f"{API}/deliveries").status_code == 401
    assert client.get(f"{API}/deliveries", headers={"Authorization": "Bearer wrong-token-0000000000"}).status_code == 401
    assert client.get(f"{API}/me", headers=headers()).json() == {"owner": "tester"}


def test_untrusted_host_rejected(client):  # DNS rebinding
    assert client.get(f"{API}/health", headers={"Host": "attacker.example"}).status_code == 400


def test_cross_origin_and_missing_client_header_blocked(client):  # CSRF / drive-by POST
    h = {**headers(), "Origin": "https://evil.example"}
    assert client.post(f"{API}/analyze", headers=h).json()["code"] == "FORBIDDEN_ORIGIN"
    h = {"Authorization": headers()["Authorization"]}
    assert client.delete(f"{API}/analysis/x", headers=h).json()["code"] == "MISSING_CLIENT_HEADER"


def test_oversize_body_rejected_before_reading(client):
    h = {**headers(), "Content-Length": str(500 * 1024 * 1024), "Content-Type": "multipart/form-data; boundary=x"}
    r = client.post(f"{API}/analyze", headers=h, content=b"--x--")
    assert r.status_code == 413 and r.json()["code"] == "FILE_TOO_LARGE"


def test_security_headers(client):
    r = client.get(f"{API}/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers.get("x-request-id")


def test_nan_and_inf_rejected(client):
    for v in ("nan", "inf", "-inf", "NaN"):
        with pytest.raises(BowlIQError):
            parse_float(v, "x")
        assert upload(client, trim_start_s=v).json()["code"] == "INVALID_REQUEST"
    with pytest.raises(BowlIQError):
        resolve_trim(3.0, float("nan"), None)


def test_rate_limiter():
    rl = RateLimiter()
    for _ in range(3):
        rl.check("k", 3)
    with pytest.raises(BowlIQError) as e:
        rl.check("k", 3)
    assert e.value.http_status == 429


def test_signed_media_links():
    s = sign_media("abc123", "tester", ttl_s=60)
    assert verify_media("abc123", "tester", s["exp"], s["sig"])
    assert not verify_media("abc124", "tester", s["exp"], s["sig"])
    assert not verify_media("abc123", "other", s["exp"], s["sig"])
    expired = sign_media("abc123", "tester", ttl_s=-10)
    assert not verify_media("abc123", "tester", expired["exp"], expired["sig"])


def _req(h: str) -> Request:
    return Request({"type": "http", "headers": [(b"range", h.encode())], "method": "GET", "path": "/"})


def test_range_edge_cases(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"x" * 1000)
    assert _range_response(p, _req("bytes=-0"), "video/mp4").status_code == 416
    assert _range_response(p, _req("bytes=5000-"), "video/mp4").status_code == 416
    assert _range_response(p, _req("bytes=abc"), "video/mp4").status_code == 416
    r = _range_response(p, _req("bytes=990-"), "video/mp4")
    assert r.status_code == 206 and r.headers["content-length"] == "10"
    r = _range_response(p, _req("bytes=-10"), "video/mp4")
    assert r.status_code == 206 and r.headers["content-range"] == "bytes 990-999/1000"


def test_interrupted_only_when_lease_expired():
    with session() as s:
        s.add(Job(id="stuck1", owner_id="tester", status="processing"))
        s.add(Job(id="alive1", owner_id="tester", status="processing", lease_until=utcnow() + timedelta(minutes=5)))
        s.add(Job(id="queued1", owner_id="tester", status="queued"))
        s.commit()
    jobs.recover_interrupted()
    assert jobs.get_job("stuck1").error_code == "INTERRUPTED"
    assert jobs.get_job("alive1").status == "processing"  # another worker still holds it
    assert jobs.get_job("queued1").status == "queued"      # queued jobs survive restarts
    with session() as s:
        for i in ("stuck1", "alive1", "queued1"):
            s.delete(s.get(Job, i))
        s.commit()


def test_claim_is_exclusive():
    with session() as s:
        s.add(Job(id="claimme", owner_id="tester", status="queued", created_at=utcnow() - timedelta(days=365)))
        s.commit()
    first = jobs.claim_next("w1")
    assert first == "claimme"
    assert jobs.claim_next("w2") != "claimme"
    with session() as s:
        s.delete(s.get(Job, "claimme"))
        s.commit()


def test_params_persist_as_strict_json():
    jobs.create_job("strictjson", "tester", {"a": 1.0})
    assert json.loads(jobs.get_job("strictjson").params_json) == {"a": 1.0}
    with pytest.raises(ValueError):
        jobs.create_job("badjson", "tester", {"a": float("nan")})
    with session() as s:
        s.delete(s.get(Job, "strictjson"))
        s.commit()


def test_compare_verdict_directions():
    assert verdict("stride_length_pct_height", 80, 70, "% height", "% height")[0] == "better"
    assert verdict("shoulder_counter_rotation", 35, 20, "°", "°")[0] == "worse"
    assert verdict("runup_speed", 6.0, 5.0, "m/s", "% height/s")[0] == "not_comparable"
    assert verdict("release_height_pct_height", 110, 100, "% height", "% height")[0] == "neutral"


def test_quality_low_fps_and_vfr_warn_but_continue():
    meta = VideoMeta("x", 30.0, 1920, 1080, 120, 4.0, vfr=True)
    checks, _reasons, level = metadata_checks(meta, 4.0)
    assert level == "Low"
    assert any(c["label"] == "Frame timing" for c in checks)


def test_non_mp4_bytes_rejected_by_signature(tmp_path):
    from app.core.ffmpeg import looks_like_mp4_mov

    p = tmp_path / "x.mp4"
    p.write_bytes(b"#EXTM3U\nhttp://127.0.0.1/x.ts\n")
    assert not looks_like_mp4_mov(p)


def test_path_like_ids_refused():
    from app.core.storage import AnalysisFiles

    with pytest.raises(BowlIQError):
        AnalysisFiles("../etc")

"""Shared authenticated TestClient for API tests."""
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import OTHER, TOKEN

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "sample_delivery.mp4"
API = "/api/v1"


def headers(token: str = TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}", "X-BowlIQ-Client": "web"}


def ensure_sample() -> None:
    if not SAMPLE.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample.py")], check=True)


def upload(client: TestClient, token: str = TOKEN, **data):
    with open(SAMPLE, "rb") as f:
        return client.post(f"{API}/analyze", files={"side_video": ("sample.mp4", f, "video/mp4")}, data=data,
                           headers=headers(token))


def poll(client: TestClient, job_id: str, token: str = TOKEN, timeout: float = 600) -> dict:
    t0, last = time.time(), -1
    while time.time() - t0 < timeout:
        r = client.get(f"{API}/jobs/{job_id}", headers=headers(token))
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["progress"] >= last or j["status"] in ("done", "failed")
        last = j["progress"]
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError("job did not finish")


__all__ = ["API", "OTHER", "SAMPLE", "TOKEN", "ensure_sample", "headers", "poll", "upload"]

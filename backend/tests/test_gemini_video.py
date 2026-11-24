"""Optional Gemini video check: its hints are merged with the pipeline's own evidence, never trusted blindly, and any
failure leaves the analysis exactly as without it. No network: Gemini is faked."""
import json

import pytest
from fastapi.testclient import TestClient

from app.pipeline import gemini_video as GV
from app.pipeline.clip_context import Detection
from tests.apiclient import ensure_sample, poll, upload


def _brief(**kw) -> GV.ClipBrief:
    base = dict(is_bowling_delivery=True, release_time_s=None, view="side-on", view_confidence="high", bowling_arm="right",
                bowling_arm_confidence="high", run_direction="left_to_right", run_direction_confidence="high",
                slow_motion="real_time", slow_motion_confidence="high", people_in_frame=1)
    base.update(kw)
    return GV.ClipBrief(**base)


# ───────────────────────────── merge rules ─────────────────────────────

def test_agreement_raises_confidence():
    d, out = GV.combine(Detection("right", "Low", ["x"]), "right", "medium", "bowling arm")
    assert (d.value, d.confidence, out) == ("right", "Medium", "agree")


def test_confident_gemini_replaces_an_unsure_measurement():
    d, out = GV.combine(Detection("left", "Low", ["x"]), "right", "high", "bowling arm")
    assert (d.value, d.confidence, out) == ("right", "Medium", "gemini") and d.details["pipeline_value"] == "left"


def test_disagreement_with_a_sure_measurement_is_flagged_not_followed():
    d, out = GV.combine(Detection("side-on", "High", ["x"]), "behind", "high", "camera view")
    assert (d.value, d.confidence, out) == ("side-on", "Low", "conflict")
    d, out = GV.combine(Detection("left", "Medium", ["x"]), "right", "medium", "bowling arm")
    assert (d.value, d.confidence, out) == ("left", "Low", "conflict")


def test_user_corrections_and_unclear_answers_are_left_alone():
    u = Detection("left", "High", ["Set by hand."], "user")
    assert GV.combine(u, "right", "high", "bowling arm") == (u, "none")
    d = Detection("right", "Medium", ["x"])
    assert GV.combine(d, "unclear", "low", "bowling arm") == (d, "none")


def test_off_without_switch_or_key(monkeypatch, tmp_path):
    import dataclasses

    from app.core.config import settings
    monkeypatch.setattr(GV, "settings", dataclasses.replace(settings, gemini_video=False))
    assert GV.brief(tmp_path / "x.mp4", 3.0) == (None, "Gemini video check is off.")


def test_answer_parsed_and_box_normalised(monkeypatch, tmp_path):
    """A model answering on a 0–1000 grid; the request carries the clip with a frame-rate hint."""
    import dataclasses

    from app.core.config import settings
    from google import genai

    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"\x00" * 100)
    seen = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            seen["fps"] = contents[0].video_metadata.fps
            ans = _brief(bowler=GV.BowlerBox(time_s=2.0, x0=100, y0=200, x1=300, y1=900)).model_dump()

            class R:
                text = json.dumps(ans)
            return R()

    class FakeClient:
        def __init__(self, **kw):
            self.models = FakeModels()

    monkeypatch.setattr(GV, "settings", dataclasses.replace(settings, gemini_video=True, gemini_api_key="test-key"))
    monkeypatch.setattr(genai, "Client", FakeClient)
    b, err = GV.brief(clip, 5.0)
    assert err is None and b.bowler.x0 == pytest.approx(0.1) and b.bowler.y1 == pytest.approx(0.9)
    assert 1.0 <= seen["fps"] <= 8.0


def test_failure_is_quiet(monkeypatch, tmp_path):
    import dataclasses

    from app.core.config import settings
    from google import genai

    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"\x00" * 100)

    class Boom:
        def __init__(self, **kw):
            raise ConnectionError("no network")

    monkeypatch.setattr(GV, "settings", dataclasses.replace(settings, gemini_video=True, gemini_api_key="test-key"))
    monkeypatch.setattr(genai, "Client", Boom)
    b, err = GV.brief(clip, 5.0)
    assert b is None and "ConnectionError" in err


# ───────────────────────────── inside the pipeline ─────────────────────────────

@pytest.fixture(scope="module")
def analysed():
    ensure_sample()
    from app.main import app

    with TestClient(app) as c:
        job = poll(c, upload(c, bowler_name="Gem", bowler_height_cm="180").json()["job_id"])
    assert job["status"] == "done", job
    return job["result"]


def _with_brief(aid: str, brief):
    from app.core.storage import AnalysisFiles
    from app.pipeline import orchestrator as O

    files = AnalysisFiles(aid)
    st = O.load_cache(files)
    st.gemini, st.gemini_error = brief, None
    O.save_cache(files, st)
    return O.recompute(aid, {"bowler_name": "Gem", "bowler_height_cm": 180}, [], render=False)


def test_agreeing_brief_confirms_and_is_reported(analysed):
    res = _with_brief(analysed["id"], _brief(release_time_s=analysed["events"]["release"]["time_s"]))
    g = res["gemini"]
    assert g["used"] and g["outcomes"]["bowling_arm"] == "agree" and g["outcomes"]["view"] == "agree"
    assert g["outcomes"]["release"] == "agree"
    assert res["context"]["bowling_arm"]["confidence"] in ("Medium", "High")
    assert any("Gemini" in r for r in res["context"]["bowling_arm"]["reasons"])


def test_wrong_hint_is_overruled_by_the_measured_release(analysed):
    """Gemini says left arm (wrong): the release check finds the clear release with the right arm and keeps it."""
    res = _with_brief(analysed["id"], _brief(bowling_arm="left"))
    assert res["bowling_arm"] == "right"
    assert res["events"]["release"]["frame"] == analysed["events"]["release"]["frame"]


def test_release_far_from_geminis_is_flagged(analysed):
    t = analysed["events"]["release"]["time_s"]
    res = _with_brief(analysed["id"], _brief(release_time_s=t + 3.0))
    assert res["needs_review"] and any("Gemini" in r and "Release" in r for r in res["review_reasons"])
    _with_brief(analysed["id"], None)  # leave the cache as it was

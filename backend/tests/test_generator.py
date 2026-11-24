from app.report import generator as G
from app.report.generator import LLMReport, build_payload, generate_report, invented_numbers, validate_llm

FINDINGS = [{
    "rank": 1, "title": "Front knee bends after landing", "metric": "front_knee_flexion_ffc_to_release",
    "measured": 17.4, "unit": "°", "target_text": "Under 10°", "target_min": None, "target_max": 10.0, "frame": 66,
    "view": "side-on", "confidence": "High", "confidence_reason": "ok", "source": "Portus", "kind": "pace", "severity": 1.0,
}]
QUALITY = {"level": "High", "reasons": ["120 fps, steady camera."], "fps": 120.0}
METRICS = [{"key": "front_knee_flexion_ffc_to_release", "name": "Front knee flexion, FFC to release"}]


def payload():
    return build_payload(FINDINGS, [], [], QUALITY, {"value_kmh": None, "plus_minus": None, "reason": "no height"}, METRICS, False)


def notes():
    return G.coaching_notes()["front_knee_flexion_ffc_to_release"]


def good_report(**kw) -> LLMReport:
    base = dict(headline_insight="Your front knee bends 17° after landing (side-on, high quality clip).",
                headline_sub="Measured at frame 66, target under 10°.",
                findings=[{"rank": 1, "why": "It absorbs energy.", "cue": notes()["cue"][0], "drill": notes()["drill"][0]}],
                strengths_text="Keep working.")
    base.update(kw)
    return LLMReport.model_validate(base)


def test_valid_report_passes():
    assert validate_llm(good_report(), payload()) == []


def test_invented_numbers_rejected():
    rep = good_report(headline_insight="You could gain 8 km/h by bracing (side-on, 120 fps).")
    probs = validate_llm(rep, payload())
    assert any("invented" in p for p in probs)
    assert invented_numbers("17.4° and 17°", payload()) == []
    assert invented_numbers("18°", payload()) == ["18"]


def test_cue_must_come_from_notes():
    rep = good_report(findings=[{"rank": 1, "why": "x", "cue": "Jump higher like a kangaroo", "drill": notes()["drill"][0]}])
    assert any("cue" in p for p in validate_llm(rep, payload()))


def test_template_without_api_key():
    assert G.settings.gemini_api_key == ""  # conftest blanks the key, so no network call happens
    out = generate_report(payload(), 120.0, False)
    assert out["generated_by"] == "template"
    assert "front knee" in out["headline_insight"].lower()
    assert out["findings"][0]["cue"] in notes()["cue"]
    assert "front-on" in out["limits_note"]


def test_invalid_llm_output_falls_back(monkeypatch):
    monkeypatch.setattr(G, "call_gemini", lambda p: good_report(headline_sub="Elite bowlers bend 3°."))
    out = generate_report(payload(), 120.0, False)
    assert out["generated_by"] == "template"
    monkeypatch.setattr(G, "call_gemini", lambda p: good_report())
    assert generate_report(payload(), 120.0, False)["generated_by"] == "gemini"

from app.analysis.baseline import Baseline, compute_baseline
from app.analysis.rules import BASELINE_PENDING, INJURY_TEXT, judge
from app.pipeline.metrics import Measurement


def M(key, value, unit, conf="High", view="side-on"):
    return Measurement(key=key, value=value, unit=unit, view=view, frame=10, event="FFC", confidence=conf,
                       confidence_reason="ok", measured=True, joint="left_knee")


NO_BASE = Baseline(False, 3, 7)


def test_ranking_and_injury_flag():
    j = judge([
        M("shoulder_counter_rotation", 35.0, "°", "Medium", "front-on"),
        M("front_knee_flexion_ffc_to_release", 20.0, "°"),
        M("stride_length_pct_height", 70.0, "% height"),
        M("hip_shoulder_separation_bfc", 10.0, "°", "Medium", "front-on"),
    ], NO_BASE)
    assert [f["metric"] for f in j.findings] == ["front_knee_flexion_ffc_to_release", "stride_length_pct_height",
                                                  "shoulder_counter_rotation"]
    assert [f["rank"] for f in j.findings] == [1, 2, 3]
    assert j.headline_metric == "front_knee_flexion_ffc_to_release"
    assert len(j.injury_flags) == 1
    flag = j.injury_flags[0]
    assert flag["metric"] == "shoulder_counter_rotation" and flag["text"] == INJURY_TEXT
    status = {m["key"]: m["status"] for m in j.metrics}
    assert status["hip_shoulder_separation_bfc"] == "in_range"
    assert status["stride_length_pct_height"] == "outside"
    f2 = j.findings[1]
    assert (f2["target_min"], f2["target_max"]) == (75.0, 85.0)
    assert f2["source"].startswith("Spratford")


def test_fewer_findings_not_padded_and_strengths():
    j = judge([M("stride_length_pct_height", 80.0, "% height"), M("front_knee_flexion_ffc_to_release", 4.0, "°")], NO_BASE)
    assert j.findings == []
    assert j.headline_metric is None
    assert {s["metric"] for s in j.strengths} == {"stride_length_pct_height", "front_knee_flexion_ffc_to_release"}


def test_higher_better_before_and_after_baseline():
    j = judge([M("runup_speed", 5.0, "m/s")], NO_BASE)
    m = j.metrics[0]
    assert m["status"] == "info" and m["reason"] == BASELINE_PENDING
    hist = [{"ball_speed_kmh": 110 + i, "metrics": {"runup_speed": {"value": 6.0 + 0.1 * (i % 3), "unit": "m/s"}}}
            for i in range(12)]
    base = compute_baseline(hist)
    assert base.active and "fastest" in base.based_on
    j2 = judge([M("runup_speed", 5.0, "m/s")], base)
    assert j2.metrics[0]["status"] == "outside"
    assert j2.findings[0]["metric"] == "runup_speed"
    j3 = judge([M("runup_speed", 6.1, "m/s")], base)
    assert j3.metrics[0]["status"] == "in_range"


def test_low_confidence_reduces_severity():
    hi = judge([M("front_knee_flexion_ffc_to_release", 20.0, "°", "High")], NO_BASE).findings[0]["severity"]
    lo = judge([M("front_knee_flexion_ffc_to_release", 20.0, "°", "Low")], NO_BASE).findings[0]["severity"]
    assert lo == round(hi * 0.4, 4)


def test_baseline_unlock_counts():
    assert compute_baseline([{}] * 9).deliveries_until_baseline == 1
    recent = compute_baseline([{"ball_speed_kmh": None, "metrics": {"runup_speed": {"value": 6.0, "unit": "m/s"}}}] * 10)
    assert recent.active and "most recent 10" in recent.based_on

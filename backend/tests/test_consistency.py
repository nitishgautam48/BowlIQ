"""Phase 5: verdicts across several deliveries (consistent vs occasional vs one-off)."""
import json

import pytest
from fastapi.testclient import TestClient

from app.analysis.consistency import summarise
from app.models.db import Analysis, session
from app.report.generator import coaching_notes
from tests.apiclient import API, ensure_sample, headers


def _m(key, value, status, conf="High", unit="°", frame=100, view="side-on"):
    return {"key": key, "value": value, "status": status, "confidence": conf, "unit": unit, "frame": frame,
            "view": view, "target_text": "Under 10°", "target_max": 10.0, "target_min": None}


def _res(i, knee_flex, scr=None, scr_conf="Medium", head=5.0, speed=None):
    ms = [_m("front_knee_flexion_ffc_to_release", knee_flex, "outside" if knee_flex >= 10 else "in_range"),
          _m("head_ahead_of_front_foot_release", head, "info", unit="% height")]
    if scr is not None:
        ms.append(_m("shoulder_counter_rotation", scr, "outside" if scr >= 30 else "in_range", conf=scr_conf, view="3d"))
    return {"id": f"d{i}", "metrics": ms, "ball_speed": {"value_kmh": speed}}


def test_consistent_issue_vs_one_off():
    # newest first: knee bends in 7 of 8 deliveries; counter-rotation flagged once in 6
    res = [_res(i, 18 if i != 3 else 6, scr=35 if i == 0 else 20, speed=120 + i) for i in range(8)]
    for r in res[6:]:
        r["metrics"] = r["metrics"][:2]  # counter-rotation not measured in two of them
    s = summarise(res, coaching_notes())
    by = {i["metric"]: i for i in s["issues"]}
    knee = by["front_knee_flexion_ffc_to_release"]
    assert knee["pattern"] == "consistent" and (knee["count"], knee["measured"]) == (7, 8)
    assert knee["drill"] and knee["cue"]  # from the coaching notes
    scr = by["shoulder_counter_rotation"]
    assert scr["pattern"] == "one-off" and (scr["count"], scr["measured"]) == (1, 6) and scr["kind"] == "injury"
    assert scr["example"]["id"] == "d0" and scr["example"]["view"] == "3d"
    assert s["issues"][0]["metric"] == "front_knee_flexion_ffc_to_release"  # consistent ranks first
    assert "7 of 8" in s["summary"] and not s["provisional"]
    assert s["ball_speed"] == {"median": 123.5, "best": 127.0, "measured": 8}


def test_low_confidence_flags_dont_make_a_pattern():
    res = [_res(i, 4, scr=40, scr_conf="Low") for i in range(6)]
    s = summarise(res, coaching_notes())
    assert all(i["metric"] != "shoulder_counter_rotation" for i in s["issues"])
    assert any(st["metric"] == "front_knee_flexion_ffc_to_release" for st in s["strengths"])


def test_few_deliveries_are_provisional_and_need_three_measurements():
    s = summarise([_res(0, 20), _res(1, 20)], coaching_notes())
    assert s["provisional"] and s["issues"] == []
    s3 = summarise([_res(i, 20) for i in range(3)], coaching_notes())
    assert s3["provisional"] and s3["issues"][0]["pattern"] == "consistent" and "early signs" in s3["summary"]


def test_monitored_metrics_become_trends():
    res = [_res(i, 4, head=float(h)) for i, h in enumerate([12, 11, 12, 13, 4, 5, 4, 3])]  # newest first: went up
    t = {x["metric"]: x for x in summarise(res, coaching_notes())["trends"]}["head_ahead_of_front_foot_release"]
    assert t["direction"] == "up" and t["change"] > 5 and t["measured"] == 8


@pytest.fixture(scope="module")
def client():
    ensure_sample()
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_summary_endpoint_is_per_owner(client):
    with session() as s:
        for i in range(4):
            s.add(Analysis(id=f"sumrow{i}", owner_id="tester", bowler_name="Summary Bowler",
                           result_json=json.dumps(_res(i, 15))))
        s.commit()
    r = client.get(f"{API}/bowlers/Summary%20Bowler/summary?last=10", headers=headers()).json()
    assert r["deliveries"] == 4 and r["issues"][0]["count"] == 4 and r["issues"][0]["pattern"] == "consistent"
    other = client.get(f"{API}/bowlers/Summary%20Bowler/summary", headers=headers("other-token-0123456789abcdef")).json()
    assert other["deliveries"] == 0 and other["issues"] == []

"""Judgement and ranking. severity = normalised deviation × pace_weight (or injury_weight for flags) × confidence factor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.analysis.baseline import Baseline
from app.analysis.benchmarks import rule, target_text
from app.pipeline.metrics import METHODS, Measurement

CONF_FACTOR = {"High": 1.0, "Medium": 0.7, "Low": 0.4}
BASELINE_PENDING = "Personal baseline unlocks after 10 deliveries"
INJURY_TEXT = "Load pattern worth checking with a coach or physio"
BASE_DEV = 0.25

DISPLAY_DEC = {"°": 0, "% height": 1, "m/s": 1, "% height/s": 0, "%": 0, "ms": 0, "% leg": 1}


def fmt(v: float, unit: str) -> str:
    d = DISPLAY_DEC.get(unit, 1)
    s = f"{v:.{d}f}"
    return f"{s}°" if unit == "°" else f"{s} {unit}"


@dataclass
class Judged:
    metrics: list[dict]
    findings: list[dict]
    strengths: list[dict]
    kinetic_chain: list[dict]
    injury_flags: list[dict]
    headline_metric: Optional[str]


def _title(key: str, value: float, r: dict) -> str:
    if key == "stride_length_pct_height":
        return "Short delivery stride" if value < r["band"][0] else "Over-long delivery stride"
    return {
        "front_knee_flexion_ffc_to_release": "Front knee bends after landing",
        "shoulder_counter_rotation": "Large shoulder counter-rotation",
        "hip_shoulder_separation_bfc": "Large hip–shoulder separation at back foot contact",
        "runup_speed": "Slower run-up than your best deliveries",
        "trunk_forward_flexion_release": "Less trunk flexion at release than your best",
        "front_knee_angle_release": "Front knee more bent at release than your best",
    }.get(key, r["name"])


def judge(meas: list[Measurement], baseline: Baseline) -> Judged:
    metrics: list[dict] = []
    issues: list[dict] = []
    goods: list[dict] = []
    flags: list[dict] = []
    for m in meas:
        r = rule(m.key)
        typ = r["type"]
        base_stat = baseline.stats.get(m.key) if baseline.active else None
        if base_stat is not None and base_stat.unit != m.unit:
            base_stat = None
        d = {
            "key": m.key, "name": r["name"], "value": None if m.value is None else round(m.value, 2), "unit": m.unit,
            "view": m.view, "frame": m.frame, "event": m.event, "confidence": m.confidence,
            "confidence_reason": m.confidence_reason, "status": "not_measured", "reason": m.reason, "direction": typ,
            "target_text": target_text(m.key, base_stat.mean if base_stat else None, m.unit),
            "target_min": None, "target_max": None, "baseline": round(base_stat.mean, 2) if base_stat else None,
            "method": METHODS[m.key]["method"], "source": r.get("source"), "joint": m.joint,
        }
        if typ == "band":
            d["target_min"], d["target_max"] = float(r["band"][0]), float(r["band"][1])
        elif typ == "max":
            d["target_max"] = float(r["threshold"])
        if not m.measured or m.value is None:
            metrics.append(d)
            continue
        v = float(m.value)
        cf = CONF_FACTOR[m.confidence or "Low"]
        dev = 0.0
        kind = "pace"
        weight = 0.0
        if typ == "band":
            lo, hi = r["band"]
            if lo <= v <= hi:
                d["status"] = "in_range"
            else:
                d["status"] = "outside"
                dist = lo - v if v < lo else v - hi
                dev = BASE_DEV + dist / (hi - lo)
                weight = max(r.get("pace_weight", 0), r.get("injury_weight", 0))
        elif typ == "max":
            thr = float(r["threshold"])
            if v >= thr:
                d["status"] = "outside"
                dev = BASE_DEV + (v - thr) / thr
                if r.get("injury_weight"):
                    kind, weight = "injury", r["injury_weight"]
                else:
                    weight = r.get("pace_weight", 0)
            else:
                d["status"] = "in_range"
        elif typ == "higher_better":
            if base_stat is None:
                d["status"] = "info"
                d["reason"] = BASELINE_PENDING if not baseline.active else "Not enough baseline deliveries with this metric yet."
            else:
                floor = base_stat.mean - base_stat.sd
                if v < floor:
                    d["status"] = "outside"
                    dev = BASE_DEV + min(3.0, (floor - v) / max(base_stat.sd, 1e-6))
                    weight = r.get("pace_weight", 0)
                else:
                    d["status"] = "in_range"
        else:
            d["status"] = "info"
        metrics.append(d)
        if d["status"] == "outside" and weight > 0:
            sev = round(dev * weight * cf, 4)
            tmin, tmax = d["target_min"], d["target_max"]
            if typ == "higher_better" and base_stat:
                tmin = round(base_stat.mean - base_stat.sd, 2)
            issues.append({
                "title": _title(m.key, v, r), "metric": m.key, "measured": round(v, 1), "unit": m.unit,
                "target_text": d["target_text"] or "", "target_min": tmin, "target_max": tmax,
                "frame": m.frame, "view": m.view, "confidence": m.confidence,
                "confidence_reason": m.confidence_reason, "source": r.get("source"), "kind": kind, "severity": sev,
            })
            if kind == "injury":
                flags.append({
                    "metric": m.key, "title": _title(m.key, v, r), "measured": round(v, 1), "unit": m.unit,
                    "threshold": float(r["threshold"]), "text": INJURY_TEXT, "source": r.get("source"),
                    "view": m.view, "confidence": m.confidence, "frame": m.frame,
                })
        elif d["status"] == "in_range" and m.confidence in ("High", "Medium"):
            goods.append(d)

    issues.sort(key=lambda x: x["severity"], reverse=True)
    findings = [{**f, "rank": i + 1} for i, f in enumerate(issues[:3])]
    # prefer High confidence, then metrics judged against a research rule (band/max) over baseline comparisons
    goods.sort(key=lambda d: (d["confidence"] == "High", d["direction"] in ("band", "max")), reverse=True)
    strengths = []
    for g in goods[:3]:
        val = fmt(g["value"], g["unit"])
        if g["direction"] == "band":
            detail = f"{g['name']} {val}, within the {g['target_text']} target."
        elif g["direction"] == "max":
            detail = f"{g['name']} {val}, under the {fmt(g['target_max'], g['unit'])} flag."
        else:
            detail = f"{g['name']} {val}, in line with your baseline."
        strengths.append({"title": f"Good {g['name'][0].lower() + g['name'][1:]}", "detail": detail, "metric": g["key"]})
    by_key = {d["key"]: d for d in metrics}
    chain = [_segment("Run-up", [by_key.get("runup_speed")]),
             _segment("Back foot", [by_key.get("shoulder_counter_rotation"), by_key.get("hip_shoulder_separation_bfc")]),
             _segment("Front leg brace", [by_key.get("front_knee_flexion_ffc_to_release")]),
             _segment("Trunk", [by_key.get("trunk_forward_flexion_release")]),
             _segment("Arm/Release", [by_key.get("release_height_pct_height")])]
    return Judged(metrics, findings, strengths, chain, flags, findings[0]["metric"] if findings else None)


def _segment(name: str, ms: list[Optional[dict]]) -> dict:
    ms = [m for m in ms if m is not None]
    measured = [m for m in ms if m["value"] is not None]
    if not measured:
        reason = ms[0]["reason"] if ms else "Not measured"
        text = "Add a front-on clip" if ms and ms[0]["view"] != "side-on" and reason == "Add a front-on clip" else "Not measured"
        return {"segment": name, "status": "unknown", "metric_text": text, "metric": ms[0]["key"] if ms else None}
    worst = next((m for m in measured if m["status"] == "outside"), measured[0])
    status = {"outside": "warn", "in_range": "ok"}.get(worst["status"], "unknown")
    text = fmt(worst["value"], worst["unit"])
    if worst["status"] == "info" and worst["direction"] == "higher_better":
        text += " · baseline pending"
    return {"segment": name, "status": status, "metric_text": text, "metric": worst["key"]}

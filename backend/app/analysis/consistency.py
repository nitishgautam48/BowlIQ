"""Phase 5: verdicts across several deliveries.

One delivery can be a one-off (a slip on landing, a tired ball, a bad camera frame). A coach looks at what keeps
happening. For each metric with a rule, count in how many of the bowler's recent deliveries it was flagged (only
Medium/High confidence measurements count; Low-confidence ones are listed as uncertain), then classify:

    consistent  ≥ 60 % of the deliveries where it was measured   ("front knee bends after landing in 7 of 8 balls")
    occasional  25–60 %
    one-off     < 25 %

Issues are ranked by how consistently they show up, then by the rule's weight. Metrics without research thresholds
are summarised as personal trends (older half vs newer half of the deliveries).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.analysis.benchmarks import benchmarks, rule
from app.analysis.rules import _title, fmt

CONSISTENT = 0.6
OCCASIONAL = 0.25
MIN_MEASURED = 3  # a verdict needs the metric measured (Medium/High) in at least this many deliveries
FIRM = 5  # below this many deliveries the summary says verdicts are provisional


def _pattern(share: float) -> str:
    return "consistent" if share >= CONSISTENT else ("occasional" if share >= OCCASIONAL else "one-off")


def summarise(results: list[dict], notes: dict[str, dict[str, list[str]]]) -> dict:
    """results: full analysis results of one bowler, newest first."""
    n = len(results)
    rules = benchmarks()
    issues, strengths, trends = [], [], []
    for key, r in rules.items():
        rows = []
        for res in results:
            m = next((x for x in res.get("metrics", []) if x["key"] == key), None)
            if m is not None and m.get("value") is not None:
                rows.append((res, m))
        if r["type"] == "monitor":
            t = _trend(key, r, rows)
            if t:
                trends.append(t)
            continue
        solid = [(res, m) for res, m in rows if m.get("confidence") in ("High", "Medium")]
        unsure = [(res, m) for res, m in rows if m.get("confidence") == "Low"]
        judged = [(res, m) for res, m in solid if m["status"] in ("outside", "in_range")]
        if len(judged) < MIN_MEASURED:
            continue
        out = [(res, m) for res, m in judged if m["status"] == "outside"]
        share = len(out) / len(judged)
        units = [m["unit"] for _, m in judged]
        unit = max(set(units), key=units.count)
        vals = [m["value"] for _, m in judged if m["unit"] == unit]
        if out:
            worst_res, worst_m = max(out, key=lambda rm: abs(rm[1]["value"] - (rm[1].get("target_max") or rm[1].get("target_min") or 0)))
            kind = "injury" if r["type"] == "max" and r.get("injury_weight") else "pace"
            weight = max(r.get("pace_weight", 0), r.get("injury_weight", 0))
            note = notes.get(key, {})
            issues.append({
                "metric": key, "title": _title(key, float(np.median([m["value"] for _, m in out])), r),
                "name": r["name"], "pattern": _pattern(share), "count": len(out), "measured": len(judged),
                "uncertain": len(unsure), "share": round(share, 3), "kind": kind,
                "typical": round(float(np.median(vals)), 2), "unit": unit, "target_text": judged[0][1].get("target_text"),
                "cue": (note.get("cue") or [""])[0], "drill": (note.get("drill") or [""])[0],
                "example": {"id": worst_res["id"], "frame": worst_m.get("frame"), "view": worst_m.get("view"),
                            "value": round(float(worst_m["value"]), 2)},
                "_rank": (share >= CONSISTENT, share >= OCCASIONAL, share * weight, len(out)),
            })
        elif share == 0 and len(judged) >= MIN_MEASURED:
            strengths.append({"metric": key, "name": r["name"], "measured": len(judged),
                              "typical": round(float(np.median(vals)), 2), "unit": unit,
                              "text": f"{r['name']} was on target in all {len(judged)} deliveries where it was measured "
                                      f"(typically {fmt(float(np.median(vals)), unit)})."})
    issues.sort(key=lambda i: i["_rank"], reverse=True)
    for i in issues:
        del i["_rank"]
    speeds = [res["ball_speed"]["value_kmh"] for res in results if res.get("ball_speed", {}).get("value_kmh") is not None]
    return {
        "deliveries": n,
        "provisional": n < FIRM,
        "ball_speed": ({"median": round(float(np.median(speeds)), 1), "best": round(float(max(speeds)), 1),
                        "measured": len(speeds)} if speeds else None),
        "issues": issues,
        "strengths": strengths,
        "trends": trends,
        "summary": _text(n, issues, strengths),
    }


def _trend(key: str, r: dict, rows: list[tuple[dict, dict]]) -> Optional[dict]:
    rows = [(res, m) for res, m in rows if m.get("confidence") in ("High", "Medium", "Low")]
    if len(rows) < 4:
        return None
    units = [m["unit"] for _, m in rows]
    unit = max(set(units), key=units.count)
    vals = [m["value"] for _, m in rows if m["unit"] == unit][::-1]  # oldest first
    if len(vals) < 4:
        return None
    half = len(vals) // 2
    old, new = np.array(vals[:half]), np.array(vals[half:])
    spread = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    change = float(np.median(new) - np.median(old))
    direction = "steady" if abs(change) <= max(spread * 0.5, 1e-9) else ("up" if change > 0 else "down")
    return {"metric": key, "name": r["name"], "unit": unit, "measured": len(vals), "typical": round(float(np.median(vals)), 2),
            "spread": round(spread, 2), "change": round(change, 2), "direction": direction}


def _text(n: int, issues: list[dict], strengths: list[dict]) -> str:
    if n == 0:
        return "No deliveries analysed yet."
    parts = [f"Across your last {n} {'delivery' if n == 1 else 'deliveries'}:"]
    cons = [i for i in issues if i["pattern"] == "consistent"]
    occ = [i for i in issues if i["pattern"] == "occasional"]
    if cons:
        parts.append(" ".join(f"{i['title'].lower().capitalize()} in {i['count']} of {i['measured']}." for i in cons[:2]))
        parts.append(f"Work on this first: {cons[0]['drill'] or cons[0]['cue'] or 'see the drill in the report'}")
    elif occ:
        parts.append(f"Nothing shows up consistently. {occ[0]['title']} appeared in {occ[0]['count']} of "
                     f"{occ[0]['measured']}: worth watching, not yet a pattern.")
    else:
        parts.append("No flagged issue repeats across your deliveries.")
    if strengths:
        parts.append(f"Consistently good: {', '.join(s['name'].lower() for s in strengths[:2])}.")
    if n < FIRM:
        parts.append(f"With fewer than {FIRM} deliveries these are early signs, not verdicts.")
    return " ".join(parts)


def rule_names() -> dict[str, str]:
    return {k: rule(k)["name"] for k in benchmarks()}

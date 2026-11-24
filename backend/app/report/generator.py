"""Report text. Gemini Flash explains ONLY the findings JSON it is given; every number it writes must appear in the input.

Cues and drills must come from coaching_notes.md. Any violation, API error or rate limit falls back to the deterministic
template, so a report is always produced.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from app.core.config import settings

log = logging.getLogger("bowliq.report")
NOTES_PATH = Path(__file__).with_name("coaching_notes.md")
NUM_RE = re.compile(r"(?<![A-Za-z])[-−]?\d+(?:\.\d+)?")

SYSTEM_PROMPT = """You are BowlIQ, a calm fast-bowling coach explaining a biomechanics report.
Rules you must follow exactly:
- Use ONLY the numbers present in the input JSON. Never invent, convert, combine or estimate numbers.
- Always name the camera view (side-on, front-on, behind or diagonal) and the clip quality when you mention a measurement.
- For each finding choose the cue and the drill VERBATIM from that finding's allowed_cues and allowed_drills lists.
- The "why" must paraphrase that finding's allowed_why text, in at most two sentences.
- Calm, encouraging coaching voice. Plain English. No medical diagnosis; for load patterns say they are worth checking with a coach or physio.
- Output JSON only, matching the schema."""


class _FindingText(BaseModel):
    rank: int
    why: str
    cue: str
    drill: str


class LLMReport(BaseModel):
    headline_insight: str
    headline_sub: str
    findings: list[_FindingText]
    strengths_text: str


@lru_cache(maxsize=1)
def coaching_notes() -> dict[str, dict[str, list[str]]]:
    notes: dict[str, dict[str, list[str]]] = {}
    cur: Optional[str] = None
    for line in NOTES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("## "):
            cur = line[3:].strip()
            notes[cur] = {"why": [], "cue": [], "drill": []}
        elif cur and ":" in line:
            tag, text = line.split(":", 1)
            tag = tag.strip().lower()
            if tag in ("why", "cue", "drill"):
                notes[cur][tag].append(text.strip())
    return notes


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("−", "-")).strip().lower().rstrip(".")


def _numbers_in(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(x.replace("−", "-")) for x in NUM_RE.findall(obj)]
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    return out


def _matches(token: str, allowed: list[float]) -> bool:
    """`token` equals an allowed number to its own rounding precision. Unsigned tokens may match a negative value
    written as a magnitude ("bends 17°" for -17), but a signed token must match the sign too."""
    signed = token.startswith(("-", "−"))
    val = float(token.replace("−", "-"))
    dec = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10 ** (-dec) + 1e-9
    if signed:
        return any(abs(val - a) <= tol for a in allowed)
    return any(abs(val - abs(a)) <= tol for a in allowed)


def invented_numbers(text: str, source: Any) -> list[str]:
    """Numbers in `text` that don't appear (to rounding tolerance) anywhere in `source`."""
    allowed = _numbers_in(source)
    return [t for t in NUM_RE.findall(text) if not _matches(t, allowed)]


def build_payload(findings: list[dict], strengths: list[dict], injury_flags: list[dict], quality: dict,
                  ball_speed: dict, metrics: list[dict], has_front: bool) -> dict:
    names = {m["key"]: m.get("name", m["key"]) for m in metrics}
    notes = coaching_notes()
    return {
        "clip_quality": {"level": quality["level"], "reasons": quality["reasons"], "fps": round(quality["fps"])},
        "views": ["side-on"] + sorted({m["view"] for m in metrics if m.get("view") != "side-on" and m.get("value") is not None}),
        "ball_speed": {"km_h": ball_speed.get("value_kmh"), "plus_minus": ball_speed.get("plus_minus"),
                       "note": ball_speed.get("reason")},
        "findings": [{
            "rank": f["rank"], "title": f["title"], "metric": names.get(f["metric"], f["metric"]), "metric_key": f["metric"],
            "measured": f["measured"], "unit": f["unit"], "target": f["target_text"], "view": f["view"],
            "confidence": f["confidence"], "frame": f["frame"], "kind": f["kind"],
            "allowed_why": notes.get(f["metric"], {}).get("why", []),
            "allowed_cues": notes.get(f["metric"], {}).get("cue", []),
            "allowed_drills": notes.get(f["metric"], {}).get("drill", []),
        } for f in findings],
        "strengths": [{"title": s["title"], "detail": s["detail"]} for s in strengths],
        # metrics that need a second clip the user didn't upload aren't counted as "couldn't be measured"
        "coverage": {"measured": sum(1 for m in metrics if m.get("value") is not None),
                     "total": sum(1 for m in metrics if m.get("reason") != "Add a front-on clip"),
                     "not_measured_reasons": list(dict.fromkeys(m["reason"] for m in metrics
                                                                if m.get("value") is None and m.get("reason")
                                                                and m["reason"] != "Add a front-on clip"))},
        "load_flags": [{"title": fl["title"], "measured": fl["measured"], "unit": fl["unit"], "threshold": fl["threshold"],
                        "text": fl["text"]} for fl in injury_flags],
    }


def _data_only(payload: dict) -> dict:
    """The payload without coaching-note text: numbers in notes (e.g. "3 sets of 8") don't license new numbers."""
    return {**payload, "findings": [{k: v for k, v in f.items() if not k.startswith("allowed_")} for f in payload["findings"]]}


def validate_llm(rep: LLMReport, payload: dict) -> list[str]:
    """Return a list of problems (empty = valid).

    Cue and drill must be verbatim from the coaching notes (so their numbers are pre-approved). Every number anywhere
    else must appear in the measurement data; a finding's "why" may also reuse numbers from its own allowed_why note.
    """
    problems: list[str] = []
    data = _data_only(payload)
    by_rank = {f["rank"]: f for f in payload["findings"]}
    if sorted(f.rank for f in rep.findings) != sorted(by_rank):
        problems.append("finding ranks don't match the input")
    for f in rep.findings:
        src = by_rank.get(f.rank)
        if not src:
            continue
        if _norm(f.cue) not in {_norm(c) for c in src["allowed_cues"]}:
            problems.append(f"cue for finding {f.rank} is not from the coaching notes")
        if _norm(f.drill) not in {_norm(d) for d in src["allowed_drills"]}:
            problems.append(f"drill for finding {f.rank} is not from the coaching notes")
        bad = invented_numbers(f.why, [data, src["allowed_why"]])
        if bad:
            problems.append(f"invented numbers {bad} in finding {f.rank}")
    for t in (rep.headline_insight, rep.headline_sub, rep.strengths_text):
        bad = invented_numbers(t, data)
        if bad:
            problems.append(f"invented numbers {bad}")
    return problems


def _fmt_measured(f: dict) -> str:
    v = f["measured"]
    s = f"{v:.0f}" if f["unit"] == "°" else f"{v:.1f}"
    return f"{s}°" if f["unit"] == "°" else f"{s} {f['unit']}"


def template_report(payload: dict) -> dict:
    """Deterministic report built only from the payload and coaching notes."""
    notes = coaching_notes()
    q = payload["clip_quality"]
    fnd = payload["findings"]
    texts = []
    for f in fnd:
        n = notes.get(f["metric_key"], {"why": [""], "cue": [""], "drill": [""]})
        texts.append({"rank": f["rank"], "why": (n["why"] or [""])[0], "cue": (n["cue"] or [""])[0],
                      "drill": (n["drill"] or [""])[0]})
    if fnd:
        f = fnd[0]
        if f["kind"] == "injury":
            headline = f"{f['title']} ({_fmt_measured(f)}): a load pattern worth checking with a coach or physio."
        else:
            headline = f"Your biggest gain: {f['title'][0].lower() + f['title'][1:]} ({_fmt_measured(f)}; target {f['target']})."
        sub = (f"Measured on frame {f['frame']}, {f['view']}, {f['confidence'].lower()} confidence. "
               f"Clip quality {q['level'].lower()}.")
    else:
        cov = payload.get("coverage") or {}
        side_reasons = cov.get("not_measured_reasons") or []
        if side_reasons:
            headline = (f"Only {cov.get('measured', 0)} of {cov.get('total', 0)} metrics could be measured reliably from "
                        "this clip, and none of those crossed a target.")
            sub = f"Clip quality {q['level'].lower()}. {side_reasons[0]}"
        else:
            headline = "No clear technique issues were found in this delivery."
            sub = (f"Every judged metric was in range or couldn't be judged yet. Clip quality {q['level'].lower()}; "
                   "see the metrics table for what was measured.")
    strengths = " ".join(s["detail"] for s in payload["strengths"]) or \
        "No strengths could be confirmed at medium or high confidence in this clip."
    return {"headline_insight": headline, "headline_sub": sub, "findings": texts, "strengths_text": strengths}


def limits_note(fps: float, has_front: bool) -> str:
    parts = [
        "Measured from 2D video with a single camera per view, not a motion-capture lab; rotation towards or away from "
        "the camera cannot be measured exactly.",
        f"At {fps:.0f} fps, events can be off by up to {1000 / fps:.0f} ms (one frame).",
        "Targets are research-based population ranges; your coach may set different ones.",
        "Load patterns are prompts to talk to a coach or physio, not a diagnosis.",
    ]
    if not has_front:
        parts.append("No second clip (front-on or behind the bowler) was provided, so alignment, side bend, head "
                     "falling away and front-foot alignment were not measured.")
    return " ".join(parts)


def call_gemini(payload: dict) -> Optional[LLMReport]:
    """One call with a 15 s timeout and a single retry on rate-limit/unavailable errors; None on any failure."""
    if not settings.gemini_api_key:
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        log.warning("google-genai is not installed; using the template")
        return None
    client = genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(timeout=15_000))
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, response_mime_type="application/json",
                                         response_schema=LLMReport, temperature=0.3)
    for attempt in range(2):
        try:
            resp = client.models.generate_content(model=settings.gemini_model,
                                                  contents=json.dumps(payload, ensure_ascii=False), config=config)
            return LLMReport.model_validate_json(resp.text or "")
        except Exception as exc:  # network errors, 429 rate limits, invalid JSON → retry once, then template
            transient = any(code in str(exc) for code in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if attempt == 0 and transient:
                time.sleep(1.5 + random.random())  # noqa: S311  # nosec B311 (retry jitter, not security)
                continue
            log.warning("Gemini unavailable (%s); using the template", type(exc).__name__)
            return None
    return None


def generate_report(payload: dict, fps: float, has_front: bool) -> dict:
    """Returns {headline_insight, headline_sub, strengths_text, limits_note, generated_by, findings:[{rank,why,cue,drill}]}."""
    rep = call_gemini(payload) if payload["findings"] or payload["strengths"] else None
    out: dict
    if rep is not None:
        problems = validate_llm(rep, payload)
        if problems:
            log.warning("Discarding Gemini report: %s", "; ".join(problems))
            out = {**template_report(payload), "generated_by": "template"}
        else:
            out = {**rep.model_dump(), "generated_by": "gemini"}
    else:
        out = {**template_report(payload), "generated_by": "template"}
    out["limits_note"] = limits_note(fps, has_front)
    return out

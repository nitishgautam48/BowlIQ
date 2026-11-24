"""PDF report (ReportLab): headline, ball speed, top fixes with sources, metrics table, 3 key frames, limits note."""
from __future__ import annotations

import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.analysis.rules import fmt

GREEN = colors.HexColor("#0C7A3E")
AMBER = colors.HexColor("#8F5800")
MUTED = colors.HexColor("#535961")
LINE = colors.HexColor("#D9DDE1")

STATUS_TXT = {"in_range": "In range", "outside": "Outside target", "not_measured": "Not measured", "info": "Info"}


def _styles():
    ss = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=20, leading=24, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13),
        "small": ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, leading=10.5, textColor=MUTED),
        "big": ParagraphStyle("big", parent=ss["BodyText"], fontSize=14, leading=18, textColor=colors.black),
    }


def _esc(s: str | None) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf(result: dict, key_frames: dict[str, bytes], out: Path) -> Path:
    S = _styles()
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title=f"BowlIQ report – {result['bowler_name']}")
    st: list = []
    rep = result["report"]
    q = result["quality"]
    st.append(Paragraph("BowlIQ delivery report", S["h1"]))
    st.append(Paragraph(_esc(f"{result['bowler_name']} · {result['created_at'][:16].replace('T', ' ')} UTC · "
                             f"{result['bowling_arm']}-arm · clip quality {q['level']} ({q['fps']:.0f} fps)"), S["small"]))
    st.append(Spacer(1, 6))
    st.append(Paragraph(_esc(rep["headline_insight"]), S["big"]))
    st.append(Paragraph(_esc(rep["headline_sub"]), S["body"]))
    bs = result["ball_speed"]
    st.append(Spacer(1, 4))
    if bs["value_kmh"] is not None:
        st.append(Paragraph(f"<b>Ball speed:</b> {bs['value_kmh']:.0f} km/h ± {bs['plus_minus']:.0f} "
                            f"(tracked over {bs['frames_tracked']} frames)", S["body"]))
    else:
        st.append(Paragraph(f"<b>Ball speed:</b> not measured. {_esc(bs['reason'])}", S["body"]))

    st.append(Paragraph("Top fixes", S["h2"]))
    if not result["findings"]:
        st.append(Paragraph("No issues were found that cross a research threshold at the available confidence.", S["body"]))
    for f in result["findings"]:
        block = [Paragraph(f"<b>{f['rank']}. {_esc(f['title'])}</b> — {fmt(f['measured'], f['unit'])} "
                           f"(target {_esc(f['target_text'])}) · frame {f['frame']} · {f['view']} · {f['confidence']} confidence",
                           S["body"])]
        if f.get("why"):
            block.append(Paragraph(f"<i>Why:</i> {_esc(f['why'])}", S["body"]))
        if f.get("cue"):
            block.append(Paragraph(f"<i>Cue:</i> {_esc(f['cue'])}", S["body"]))
        if f.get("drill"):
            block.append(Paragraph(f"<i>Drill:</i> {_esc(f['drill'])}", S["body"]))
        if f.get("source"):
            block.append(Paragraph(f"Source: {_esc(f['source'])}", S["small"]))
        block.append(Spacer(1, 5))
        st.append(KeepTogether(block))
    if result["injury_flags"]:
        st.append(Paragraph("Load patterns", S["h2"]))
        st.extend(Paragraph(f"<b>{_esc(fl['title'])}</b>: {fmt(fl['measured'], fl['unit'])} "
                            f"(flag at {fl['threshold']:.0f}°). {_esc(fl['text'])}.", S["body"])
                  for fl in result["injury_flags"])
    if result["strengths"]:
        st.append(Paragraph("Strengths", S["h2"]))
        st.append(Paragraph(_esc(rep["strengths_text"]), S["body"]))

    st.append(Paragraph("Key frames", S["h2"]))
    imgs, caps = [], []
    for label, key in (("Back foot contact", "bfc"), ("Front foot contact", "ffc"), ("Release", "release")):
        data = key_frames.get(key)
        if data:
            from PIL import Image as PILImage  # Pillow ships with ReportLab

            w, h = PILImage.open(io.BytesIO(data)).size
            width = 56 * mm
            imgs.append(Image(io.BytesIO(data), width=width, height=width * h / w))
            ev = result["events"][key]
            caps.append(Paragraph(f"{label} · frame {ev['frame']} · {ev['confidence']}", S["small"]))
    if imgs:
        t = Table([imgs, caps], colWidths=[58 * mm] * len(imgs))
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1)]))
        st.append(t)

    st.append(Paragraph("All metrics", S["h2"]))
    rows = [["Metric", "Value", "Target", "Status", "View", "Frame", "Confidence"]]
    for m in result["metrics"]:
        val = fmt(m["value"], m["unit"]) if m["value"] is not None else "—"
        rows.append([Paragraph(_esc(m["name"]), S["small"]), val, Paragraph(_esc(m["target_text"] or "Trend only"), S["small"]),
                     STATUS_TXT[m["status"]], m["view"], "" if m["frame"] is None else str(m["frame"]),
                     Paragraph(_esc((m["confidence"] or "—") + (": " + m["confidence_reason"] if m["status"] != "not_measured" else ": " + (m["reason"] or ""))), S["small"])])
    t = Table(rows, colWidths=[36 * mm, 20 * mm, 30 * mm, 20 * mm, 16 * mm, 12 * mm, 44 * mm], repeatRows=1)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    for i, m in enumerate(result["metrics"], start=1):
        if m["status"] == "outside":
            style.append(("TEXTCOLOR", (3, i), (3, i), AMBER))
        elif m["status"] == "in_range":
            style.append(("TEXTCOLOR", (3, i), (3, i), GREEN))
    t.setStyle(TableStyle(style))
    st.append(t)
    st.append(Paragraph("Sources", S["h2"]))
    seen = []
    for m in result["metrics"]:
        if m["source"] and m["source"] not in seen:
            seen.append(m["source"])
            st.append(Paragraph(f"{_esc(m['name'])}: {_esc(m['source'])}", S["small"]))
    st.append(Paragraph("Limits", S["h2"]))
    st.append(Paragraph(_esc(rep["limits_note"]), S["small"]))
    doc.build(st)
    return out

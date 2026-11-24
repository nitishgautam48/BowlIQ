"""Optional first pass: Gemini watches the clip once and describes it (a "clip brief").

Gemini is good at understanding what happens in a clip (which person is bowling, from which side the camera films,
which arm bowls, slow motion or not) and bad at measuring: it looks at a few frames per second, so it can't time foot
contacts or measure angles. So the brief is only ever a set of HINTS for the local pipeline, which still measures
everything and checks each hint against its own evidence (see orchestrator.analyse and combine()).

Off unless BOWLIQ_GEMINI_VIDEO=1 and GEMINI_API_KEY are set. When on, the processed clip (trimmed, ≤1280 px, no
sound) is sent to Google. Any failure (no key, no network, quota, bad answer) returns an error string and the
analysis continues exactly as without it.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.core.config import settings
from app.pipeline.clip_context import Detection
from app.pipeline.params import P

log = logging.getLogger("bowliq.gemini_video")
G = P.gemini_video
Conf = Literal["high", "medium", "low"]


class BowlerBox(BaseModel):
    time_s: float = Field(description="Seconds from the start of this clip at which the box is given")
    x0: float = Field(description="Left edge, fraction of the frame width (0 = left, 1 = right)")
    y0: float = Field(description="Top edge, fraction of the frame height (0 = top, 1 = bottom)")
    x1: float = Field(description="Right edge, fraction of the frame width")
    y1: float = Field(description="Bottom edge, fraction of the frame height")


class ClipBrief(BaseModel):
    is_bowling_delivery: bool = Field(description="True if the clip shows a cricket bowler delivering a ball")
    delivery_start_s: Optional[float] = Field(None, description="Roughly when the last run-up strides begin")
    release_time_s: Optional[float] = Field(None, description="Roughly when the ball leaves the bowler's hand")
    delivery_end_s: Optional[float] = Field(None, description="Roughly when the follow-through ends")
    bowler: Optional[BowlerBox] = Field(None, description="Box around the BOWLER (not umpire/fielders/batter), "
                                                          "at a moment shortly before release when fully visible")
    people_in_frame: int = Field(0, description="Most people visible at once")
    view: Literal["side-on", "front-on", "behind", "oblique", "unclear"]
    view_confidence: Conf
    bowling_arm: Literal["right", "left", "unclear"]
    bowling_arm_confidence: Conf
    run_direction: Literal["left_to_right", "right_to_left", "towards_camera", "away_from_camera", "unclear"]
    run_direction_confidence: Conf
    slow_motion: Literal["real_time", "slow_motion", "unclear"]
    slow_motion_factor: Optional[float] = Field(None, description="If slowed down: about how many times slower")
    slow_motion_confidence: Conf
    issues: list[str] = Field(default_factory=list, description="Short notes on problems: bowler cut off, too dark, "
                                                               "several deliveries, camera shake, etc.")


PROMPT = """You are helping a cricket fast-bowling analysis app. Watch the video and describe it. Answer only what you
can see; use "unclear" (and low confidence) rather than guessing.

Definitions:
- bowler: the player who runs in and delivers the ball. Not the umpire, batter, keeper or fielders.
- view: where the CAMERA is. "side-on" = beside the pitch, the bowler runs across the picture. "front-on" = in front
  of the bowler (behind the batter's stumps), the bowler runs towards the camera. "behind" = behind the bowler, who
  runs away from the camera. "oblique" = clearly diagonal, in between.
- bowling_arm: the bowler's OWN arm that delivers the ball (anatomical right or left), not the side of the picture.
  A right-arm bowler filmed from behind delivers on the picture's right; filmed front-on, on the picture's left.
- run_direction: the bowler's movement in the picture during the run-up.
- slow_motion: "slow_motion" only if the footage itself is slowed down (e.g. a TV replay); estimate the factor.
- Times are in seconds from the start of THIS video. The bowler box uses fractions of the frame (0–1).
"""


def enabled() -> bool:
    return bool(settings.gemini_video and settings.gemini_api_key)


def _norm_box(b: BowlerBox) -> BowlerBox:
    vals = [b.x0, b.y0, b.x1, b.y1]
    if max(vals) > 1.5:  # answered on a 0–1000 grid
        vals = [v / 1000.0 for v in vals]
    x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in vals)
    return BowlerBox(time_s=b.time_s, x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1))


def brief(video: Path, duration_s: float) -> tuple[Optional[ClipBrief], Optional[str]]:
    """(brief, None) or (None, why it couldn't be made). Never raises."""
    if not enabled():
        return None, "Gemini video check is off."
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, "google-genai isn't installed."
    data = Path(video).read_bytes()
    if len(data) > G.max_upload_mb * 1024 * 1024:
        return None, f"The clip is larger than {G.max_upload_mb} MB, so it wasn't sent to Gemini."
    fps = float(min(G.max_fps, max(1.0, G.frames_budget / max(duration_s, 0.1))))
    try:
        client = genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(timeout=int(G.timeout_s * 1000)))
    except Exception as exc:  # bad key format, no network at start-up: carry on without the check
        log.warning("Gemini client failed: %s", type(exc).__name__)
        return None, f"Gemini couldn't be reached ({type(exc).__name__})."
    part = types.Part(inline_data=types.Blob(data=data, mime_type="video/mp4"), video_metadata=types.VideoMetadata(fps=fps))
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=ClipBrief, temperature=0.1)
    model = settings.gemini_video_model or settings.gemini_model
    for attempt in range(2):
        try:
            resp = client.models.generate_content(model=model, contents=[part, PROMPT], config=config)
            b = ClipBrief.model_validate_json(resp.text or "")
            if b.bowler is not None:
                b.bowler = _norm_box(b.bowler)
            log.info("Gemini brief (%s, %.1f fps): view %s/%s arm %s/%s slow %s release %s", model, fps, b.view,
                     b.view_confidence, b.bowling_arm, b.bowling_arm_confidence, b.slow_motion, b.release_time_s)
            return b, None
        except Exception as exc:  # network, quota, invalid JSON: retry once if transient, else give up quietly
            transient = any(c in str(exc) for c in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "timed out"))
            if attempt == 0 and transient:
                time.sleep(2.0 + random.random())  # noqa: S311  # nosec B311 (retry jitter)
                continue
            log.warning("Gemini video check failed: %s", type(exc).__name__)
            return None, f"Gemini couldn't check the clip ({type(exc).__name__})."
    return None, "Gemini couldn't check the clip."


LEVELS = ["Low", "Medium", "High"]


def combine(det: Detection, value: Optional[str], conf: Optional[str], label: str) -> tuple[Detection, str]:
    """Merge a Gemini hint into a pipeline detection. Returns (detection, outcome) with outcome one of
    'none' | 'agree' | 'gemini' (Gemini's answer used) | 'conflict' (kept the measurement, flagged)."""
    if det.source == "user" or value in (None, "unclear") or conf is None:
        return det, "none"
    if value == det.value:
        lv = LEVELS[min(2, LEVELS.index(det.confidence) + 1)]
        return Detection(det.value, lv, det.reasons + [f"Gemini, watching the clip, agrees ({conf} confidence)."],
                         det.source, {**det.details, "gemini": value}), "agree"
    if conf in G.override_confidence and det.confidence in G.override_when_pipeline:
        return Detection(value, "Medium", det.reasons + [
            f"The measurements suggested {det.value} with {det.confidence.lower()} confidence; Gemini, watching the "
            f"clip, says {value} ({conf} confidence), so {value} was used."],
            "auto", {**det.details, "gemini": value, "pipeline_value": det.value}), "gemini"
    return Detection(det.value, "Low", det.reasons + [
        f"Gemini, watching the clip, says {value} ({conf} confidence): the {label} is uncertain — check it."],
        det.source, {**det.details, "gemini": value}), "conflict"

"""Pydantic v2 API schemas. Mirrors frontend/types/analysis.ts exactly."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Confidence = Literal["High", "Medium", "Low"]
MetricStatus = Literal["in_range", "outside", "not_measured", "info"]
View = Literal["side-on", "front-on", "behind", "diagonal", "3d"]
Direction = Literal["band", "max", "higher_better", "monitor"]
JobStatus = Literal["queued", "processing", "done", "failed"]
Stage = Literal["Queued", "Checking quality", "Detecting skeleton", "Finding key moments", "Measuring", "Writing report", "Done"]
EventName = Literal["BFC", "FFC", "release"]
CheckStatus = Literal["ok", "warn", "fail"]
ChainStatus = Literal["ok", "warn", "unknown"]


class ApiError(BaseModel):
    code: str
    message: str
    fix: str


class QualityCheck(BaseModel):
    label: str
    value: str
    status: CheckStatus
    note: str


class ClipQuality(BaseModel):
    level: Confidence
    reasons: list[str]
    fps: float
    width: int
    height: int
    duration_s: float
    rotation: int
    camera_motion: Optional[float] = None
    view_detected: Literal["side-on", "front-on", "behind", "diagonal", "unknown"]
    checks: list[QualityCheck]


class EventInfo(BaseModel):
    name: EventName
    frame: int
    time_s: float
    confidence: Confidence
    confidence_reason: str
    methods: dict[str, Optional[int]] = Field(default_factory=dict)  # each detection method's frame (consensus)
    agree: int = 0  # how many methods agreed


class Events(BaseModel):
    bfc: EventInfo
    ffc: EventInfo
    release: EventInfo


class FrontEvents(BaseModel):
    bfc: Optional[EventInfo] = None
    ffc: Optional[EventInfo] = None
    release: Optional[EventInfo] = None
    reason: Optional[str] = None


class Metric(BaseModel):
    key: str
    name: str
    value: Optional[float] = None
    unit: str
    view: View
    frame: Optional[int] = None
    event: Optional[str] = None
    confidence: Optional[Confidence] = None
    confidence_reason: str
    status: MetricStatus
    reason: Optional[str] = None
    direction: Direction
    target_text: Optional[str] = None
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    baseline: Optional[float] = None
    method: str
    source: Optional[str] = None
    joint: Optional[str] = None


class BallSpeed(BaseModel):
    value_kmh: Optional[float] = None
    plus_minus: Optional[float] = None
    status: Literal["measured", "not_measured"]
    reason: Optional[str] = None
    frames_tracked: int = 0


class Finding(BaseModel):
    rank: int
    title: str
    metric: str
    measured: float
    unit: str
    target_text: str
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    frame: Optional[int] = None
    view: View
    confidence: Confidence
    confidence_reason: str
    source: Optional[str] = None
    kind: Literal["pace", "injury"]
    severity: float
    why: str = ""
    cue: str = ""
    drill: str = ""


class Strength(BaseModel):
    title: str
    detail: str
    metric: str


class ChainSegment(BaseModel):
    segment: str
    status: ChainStatus
    metric_text: str
    metric: Optional[str] = None


class InjuryFlag(BaseModel):
    metric: str
    title: str
    measured: float
    unit: str
    threshold: float
    text: str
    source: Optional[str] = None
    view: View
    confidence: Confidence
    frame: Optional[int] = None


class BaselineInfo(BaseModel):
    active: bool
    deliveries: int
    deliveries_until_baseline: int
    based_on: Optional[str] = None


class ScaleInfo(BaseModel):
    method: Literal["height", "body_relative"]
    metres_per_pixel: Optional[float] = None
    body_height_px: float
    frame: int
    rel_uncertainty: float = 0.0


class SlowMo(BaseModel):
    start_frame: int
    end_frame: int
    factor: int


class VideoInfo(BaseModel):
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float
    slowmo: SlowMo


class Report(BaseModel):
    headline_insight: str
    headline_sub: str
    strengths_text: str
    limits_note: str
    generated_by: Literal["gemini", "template"]


class SyncInfo(BaseModel):
    """How the front-on clip lines up with the side-on clip: front_time = side_time + offset_s."""
    method: Literal["audio", "motion", "release", "none"]
    offset_s: Optional[float] = None
    confidence: Literal["High", "Medium", "Low"]
    detail: str
    audio_offset_s: Optional[float] = None
    audio_score: Optional[float] = None
    audio_peak_ratio: Optional[float] = None
    anchor_side_s: Optional[float] = None
    release_offset_s: Optional[float] = None
    motion_corr: Optional[float] = None
    manual_offset_s: Optional[float] = None  # set by the user in Linked playback; overrides offset_s when present


class SyncIn(BaseModel):
    offset_s: Optional[float] = Field(None, ge=-60, le=60, allow_inf_nan=False)


class AnalysisResult(BaseModel):
    id: str
    created_at: str
    bowler_name: str
    bowler_height_cm: Optional[float] = None
    bowling_arm: Literal["right", "left"]
    direction: Literal["left_to_right", "right_to_left"]
    has_front: bool
    video: VideoInfo
    quality: ClipQuality
    front_quality: Optional[ClipQuality] = None
    events: Events
    front_events: Optional[FrontEvents] = None
    key_moments_estimated: bool = False  # couldn't be detected reliably: estimated; the user should check/correct them
    front_video: Optional[VideoInfo] = None
    sync: Optional[SyncInfo] = None
    scale: ScaleInfo
    metrics: list[Metric]
    ball_speed: BallSpeed
    headline_metric: Optional[str] = None
    findings: list[Finding]
    strengths: list[Strength]
    kinetic_chain: list[ChainSegment]
    injury_flags: list[InjuryFlag]
    baseline: BaselineInfo
    report: Report
    feedback: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    # generalisation refactor: what was detected about the clip, how sure, and what the user should check
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    context: Optional[dict] = None  # clip_context.ClipContext.to_dict()
    stages: list[dict] = Field(default_factory=list)  # [{name, ok, detail}]
    overrides: dict = Field(default_factory=dict)  # corrections applied with /recompute
    can_recompute: bool = False  # cached keypoints exist: corrections take seconds
    diagnostics: bool = False  # storage/debug/{id}/ was written
    recomputed_at: Optional[str] = None
    gemini: Optional[dict] = None  # optional Gemini video check: {used, error, brief, outcomes}


class RecomputeIn(BaseModel):
    """Corrections for /analysis/{id}/recompute. Every field is optional; omitted = keep the automatic value."""
    view: Optional[Literal["side-on", "front-on", "behind", "oblique", "diagonal"]] = None
    bowling_arm: Optional[Literal["right", "left"]] = None
    direction: Optional[Literal["left_to_right", "right_to_left"]] = None
    bowler_track_id: Optional[int] = Field(default=None, ge=0, le=50)
    bfc_frame: Optional[int] = Field(default=None, ge=0)
    ffc_frame: Optional[int] = Field(default=None, ge=0)
    release_frame: Optional[int] = Field(default=None, ge=0)
    height_cm: Optional[float] = Field(default=None, ge=100, le=230)
    clear: bool = False  # true: drop every earlier correction first


class JobCreated(BaseModel):
    job_id: str


class ClipInfo(BaseModel):
    fps: float
    frame_count: int
    width: int
    height: int


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    stage: Stage
    progress: int
    error: Optional[ApiError] = None
    result: Optional[AnalysisResult] = None
    can_mark: bool = False  # failed, but the key moments can be marked by hand on the processed clip
    clip: Optional[ClipInfo] = None


class KeypointsResponse(BaseModel):
    fps: float
    width: int
    height: int
    joints: list[str]
    frames: list[list[list[float]]]


class DeliverySummary(BaseModel):
    id: str
    created_at: str
    bowler_name: str
    ball_speed_kmh: Optional[float] = None
    quality: Confidence
    top_finding: Optional[str] = None
    has_front: bool


class BowlerSummary(BaseModel):
    name: str
    deliveries: int
    last_delivery_at: Optional[str] = None


class MetricPoint(BaseModel):
    value: Optional[float] = None
    unit: str


class HistoryPoint(BaseModel):
    id: str
    created_at: str
    ball_speed_kmh: Optional[float] = None
    quality: Confidence
    top_finding: Optional[str] = None
    metrics: dict[str, MetricPoint]


class BowlerHistory(BaseModel):
    name: str
    deliveries: list[HistoryPoint]
    baseline_active: bool
    deliveries_until_baseline: int


class MetricDiff(BaseModel):
    key: str
    name: str
    unit: str
    a: Optional[float] = None
    b: Optional[float] = None
    delta: Optional[float] = None
    verdict: Literal["better", "worse", "same", "neutral", "not_comparable"]
    note: str


class CompareResponse(BaseModel):
    a: AnalysisResult
    b: AnalysisResult
    diffs: list[MetricDiff]


class FeedbackIn(BaseModel):
    finding_rank: int = Field(ge=1, le=3)
    helpful: bool


class HealthResponse(BaseModel):
    ok: bool
    ffmpeg: bool
    model_file: bool
    database: bool
    pose_device: Literal["gpu", "cpu", "unknown"]
    pose_engine: str = "unknown"
    video_encoder: str = "unknown"
    gemini_key: bool
    gemini_video: bool = False  # the optional Gemini video check is on (clips are sent to Google)
    details: dict[str, str]


class MediaLinks(BaseModel):
    query: str
    expires_at: int


class Me(BaseModel):
    owner: str


class SampleClip(BaseModel):
    name: str
    size_bytes: int


class IssueExample(BaseModel):
    id: str
    frame: Optional[int] = None
    view: Optional[str] = None
    value: float


class RepeatIssue(BaseModel):
    metric: str
    title: str
    name: str
    pattern: Literal["consistent", "occasional", "one-off"]
    count: int
    measured: int
    uncertain: int
    share: float
    kind: Literal["pace", "injury"]
    typical: float
    unit: str
    target_text: Optional[str] = None
    cue: str
    drill: str
    example: IssueExample


class RepeatStrength(BaseModel):
    metric: str
    name: str
    measured: int
    typical: float
    unit: str
    text: str


class MetricTrend(BaseModel):
    metric: str
    name: str
    unit: str
    measured: int
    typical: float
    spread: float
    change: float
    direction: Literal["up", "down", "steady"]


class SpeedSummary(BaseModel):
    median: float
    best: float
    measured: int


class BowlerSummaryOut(BaseModel):
    """Phase 5: what keeps happening across a bowler's recent deliveries."""
    name: str
    deliveries: int
    provisional: bool
    ball_speed: Optional[SpeedSummary] = None
    issues: list[RepeatIssue]
    strengths: list[RepeatStrength]
    trends: list[MetricTrend]
    summary: str


class EventsIn(BaseModel):
    """Key moments set by hand, as frame numbers of the analysis's side-on video."""
    bfc: int = Field(ge=0)
    ffc: int = Field(ge=0)
    release: int = Field(ge=0)
    slow_motion: Optional[float] = Field(None, ge=1, le=60, allow_inf_nan=False)

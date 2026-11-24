// API contract between the BowlIQ frontend and backend.
// Mirrored exactly by backend/app/models/schemas.py (Pydantic v2).

export type Confidence = "High" | "Medium" | "Low";
export type MetricStatus = "in_range" | "outside" | "not_measured" | "info";
export type View = "side-on" | "front-on" | "behind" | "diagonal" | "3d";
export type Direction = "band" | "max" | "higher_better" | "monitor";
export type JobStatus = "queued" | "processing" | "done" | "failed";
export type Stage =
  | "Queued"
  | "Checking quality"
  | "Detecting skeleton"
  | "Finding key moments"
  | "Measuring"
  | "Writing report"
  | "Done";
export type EventName = "BFC" | "FFC" | "release";
export type CheckStatus = "ok" | "warn" | "fail";
export type ChainStatus = "ok" | "warn" | "unknown";

export interface ApiErrorBody {
  code: string;
  message: string;
  fix: string;
}

export interface QualityCheck {
  label: string;
  value: string;
  status: CheckStatus;
  note: string;
}

export interface ClipQuality {
  level: Confidence;
  reasons: string[];
  fps: number;
  width: number;
  height: number;
  duration_s: number;
  rotation: number;
  camera_motion: number | null;
  view_detected: "side-on" | "front-on" | "behind" | "diagonal" | "unknown";
  checks: QualityCheck[];
}

export interface EventInfo {
  name: EventName;
  frame: number;
  time_s: number;
  confidence: Confidence;
  confidence_reason: string;
  /** each detection method's frame (null = that method found nothing); "user" when set by hand */
  methods?: Record<string, number | null>;
  agree?: number;
}

export interface Events {
  bfc: EventInfo;
  ffc: EventInfo;
  release: EventInfo;
}

export interface FrontEvents {
  bfc: EventInfo | null;
  ffc: EventInfo | null;
  release: EventInfo | null;
  reason: string | null;
}

export interface Metric {
  key: string;
  name: string;
  value: number | null;
  unit: string;
  view: View;
  frame: number | null;
  event: string | null;
  confidence: Confidence | null;
  confidence_reason: string;
  status: MetricStatus;
  reason: string | null;
  direction: Direction;
  target_text: string | null;
  target_min: number | null;
  target_max: number | null;
  baseline: number | null;
  method: string;
  source: string | null;
  joint: string | null;
}

export interface BallSpeed {
  value_kmh: number | null;
  plus_minus: number | null;
  status: "measured" | "not_measured";
  reason: string | null;
  frames_tracked: number;
}

export interface Finding {
  rank: number;
  title: string;
  metric: string;
  measured: number;
  unit: string;
  target_text: string;
  target_min: number | null;
  target_max: number | null;
  frame: number | null;
  view: View;
  confidence: Confidence;
  confidence_reason: string;
  source: string | null;
  kind: "pace" | "injury";
  severity: number;
  why: string;
  cue: string;
  drill: string;
}

export interface Strength {
  title: string;
  detail: string;
  metric: string;
}

export interface ChainSegment {
  segment: string;
  status: ChainStatus;
  metric_text: string;
  metric: string | null;
}

export interface InjuryFlag {
  metric: string;
  title: string;
  measured: number;
  unit: string;
  threshold: number;
  text: string;
  source: string | null;
  view: View;
  confidence: Confidence;
  frame: number | null;
}

export interface BaselineInfo {
  active: boolean;
  deliveries: number;
  deliveries_until_baseline: number;
  based_on: string | null;
}

export interface ScaleInfo {
  method: "height" | "body_relative";
  metres_per_pixel: number | null;
  body_height_px: number;
  frame: number;
  rel_uncertainty: number;
}

export interface SlowMo {
  start_frame: number;
  end_frame: number;
  factor: number;
}

export interface VideoInfo {
  fps: number;
  width: number;
  height: number;
  frame_count: number;
  duration_s: number;
  slowmo: SlowMo;
}

export interface Report {
  headline_insight: string;
  headline_sub: string;
  strengths_text: string;
  limits_note: string;
  generated_by: "gemini" | "template";
}

/** How the front-on clip lines up with the side-on clip: front_time = side_time + offset_s. */
export interface SyncInfo {
  method: "audio" | "motion" | "release" | "none";
  offset_s: number | null;
  confidence: "High" | "Medium" | "Low";
  detail: string;
  audio_offset_s: number | null;
  audio_score: number | null;
  audio_peak_ratio: number | null;
  anchor_side_s: number | null;
  release_offset_s: number | null;
  motion_corr?: number | null;
  manual_offset_s: number | null;
}

export interface AnalysisResult {
  id: string;
  created_at: string;
  bowler_name: string;
  bowler_height_cm: number | null;
  bowling_arm: "right" | "left";
  direction: "left_to_right" | "right_to_left";
  has_front: boolean;
  video: VideoInfo;
  quality: ClipQuality;
  front_quality: ClipQuality | null;
  events: Events;
  front_events: FrontEvents | null;
  key_moments_estimated?: boolean;
  front_video?: VideoInfo | null;
  sync?: SyncInfo | null;
  scale: ScaleInfo;
  metrics: Metric[];
  ball_speed: BallSpeed;
  headline_metric: string | null;
  findings: Finding[];
  strengths: Strength[];
  kinetic_chain: ChainSegment[];
  injury_flags: InjuryFlag[];
  baseline: BaselineInfo;
  report: Report;
  feedback: Record<string, boolean>;
  warnings: string[];
  needs_review?: boolean;
  review_reasons?: string[];
  context?: ClipContext | null;
  stages?: { name: string; ok: boolean; detail: string | null }[];
  overrides?: Overrides;
  can_recompute?: boolean;
  diagnostics?: boolean;
  recomputed_at?: string | null;
  gemini?: { used: boolean; error: string | null; brief: Record<string, unknown> | null; outcomes: Record<string, string> } | null;
}

export interface Detection<T = string | number> {
  value: T;
  confidence: Confidence;
  reasons: string[];
  source: "auto" | "user";
  details: Record<string, unknown>;
}

export interface PersonCandidate {
  id: number;
  frame: number;
  first_frame: number;
  last_frame: number;
  box: [number, number, number, number];
  score: number;
  travel: number;
  decel: number;
  arm: number;
}

export interface ClipContext {
  rotation: Detection<string>;
  direction: Detection<string>;
  bowling_arm: Detection<string>;
  view: Detection<string>;
  bowler: Detection<number>;
  candidates: PersonCandidate[];
}

export interface Overrides {
  view?: "side-on" | "front-on" | "behind" | "oblique";
  bowling_arm?: "right" | "left";
  direction?: "left_to_right" | "right_to_left";
  bowler_track_id?: number;
  bfc_frame?: number;
  ffc_frame?: number;
  release_frame?: number;
  height_cm?: number;
}

export interface JobCreated {
  job_id: string;
}

export interface JobState {
  job_id: string;
  status: JobStatus;
  stage: Stage;
  progress: number;
  error: ApiErrorBody | null;
  result: AnalysisResult | null;
  /** failed, but can be finished by marking the key moments by hand on the processed clip */
  can_mark?: boolean;
  clip?: { fps: number; frame_count: number; width: number; height: number } | null;
}

export interface KeypointsResponse {
  fps: number;
  width: number;
  height: number;
  joints: string[];
  frames: [number, number, number][][];
}

export interface DeliverySummary {
  id: string;
  created_at: string;
  bowler_name: string;
  ball_speed_kmh: number | null;
  quality: Confidence;
  top_finding: string | null;
  has_front: boolean;
}

export interface BowlerSummary {
  name: string;
  deliveries: number;
  last_delivery_at: string | null;
}

export interface MetricPoint {
  value: number | null;
  unit: string;
}

export interface HistoryPoint {
  id: string;
  created_at: string;
  ball_speed_kmh: number | null;
  quality: Confidence;
  top_finding: string | null;
  metrics: Record<string, MetricPoint>;
}

export interface BowlerHistory {
  name: string;
  deliveries: HistoryPoint[];
  baseline_active: boolean;
  deliveries_until_baseline: number;
}

export interface MetricDiff {
  key: string;
  name: string;
  unit: string;
  a: number | null;
  b: number | null;
  delta: number | null;
  verdict: "better" | "worse" | "same" | "neutral" | "not_comparable";
  note: string;
}

export interface CompareResponse {
  a: AnalysisResult;
  b: AnalysisResult;
  diffs: MetricDiff[];
}

export interface FeedbackIn {
  finding_rank: number;
  helpful: boolean;
}

export interface HealthResponse {
  ok: boolean;
  ffmpeg: boolean;
  model_file: boolean;
  database: boolean;
  pose_device: "gpu" | "cpu" | "unknown";
  pose_engine: string;
  video_encoder: string;
  gemini_key: boolean;
  gemini_video?: boolean;
  details: Record<string, string>;
}

export interface BenchmarkRule {
  name: string;
  unit: string;
  unit_relative?: string;
  view: View;
  type: Direction;
  band?: [number, number];
  threshold?: number;
  pace_weight?: number;
  injury_weight?: number;
  source: string;
  method: string;
  event: string;
}

export interface MediaLinks {
  query: string;
  expires_at: number;
}

export interface Me {
  owner: string;
}

export interface SampleClip {
  name: string;
  size_bytes: number;
}

export interface AnalyzeInput {
  sideVideo: File;
  frontVideo?: File | null;
  bowlerName?: string;
  bowlerHeightCm?: number | null;
  trimStartS?: number | null;
  trimEndS?: number | null;
  /** "t,x,y" — t in seconds of the uploaded clip, x/y fractions (0–1) of the frame */
  sideTap?: string | null;
  /** "x0,y0,x1,y1;…" fractions of the frame: people inside are ignored */
  sideIgnore?: string | null;
  frontTap?: string | null;
  frontIgnore?: string | null;
  /** slow-motion factor of the side-on clip (2 = half speed); omitted = real time */
  slowMotion?: number | null;
  cameraView?: "side-on" | "front-on" | "behind" | null;
  bowlingArm?: "right" | "left" | null;
}

export interface RepeatIssue {
  metric: string; title: string; name: string; pattern: "consistent" | "occasional" | "one-off";
  count: number; measured: number; uncertain: number; share: number; kind: "pace" | "injury";
  typical: number; unit: string; target_text: string | null; cue: string; drill: string;
  example: { id: string; frame: number | null; view: string | null; value: number };
}
export interface BowlerSummaryOut {
  name: string; deliveries: number; provisional: boolean;
  ball_speed: { median: number; best: number; measured: number } | null;
  issues: RepeatIssue[];
  strengths: { metric: string; name: string; measured: number; typical: number; unit: string; text: string }[];
  trends: { metric: string; name: string; unit: string; measured: number; typical: number; spread: number; change: number; direction: "up" | "down" | "steady" }[];
  summary: string;
}

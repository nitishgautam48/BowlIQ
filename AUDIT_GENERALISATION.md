# BowlIQ — Generalisation audit (Step 1, before any code change)

Scope: `backend/app/pipeline/*`, `backend/app/report/render_video.py`, `backend/app/core/jobs.py`.
Line numbers refer to the v2.13 code (the snapshot the refactor starts from).

Legend — **Kind**: PX = absolute pixel threshold · FR = fixed frame count · FPS = assumed frame rate ·
DIR/ARM/VIEW = assumed direction / arm / camera view · PERSON = single-person assumption ·
FATAL = one failed step fails the whole job · CLIP = constant calibrated on one specific clip.

## A. Absolute pixel thresholds

| # | File:line | What it assumes | Why it breaks |
|---|---|---|---|
| A1 | orchestrator.py:127 `MIN_BOWLER_PX = 70` | Bowler height in px means the same on every clip | Calibrated on clip 11.29.51 only; preprocessing resizes to ≤1280 px, so a 4K clip and a 480p clip give different "px" for the same framing. Should be a fraction of frame height. |
| A2 | adapt.py:197 `ZOOM_ALWAYS_PX = 120` | Same as A1 | "helped at ~90 px, hurt a 148 px clip": tuned to two clips. |
| A3 | adapt.py:200 `ZOOM_OUT_H = 512`, :284 clip `96` | Fixed crop size | OK as a model-input size, but the `96` px minimum crop is absolute. |
| A4 | adapt.py:31 `WORK_W = 480` | Working width | Fine (it's a processing resolution, transforms are rescaled) — keep, move to config. |
| A5 | orchestrator.py:235 `w > r * 1.05 + 1.0` | `+1.0` px slack on planted-foot drift | Absolute px; meaningless on a large/small bowler. Use BH. |
| A6 | tracking.py:311 `edge = max(2.0, 0.01*min(w,h))` | 2 px floor | Minor; keep relative part only. |
| A7 | tracking.py:59 `x1-x0 < 3` | Appearance ROI ≥ 3 px | Minor; tiny bowlers lose appearance (acceptable, documented). |
| A8 | quality.py:122-131 `720 / 480` | Resolution quality bands | Informational only; keep but move to config. |
| A9 | quality.py:336-344 `motion < 4.0 / 10.0`, orchestrator.py:129 `CAMERA_STEADY = 1.5` | Mean grey difference (0–255) | Depends on scene texture & exposure, not on camera motion alone ("plain sky read 2.5"). Should be relative to the clip's own frame-difference distribution / the measured camera path. |
| A10 | metrics.py:362 `camera_motion >= 10.0` | Same grey-difference scale as A9 | Same problem; drives not_measured for run-up speed. |
| A11 | ball_speed.py:50 `d_px < 3.0`, :141 `±1 px` | Ball size in px | Physically meaningful (detector limit) — keep, but in config. |
| A12 | scene cut quality.py:269 resize to 160 px | Thumbnail width | Fine (normalised histogram). |

## B. Fixed frame counts (fps-dependent)

| # | File:line | What it assumes | Why it breaks |
|---|---|---|---|
| B1 | events.py:54-55 `ref_n = max(6, …)`, `sup_n = max(3, …)` | ≥6 / ≥3 frames regardless of fps | At 30 fps 6 frames = 0.2 s (2× the stated 0.1 s); at 240 fps the floor never applies. Behaviour differs per fps. Tuned on 30 fps clip B. |
| B2 | events.py:56 `range(max(lo, 3), …)` and `len(prev_v) < 3` | 3 frames of history | 3 frames = 0.1 s at 30 fps but 12 ms at 240 fps. |
| B3 | events.py:73 `sp[f+1:f+3]` | "next 2 frames" | 67 ms at 30 fps, 8 ms at 240 fps. |
| B4 | events.py:324 `k = 3` (release sharpness) | 3 frames each side | Sharpness score changes with fps. |
| B5 | events.py:213,274 `min_gap = max(1|2, …)` | Inconsistent floors (1 vs 2) | Two detectors give different minimum FFC→release at the same fps. |
| B6 | smooth.py:10 `MAX_GAP = 6` | Interpolate gaps ≤ 6 frames | 0.2 s at 30 fps, 25 ms at 240 fps: high-fps clips get no gap filling, low-fps clips get unrealistically long fills. |
| B7 | smooth.py:17 `max(5, w)` window floor; :105,111 `f - f1 <= 3 / 6` (swap fixer) | Frames, not seconds | Savitzky–Golay floor of 5 frames = 167 ms at 30 fps (over-smooths release). |
| B8 | tracking.py:148 `max(3, int(0.05*fps))`, :186 `max(3, …)`, :203 `max(3, …)`, :275 `max(3, …)` | 3-frame floors | Mixed seconds/frames. |
| B9 | tracking.py:176 `k = min(5, …)` | Seed velocity over 5 frames | 0.17 s at 30 fps, 21 ms at 240 fps. |
| B10 | quality.py:228-241 `max(6, …)`, `len(k) < 5`, `ok.sum() < 5` | Minimum samples in frames | OK as statistical minima, but should be documented minimums. |
| B11 | ball_speed.py:16 `MAX_FRAMES = 20` | Ball tracked 20 frames | 0.67 s at 30 fps vs 83 ms at 240 fps. Should be seconds. |
| B12 | orchestrator.py:470 `prep.frame_count < 10` | 10 frames minimum | Should be the 1 s rule (STEP 5). |
| B13 | orchestrator.py:75, adapt.py:279 `int(0.1*fps) | 1` | Mixed | Fine in spirit, move to config. |

## C. Assumed frame rate / clip-calibrated constants

| # | File:line | What it assumes | Why it breaks |
|---|---|---|---|
| C1 | events.py:85-91 `_confidence`: fps ≥ 100 → High, < 50 → always Low | Confidence from fps bands only | A clean 60 fps clip can never be High; no notion of method agreement. |
| C2 | orchestrator.py:283-303 `SLOW_HAND_SPEED 4.0`, `SLOW_GUESS_SPEED 26`, `SLOW_CHOICES` | Calibrated on one TV replay (1.6 BH/s) | **CLIP**. The factor guess is from a single clip. |
| C3 | orchestrator.py:323-324 `0.12 s`, `0.18 s` fallback offsets | Typical timings | OK as priors; move to config, mark as priors. |
| C4 | quality.py:194-198 `SHRINK_BEHIND 0.5`, `GROW_FRONT 2.0`, `SIDE_BAND`, `SIDE_PROFILE_W 0.3`, `SIDE_STRIDE 0.5`, inline `1.2`, `0.55`, `1.0` | View thresholds "calibrated on the real clips" (docstring cites WhatsApp clip 0.57, behind clip 0.33, broadcast replay 0.11/0.93) | **CLIP**. Hard if/else chain — one feature slightly out of band flips the class with no confidence. |
| C5 | orchestrator.py:135 `MAX_SIDE_ON_TILT = 35` | Footprint tilt | Raises WRONG_CAMERA_ANGLE (FATAL-ish, see E). |
| C6 | adapt.py:33-34,75-77 `MOVED_PX 0.02`, `MOVED_DEG 1`, `MAX_STEP_*`, `MIN_INLIER_FRAC` | Camera path | Relative already; move to config. |
| C7 | sync.py:34-38, 213, 245-247 | Sync thresholds | Relative (correlations/seconds); move to config. |
| C8 | render_video.py:26-27 `SLOWMO_FACTOR 4`, `SLOWMO_HALF_S 0.3` | Presentation only | Fine. |

## D. Direction / arm / view / orientation assumptions

| # | File:line | What it assumes | Why it breaks |
|---|---|---|---|
| D1 | smooth.py:139-146 `detect_direction` | Median hip x of last 20 % vs first 20 % of **all valid frames** | Follow-through and walk-back after the delivery (often longer than the run-up) flip it. No confidence. |
| D2 | smooth.py:122-136 `detect_bowling_arm` | Wrist highest above the **nose** (2nd-highest frame) | Needs the nose (fails when the face is hidden at release, behind views); a front arm reaching up in the coil wins on some clips → orchestrator retries both arms (expensive, 2× event detection) with no confidence. |
| D3 | events.py:9, 49, 66 | `mode="side"` uses horizontal speed only; release-first search **only** runs for side-on | Front-on clips get only the weaker contact-first search. |
| D4 | events.py:217-221 front foot by `x` at release | Side-on only | In front-on/behind views x separation is sideways, not along the pitch. |
| D5 | events.py:140-150 `_check_runup` | Hips travel +x ≥ 0.8 BH/s before FFC (raises) | Panning camera or front-on view: hips don't cross the frame → "No run-up" error. `_running_gait` patch helps only side-on. |
| D6 | orchestrator.py:335, 649 `direction="left_to_right"` forced for front/behind | Direction meaningless there | OK but undocumented; arm/side labelling of left_on_right uses MediaPipe anatomical labels — YOLO labels the same way (verified), keep. |
| D7 | quality.py:252-254 `left_on_right` | Facing camera ⇒ anatomical left appears on image right | True for both engines; but used as a hard switch front-on vs behind. |
| D8 | Rotation: quality.py:33-44 + preprocess | Metadata rotation trusted blindly | A clip with wrong/missing rotation tag (screen recordings, WhatsApp re-encodes) is analysed sideways — never verified with pose (head above ankles). |
| D9 | metrics.py:362 run-up speed `abs(slope)` px→m | Side-on travel along x | Front-on travel is towards the camera (size growth), not measured. |

## E. Person selection (one person in frame)

| # | File:line | What it assumes | Why it breaks |
|---|---|---|---|
| E1 | tracking.py:200,211 | Tracks built from the **first half** of the clip only | A bowler entering late (long lead-in) is missed unless nobody moves at all. |
| E2 | tracking.py:202-209 `score` | Score = net horizontal travel × monotonicity / size | Front-on run-ups have little horizontal travel → the umpire walking or a fielder can win. No deceleration / arm-swing evidence. |
| E3 | tracking.py:217-218 | Fallback = **largest, longest-visible person** | Exactly the "biggest person" rule the spec forbids: the umpire in the foreground wins. |
| E4 | tracking.py:241 size ratio 0.65–1.5 gate | Constant apparent size | Front-on bowlers grow ×2–3 over the run-up; gate drops them mid-run. |
| E5 | orchestrator.py:409 `min(3, …)` candidates | Only 3 people tried | Busy nets: the bowler may be candidate 4+. |
| E6 | pose.py (MediaPipe `num_poses`) | Detector returns ≤ N people | Bounded by model config; fine. |

## F. One failure fails the whole job

| # | File:line | Failure | Effect today |
|---|---|---|---|
| F1 | scale.py:249 `estimate_scale` raises POSE_NOT_FOUND if < 3 frames have nose+ankles | No head+feet together | Job fails, although events and angles don't need scale. |
| F2 | events.py raises EVENTS_NOT_FOUND from 6 places | Any gate fails | Caught by orchestrator → estimate_events (good, v2.13) — but only on some code paths; `_choose_bowler` re-raises non-EVENTS codes (orchestrator.py:425). |
| F3 | orchestrator.py:405, 435, 437 WRONG_CAMERA_ANGLE | View classifier says front/behind or tilt > 35° | Converted to second-view analysis, but the decision is a hard switch on one classifier (C4). |
| F4 | tracking.py:301-309 POSE_NOT_FOUND if bowler seen < 0.5 s or < 30 % of the span | Short visibility | Fails the job even when the delivery stride (≈0.3 s) is visible. |
| F5 | orchestrator.py:470 CLIP_TOO_SHORT < 10 frames | — | Should be "< 1 s". |
| F6 | `measure_ball_speed`, `render_videos`, `frame_jpeg`, `build_pdf`, `generate_report`, `judge`, `side_metrics`, `front_metrics`, `apply_3d`, `scene_cuts`, `Q.camera_motion` — none wrapped | Any exception (e.g. NaN in a metric, cv2 decode error on one frame, PDF font issue) | INTERNAL_ERROR, the whole result is discarded (jobs.py:179-187). |
| F7 | `legs_consistent` retry, `_correction_helps` re-run of `_choose_bowler` | Exceptions swallowed silently | Not fatal, but no warning/confidence reported. |
| F8 | jobs.py:177 files deleted on non-markable errors | — | No way to recompute or review afterwards. |

## G. Structural problems behind the "one fix per clip" pattern

1. **No single place that decides clip context.** Arm is decided 3 times (smooth.detect_bowling_arm, `_resolve_arm_and_events` retry, `_forced_events` retry), direction twice, view 6 times (`classify_view` called at orchestrator.py:405, 420, 432, 437, 524, 568, 578, 609, 700) with different inputs.
2. **Single-method events with gates that raise.** Each event has one detector; a gate either passes or throws. There is no "two methods disagree → Low confidence + needs_review".
3. **Confidence from fps bands** rather than from evidence (agreement, visibility).
4. **Constants spread across 12 files**, several documented as calibrated on one named clip.
5. **Keypoints not reusable**: a correction re-runs pose (30–90 s) instead of seconds.

## Plan (Steps 2–9) mapped to the findings

| Step | Fixes |
|---|---|
| 2 `config/pipeline.yaml` + `pipeline/params.py` (seconds, BH, percentiles) | A1–A2, A5, A9–A10, B1–B13, C1–C3, C6–C7 |
| 3 `pipeline/clip_context.py` (rotation check, direction from run-up segment, arm from excursion + speed near release, view scored with confidence, bowler scored by run-up pattern) | D1–D2, D7–D8, C4, E1–E5 |
| 4 `pipeline/consensus.py` (3 methods per event, median of agreeing, confidence = agreement + visibility + fps, needs_review) | D3–D5, B4–B5, C1, F2–F3 |
| 5 Stage runner: every stage returns result + confidence + warnings; fail only unreadable / no person / < 1 s | F1, F4–F7 |
| 6 `/recompute` from cached keypoints + Check-detection panel + corrections.jsonl | F8, G5 |
| 7 `storage/debug/{id}/` signals.png, context.json, key frames + Diagnostics tab | — |
| 8 `tests/clips` + `label_clip.py` + `regression.py` + invariance tests | guards all of the above |

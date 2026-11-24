# v4.0: perception upgrade (two pose models, limb identity, 3D lifting)

Angles were wrong on many clips for three architectural reasons:
* the pose model swapped left/right legs;
* 2D angles were foreshortened when a limb wasn't square to the camera;
* MediaPipe struggled with motion blur and gave no reliable foot points.

This release changes the architecture behind the same API and frontend.

**1. Detection and tracking.** RTMDet-m (person) runs on every frame, and ByteTrack keeps identities. When a confirmed
track has no detection (a blurred bowler at release), RTMPose runs on its predicted box. The bowler is the track with
the run-up pattern, never the biggest person.

**2. Two pose models.**
* **Primary:** RTMPose-m Halpe-26, top-down, with heels and toes. Its scores are calibrated so that OpenMMLab's 0.3
  keypoint threshold equals BowlIQ's visibility threshold.
* **Secondary:** MediaPipe on the bowler crop.
* Per-joint disagreement, in body heights, lowers confidence; missed joints are filled from the secondary.
* A ViTPose placeholder sits behind the same interface (`BOWLIQ_POSE_BACKEND=vitpose`).
* The models are OpenMMLab's official ONNX exports on ONNX Runtime (CUDA on the RTX 4060). MMPose itself isn't used,
  because its compiled `mmcv` has no reliable Windows/CUDA wheels.

**3. Limb identity (`pipeline/identity.py`).**
* **Swap correction:** per frame, a Hungarian assignment against constant-velocity predictions, with a cost-ratio
  hysteresis.
* **Duplicate suppression:** when left and right labels sit on one point and one label jumped there against its own
  motion, that label is dropped. Example: both wrists collapsing onto the blurred bowling hand.
* **Roles:** the front leg is the foot that lands at FFC, the back leg the one landing at BFC, and the bowling arm
  comes from the context. All metrics and the overlay use roles.
* Corrections are logged in `context.json` and drawn in `storage/debug/{id}/roles.mp4`.

**4. 3D lifting.** MotionBERT-lite (Human3.6M, official checkpoint) lifts the corrected 2D sequence at 50 fps in
real time. Knee angles, trunk flexion, lateral flexion, hip–shoulder separation and shoulder counter-rotation are
computed in 3D. With a single side-on clip, lateral flexion and the two rotations are now measured from 3D, where
before they needed a second clip.

**5. View-aware gating.**
* A 2D angle is used only if the limb's plane is within 20° of the picture; otherwise the 3D value is used.
* Confidence: High only when 3D and a valid 2D agree within 7°; Medium with one source; Low when they disagree.
* The source is named in `confidence_reason`, and the 2D/3D pair is in `context.angle_sources`.

**6. Foot contacts.**
* Heel/toe methods (the foot's lowest point settling, and heel-strike speed collapse) join the consensus.
* Agreement now counts independent kinds of evidence (speed / height / hip), so correlated methods can't confirm
  each other's mistake.
* Near-tied release peaks are settled by the run-up: the delivery comes at the end of the fastest run-up.

**Measured on the 14 labelled clips** (`scripts/regression.py`; the analysis code is the same for both engines):

| Pose engine | pass | flagged for review (right reason) | fail | error |
|---|---|---|---|---|
| v3.2 (MediaPipe) | 8 | 6 | 0 | 0 |
| v4.0, MediaPipe primary | 8 | 6 | 0 | 0 |
| v4.0, RTMPose primary | 5 | 3 | 5 | 1 |

* RTMPose is clearly better filmed from behind the bowler: the right arm and close key moments on two clips where
  MediaPipe had the wrong arm.
* It is worse on oblique and follow-pan clips, on the rendered synthetic figure, and on the TV replay.
* By the "only accept if the regression passes on every clip" rule it is **not** the default: `auto` behaves as in
  v3. Choose it with `BOWLIQ_POSE_BACKEND=rtm`, or `perception.auto_primary: rtmpose`.
* All the downstream improvements (identity, roles, heel/toe contacts, 3D lifting, gating) are on for every engine.

**2D vs 3D, measured on the 5 real side-on regression clips** (2D side-on is reliable there: the body's plane was
4–10° from the picture on every clip):

| Metric | 3D minus 2D, per clip (deg) | Default source |
|---|---|---|
| trunk forward flexion at release | −1.0, −0.6, 0.0, +3.8, −0.2 | `auto` (2D side-on, 3D cross-check → High) |
| front knee at FFC | −0.8, −20.6, −14.7, −14.9, −12.6 | `off` (2D only) |
| back knee at BFC | −24.3, −41.7, −12.9, −17.0, −16.6 | `off` (2D only) |

* The trunk agrees, but MotionBERT bends the knees much more than they look side-on. So 3D is switched off for the
  four knee metrics (new `off` source: the 2D value stands, as before v4.0). Re-enable it with `auto` if Kinovea
  validation shows otherwise.
* Lateral flexion, hip–shoulder separation and counter-rotation stay on 3D (`auto`) because a single side-on clip
  has no 2D value for them. These are *new* values, with Medium confidence, and are unvalidated until checked
  against hand measurements.
* Bug fixed during this check: the knee's gating plane was taken from the lifted leg itself. For a nearly straight
  leg that plane is noise (offsets of 41–89° on side-on clips), which wrongly sent knees to 3D. Gating now uses the
  body's sagittal plane, from the line of the run.

**Angle accuracy against hand measurements is not yet measured.** It needs Kinovea data. Run
`scripts/validate.py your_kinovea.csv --sources --clips-dir tests/clips` to get the MAE table for MediaPipe-2D,
RTMPose-2D and 3D-lifted per metric. Add `--write-config` to keep the best source per metric.

**3D viewer (Coach view → Diagnostics → Roles and 3D pose).**
* The roles video (front leg red, back leg blue, bowling arm green, corrected swaps marked) plays next to the lifted
  3D skeleton, which follows the video frame by frame.
* It opens as a solid body (shaded limbs, torso and head, coloured by role); the Skeleton button switches to plain
  lines. It has perspective and a floor with its shadow. Drag to rotate, or use the 3/4, Camera, Side, Front and Top
  views, or Spin. BFC / FFC / Release buttons jump to the key moments.
* A metric taken from the lifted pose now says "3D pose lifted from the video" instead of "Both clips combined".
* Data: `storage/debug/{id}/pose3d.json` (new diagnostics file, about 20 KB). Older analyses need a Recompute to get it.

**Speed.**
* On a 4-core CPU the RTMPose stack takes about 0.8–1.7 s per frame, too slow for CPU-only use.
* On the RTX 4060 the detector and pose model run on CUDA; the CPU MediaPipe cross-check then dominates. Estimate:
  roughly 1–1.5 min for 10 s at 60 fps (not measured here).
* Everything after pose takes about 8 s for 380 frames, including video, so recompute stays at seconds.
* Each analysis records its timings in `context.json`.

**7. Validation tools.**
* `scripts/validate.py --sources`: MAE per metric for MediaPipe-2D, RTMPose-2D and 3D-lifted against Kinovea, and
  `--write-config` to keep the best source.
* Stage timings are in `context.json`.

# v3.2: optional Gemini video check (Gemini watches, the GPU/CPU measures)

**Off by default.** Turn it on in `backend\.env` with `BOWLIQ_GEMINI_VIDEO=1`; it needs `GEMINI_API_KEY`.

**What Gemini does.** It watches the processed main clip once (trimmed, reduced, no sound; this is uploaded to
Google) and returns a short structured brief:
* is this a bowling delivery;
* roughly when the release happens;
* a box around the bowler;
* camera view, bowling arm and run-up direction, each with its own confidence;
* real-time or slow motion, and roughly how slow;
* problems it noticed.

**How the pipeline uses the brief.** Everything is still measured locally (pose model, key moments, angles, speeds);
the brief only gives hints, and each hint is checked:
* **Bowler:** the person Gemini points to is compared with the run-up scoring. If they match, confidence goes up;
  if not, Gemini's person is analysed, the report says so, and "Choose bowler" stays available.
* **View, arm, direction:**
  * if Gemini agrees, confidence goes up one level;
  * if Gemini is highly confident and the measurement was unsure, Gemini's answer is used at Medium confidence;
  * if Gemini disagrees with a sure measurement, the measurement is kept and the item is flagged in "Check
    detection".
  * The arm is still confirmed by the measured release: a wrong hint is overruled by a clear release with the
    other arm.
* **Slow motion:** Gemini's estimate replaces the hand-speed guess. If Gemini sees real-time footage, the clip is
  never slowed down on a guess.
* **Release:** if the measured release is more than 1 s from where Gemini places it, the report flags it.

**Where to see it.** Coach view → Diagnostics shows what Gemini said and which hints were used, agreed or
conflicted.

**Failures.** Any problem (no key, no internet, quota, bad answer) skips the check; the analysis is exactly as
without it. The brief is cached with the skeleton, so Recompute never calls Gemini again.

**Measuring it.** `python scripts\regression.py --gemini` runs the 14 labelled clips with the check on (each clip
sent once). Compare with the normal run before leaving it on.

**Tests.** 10 new ones with a fake Gemini, covering the merge rules, wrong hints overruled, conflicts flagged,
0–1000 boxes and a quiet failure.

# v3.1: skeleton and angles line up with the bowler on every clip

**What was wrong.** The video in the player is written at the clip's own frame rate, but it was being tagged with
the skeleton's rate. When a clip was treated as slow motion (chosen on upload, or guessed when the hand looked slow),
the skeleton's rate is the clip's rate × the slow-motion factor. So the player drew the skeleton and angles of a
different moment on each picture, and the mismatch got worse further into the clip.

**What changed:**
* The served videos, the skeleton file and the page now always use the clip's own frame rate. Slow motion only
  changes real-time speeds and durations.
* Videos with non-square pixels (some phone and messaging-app exports) are converted to square pixels in their
  display proportions. Before, the browser stretched the picture but not the skeleton.
* New tests check that the frame shown at any time is exactly the frame the skeleton was measured on (real time and
  ×4 slow motion), and that non-square pixels become square.
* Regression check: 14 clips, 8 pass and 6 flagged, unchanged.

# v3.0: one pipeline for any reasonable bowling clip (generalisation refactor)

Each earlier fix had been tuned on the clip that showed the problem. This release replaces those per-clip numbers
with rules that hold for any camera, frame rate, resolution, bowler and direction, and adds the tools to prove it.

**1. Audit.** `AUDIT_GENERALISATION.md` lists every clip-specific assumption that was in v2.13 (file, line, what it
assumed, why it breaks): absolute pixel thresholds, frame counts, fps bands, direction/arm/view assumptions,
"biggest person" selection and single points of failure.

**2. Scale- and fps-free maths; one config file.** Every tunable now lives in `backend/config/pipeline.yaml`, with a
comment and its unit: seconds (converted per clip as `frames = round(s × fps)`), body heights (BH = nose-to-ankle ×
1.07, median over upright frames) or fractions of this clip's own signal. Notable changes:
* Camera movement is measured as background shift in **frame widths per second** (grey difference ÷ image gradient),
  not grey levels. That's independent of resolution, fps and texture. The broadcast pan that read "steady" before
  (1.09) now reads moving (0.17 widths/s).
* Gap filling, the left/right swap fixer and ball tracking now work in seconds, not frames.
* The foot-contact reference window is max(0.1 s, the span of one smoothed speed sample). This reproduces v2.13
  exactly at 30/60/120 fps, but for a stated reason.
* The zoom and small-bowler limits are tied to the pose model's 256 px input, not to any clip.

**3. `pipeline/clip_context.py`** decides, once, with a confidence and reasons:
* **rotation:** metadata, then verified with the pose (head above ankles). Sideways or upside-down clips are turned
  upright and re-analysed.
* **direction:** from mid-hip travel over the run-up only.
* **bowling arm:** vertical swing plus peak hand speed, confirmed by the clearest release peak. For front-on and
  behind views it is cross-checked against the picture; if they disagree it is marked Low and flagged.
* **view:** from three independent cues: run-up travel across vs towards the camera, shoulder width, and projected
  stride length.
* **bowler:** every person track is scored on travel, deceleration and arm swing, never on size.

**4. Key moments by consensus** (`pipeline/consensus.py`):
* **Release:** wrist peak height, peak hand speed (+ offset), arm passing vertical.
* **FFC:** front-ankle speed collapse, ankle height reaching and staying at its low point, mid-hip deceleration onset.
* **BFC:** speed collapse and height plateau.
* Methods agreeing within 0.05 s confirm the definitional (speed / height) method.
* On disagreement, the best estimate is kept at Low confidence and `needs_review` is set with the reason.

**5. Graceful degradation.** Only an unreadable video, no person, or a clip under 1 s fail a job. Every other stage
reports `ok/detail` (`result.stages`). If a stage breaks, its measurements become not_measured with the reason and
everything else is still produced.

**6. Corrections in seconds.**
* `POST /api/v1/analysis/{id}/recompute` takes `{view, bowling_arm, direction, bowler_track_id, bfc_frame, ffc_frame,
  release_frame, height_cm}`. It reuses the cached keypoints (`cache.pkl`, no pose re-run) and redoes metrics,
  report, video and PDF in place.
* Every correction is logged to `storage/corrections.jsonl`.
* The results page has a **Check detection** panel. It opens by itself when something needs review and offers
  dropdowns, "Set as BFC / FFC / Release", "Choose bowler" with thumbnails, Recompute, and Undo.

**7. Diagnostics.** `storage/debug/{id}/` holds `signals.png` (ankle speeds, wrist height, hip speed, final moments
and each method's guess), `context.json` and the key frames. They are shown in a **Diagnostics** section in Coach view.

**8. Regression suite.**
* `backend/tests/clips/` holds 14 labelled clips with sidecar JSON ground truth.
* `scripts/label_clip.py` labels new ones frame by frame.
* `scripts/regression.py` prints the table and exits non-zero if any clip fails or gets worse than
  `tests/clips/best.json`.
* `tests/test_invariance.py` covers mirror, 30/60/120/240 fps, ×0.5/×1.5 scale, 10 % crop and 90° rotation, at
  keypoint level and on video (`-m slow`).

**Rule from now on:** change `pipeline.yaml` or the pipeline only if `python scripts/regression.py` passes on all clips.

# v2.13: no more rejected bowling videos

Every bowling clip now produces a report. Where something can't be worked out automatically the app makes its best
estimate, says so, and points to the fix, instead of stopping.

**New upload fields:** camera view (auto / side-on / front-on / behind), bowling arm (auto / right / left), and
"Is the side-on clip slow motion?" (now defaulting to *Not sure (detect automatically)*; real time; ×2 … ×32).

| Used to be an error | Now |
|---|---|
| Wrong camera angle (front-on / behind clip) | Analysed as that view: side bend, arm angle, head position, front-foot alignment, run-up line, knee caving; side-on measurements say "needs a side-on clip" |
| Slow-motion clip without a speed | Analysed with an estimated factor (the TV replay: ×16, key moments 247 / 338 / 457 vs 230–245 / 330–340 / 445–450 in the frames); real speeds/times withheld |
| Key moments not found | Best estimate from the bowling arm's release peak, every measurement capped at Low, a banner says "estimated" and the correction panel opens (with "These are right: confirm them") |
| Bowler too small | Analysed at Low confidence with a warning |

* Still impossible (and still reported): unreadable files, no person anywhere in the clip, under 1 second.
* Ran all 11 clips received so far: 11 reports, 0 rejections. Side-on clips' key moments unchanged.
* Ball speed is reported only for side-on, real-time (or stated slow-motion) clips with detected key moments.
* Tests: 121 backend, 5 frontend.

# v2.12: new clips no longer rejected with "wrong camera angle"; slow motion; finish any clip by marking

**Why almost every new clip was rejected:** when the camera follows the bowler (broadcast replays, a friend
panning the phone), the bowler never crosses the frame, so the camera-angle check returned "unknown". Then, whenever
anything else failed (often the slow motion), the app reported "unknown" as **"filmed from in front or behind"**.
* **Side-on is now recognised from body shape** when the camera follows the bowler: a thin shoulder profile (the
  replay: 0.11 of torso length) with the feet spread wide across the picture in the stride (0.93 body lengths).
* **"Wrong camera angle" only when the angle is clearly front-on or behind** (your three behind-the-bowler clips
  still get it, correctly); otherwise the real reason is shown. The footprint-tilt rejection is skipped when an
  uncorrected following camera makes it meaningless.
* **Run-up from the legs:** with a following camera the hips don't cross the picture, so a running gait (feet
  sweeping forward and back relative to the hips) also counts as a run-up.
* **Scene cuts** (screen recordings that loop, highlight reels): only the shot with the bowler is analysed. The
  replay looped at frame 499.
* **Slow motion:**
  * Detected from the speed of the bowling hand (real-time clips: 9–18 body heights/s; the TV replay: 1.6), which
    now says "this looks like a slow-motion clip, choose its speed" instead of failing with a wrong reason.
  * New **"Is the side-on clip slow motion?"** choice on upload (×2 … ×32); with it every timing rule and speed
    works in real time. The replay with ×16: key moments 247 / 338 / 457 vs 230–245 / 330–340 / 445–450 in the frames.
  * The factor isn't guessed: release came out the same at every factor, but the foot landings didn't (×8 and ×32
    were badly wrong), so a guess would give confident wrong results.
* **Finish any fixable failure by hand:** when the key moments can't be found (or the clip is slow motion / the
  angle looks wrong), the error page shows **"Finish it by marking the key moments yourself"**. Step through the
  processed clip, set back foot / front foot / release, choose the slow-motion speed, and the analysis completes.
  The clip is kept 24 hours for this (`GET /jobs/{id}/video`, `POST /jobs/{id}/mark`).
* Checked on all your earlier clips: key moments unchanged.
* Tests: 120 backend (new: slow-motion clip → explained → finished by marking; camera following the bowler is
  side-on), 5 frontend.

# v2.11: angles on the front-on / behind video

* The second clip in **Linked playback** now shows the angles that view can measure:
  * **bend:** side bend of the trunk from vertical;
  * **arm:** bowling arm from vertical, shown while the arm is up (around release);
  * **sep:** the shoulder and hip lines drawn with the angle between them, hidden when either line is seen end-on
    (the same 0.35 × torso rule the measurements use).
* The same angles are drawn on second-clip frames in "How we measured this".
* These are flat 2D angles of the picture; the report's hip–shoulder separation and counter-rotation use true 3D from
  both clips when they're synced.
* Tests: 118 backend, 5 frontend.

# v2.10: key moments fixed on your five new clips (release-first search)

Five 30 fps WhatsApp clips from other bowlers. Three are side-on; two are filmed from behind and are correctly refused
as the main clip. Every result was checked frame by frame:

| Clip | Frames show (BFC / FFC / release) | Before | Now |
|---|---|---|---|
| 1 | 30 / 35 / 39–40 | 16 / 23 / 26, **left** arm | **30 / 34 / 39**, right arm |
| 2 | ~77 / ~80 / ~86 | not analysed ("no bowling action") / later 78 / 85 / 86 | **78 / 81 / 86** |
| 4 | 47 / 52 / 55 | 30 / 47 / 49, **left** arm | **47 / 50 / 56**, right arm |

What was wrong and what changed:
* **The bound was taken for the delivery.** In the bound both hands are above the head, so the old search
  ("highest wrist within 150 ms after a front foot contact") picked it, with the wrong bowling arm, whenever a
  contact was found a few frames early. **Release is now found first**: the bowling hand rises ≥ 25 % of body height
  in the 250 ms before it and drops ≥ 25 % in the 250 ms after (the follow-through). The bound is a plateau and
  fails this.
  * Front foot contact is then the front foot's first landing 40–300 ms before release, and back foot contact the
    other foot's last landing 80–600 ms before that.
  * The old search remains only as a fallback, and a release that passes the new checks always beats one from it.
* **Legs labelled the wrong way round for the whole clip** (clip 1). The front foot is now the one further forward at
  release, whatever the pose model calls it, and the leg labels are corrected to match. Previously the back foot's
  landing was reported as front foot contact.
* **Gradual stops at 30 fps:** approach speed is measured over ≥ 6 frames (was 3, i.e. 0.1 s at 30 fps), and the
  "one contact per ~0.1 s" rule has its own window (it was suppressing real contacts).
* **Panning over plain sky and grass** (clip 1): the frame-difference check read 2.5 for a real 37 px pan, below the
  old 4.0 cut-off, so the camera path itself now decides (steady clips still read ~0.6 and are left alone).
* Earlier clips unchanged or better: clip 11.29.51 251 / 259 / 267 (unchanged), WhatsApp clip 71 / 81 / 85
  (unchanged), handheld-follow copy front foot 258 (was 265).
* Known limit: on some clips front foot contact is still 2–3 frames early or late (heel strike vs foot flat). Use
  "Correct them" on the results page when it matters.
* Tests: 118 backend (new: the bound with both hands up, legs labelled the wrong way round).

# v2.9: correct the key moments by hand

* **"Key moments don't match the video? Correct them"** under the results player. Step frame by frame to the back
  foot landing, the front foot landing and the release, press "Set" for each, then **Re-analyse with these
  moments**. The delivery is analysed again with your frames (`POST /api/v1/analysis/{id}/reanalyze`).
  * The already-processed video is reused, so frame numbers stay identical (checked by pixel comparison) and no
    re-upload is needed.
  * The same bowler is kept (tapped automatically at your release frame), and the bowling arm is re-chosen at your
    release.
  * Every measurement uses your frames, and the report says they were set by hand. The original analysis is kept.
  * Moments must be in order, 0.03–0.6 s from back foot to front foot and 0.02–0.4 s from front foot to release.
* Fixed while testing it: the leg-label correction step re-detected the key moments and could overwrite the
  hand-set ones (seen on clip 11.29.51).
* Checked (no change needed):
  * Playback alignment: jumping to a key moment in the browser shows exactly that frame (0 frames off, by pixel
    comparison).
  * The GPU engine (YOLO) gives the same key moments as the CPU engine on clips 11.29.51, the WhatsApp clip and the
    sample.
* Known limit (measured frame by frame on clip 11.29.51): automatic back foot contact is ~3 frames (50 ms) late
  (detected 251, foot visibly lands at 247–248), front foot within 1 frame, release correct. A "first touch"
  refinement fixed this clip but put contact 2–4 frames early on the synthetic bowler with known contacts, so it
  wasn't shipped; use the correction tool, and send clips where detection is wrong so the rule can be tuned on them.
* Tests: 116 backend, 5 frontend.

# v2.8: Phase 1c (adapting to imperfect video)

Every adaptation was tested on real clips made imperfect in a known way (so the right answer is known) and kept
only if it brought the result back to the original's.

* **Camera movement correction** (`pipeline/adapt.py`). Background points are tracked between frames (all people
  masked out); a rotation + scale + shift is fitted per frame (implausible steps rejected), chained over the delivery
  only (± 0.5 s, to keep drift small), and the skeleton is measured in one fixed reference view. The video and its
  skeleton overlay stay as filmed.
  * Only used when the camera really moves (same threshold as the quality check); steady clips are left alone.
    Before this gate, a steady clip got a bogus 31 px "correction" from accumulated fitting noise.
  * **Self-check:** a planted front foot can't move, so the correction is kept only if that foot is at least as
    still in corrected coordinates as in the raw picture. Measured: 3.8 % vs 20 % of body height on your panning
    WhatsApp clip, 0.5 % vs 5.9 % on a simulated follow-pan.
  * Results:
    * a simulated "follow the bowler" handheld copy of clip 11.29.51 failed completely before ("No run-up was
      found"); now back foot, release and run-up speed match the steady original (6.44 vs 6.4 m/s);
    * your WhatsApp clip, which really pans about a frame width, now gets a run-up speed (8.3 m/s) instead of
      "not measured".
  * The quality check judges the movement left over after correction; ball speed still needs a steady camera.
* **Small, far or dark bowler.**
  * If the full frame finds too little, four enlarged, overlapping quarters of the picture are searched.
  * Then the skeleton is found again on a zoomed-in crop around the bowler. This runs always when the bowler is
    under 120 px tall, and otherwise only as a rescue when the delivery can't be found.
  * Results:
    * bowler ~90 px tall: back foot 7 frames off and a bogus 56° tilt without it; with it, back foot within 2 frames
      and run-up speed 6.1 vs 6.4 m/s;
    * darkened clip: the delivery couldn't be found without it; with it, the key moments match the original;
    * not used on clips that already work, because on a 30 fps, 148 px copy it made the result worse.
  * **Too small to trust:** under 70 px tall (measured: ~55–64 px gave key moments 15–19 frames off even when
    zoomed) the clip is refused with `BOWLER_TOO_SMALL` and "film closer".
* **Tried and dropped (measured, not shipped):**
  * contrast enhancement for dark clips: with it the zoomed pass picked the wrong arm, ~20 frames late; without it
    the dark clip comes out right;
  * key-moment times between frames at 30 fps: mean error 16.7 ms with it vs 16.2 ms without, with contacts
    biased 13–20 ms early.
* `BOWLIQ_ADAPT_VIDEO=false` turns all of this off (for comparison).
* Known limit seen again: on some clips front foot contact can land at heel strike or a few frames later at foot
  flat (up to ~6 frames at 60 fps); this comes from the contact rule, not the corrections.
* Tests: 114 backend, 5 frontend.

# v2.7: second-clip fixes from your screenshots, Phase 4 (true 3D rotation) and Phase 5 (verdicts across deliveries)

**Fixes from the screenshots**
* **Wrong skeleton in the second video.** At the frame shown, the bowler was entering at the picture's edge and
  the pose model put his hidden shoulder at x = 1 px. Joints pinned to the frame edge now count as not visible, so
  no line is drawn or measured through them.
* **"Front-on phone wasn't recording yet".** Linked playback now plays and scrubs only the stretch both phones
  recorded. The message names "the second phone", and an unsynced pair says "Not lined up yet" instead.
* **A false sync was found and closed.** On your two clips (side-on 11.29.51 and behind 11.29.47), the sound onsets
  (−1.970 s) and the release moments (−1.94 s) agreed by coincidence. The body's up-and-down movement doesn't match
  at that shift (0.52, against 0.96 for a true pair) and the follow-throughs differ, so they are two different
  deliveries. New **body-motion check**: vertical movement looks the same from any horizontal direction, so the
  bowling wrist's and head's heights (in body heights) must match between the clips.
  * A sync that isn't a clean sound match (sound onsets plus release, or release alone) must also pass it (≥ 0.7).
  * New **"motion" sync** when there's no usable sound: a single clear match ≥ 0.85, beating every other shift by
    ≥ 0.1. On the true pair with its sound removed it found −0.825 s against the true −0.800 s (about 1.5 frames).
  * When the waveforms differ but the onsets, the release and the body motion all agree, the clips are synced on
    sound at Medium confidence.

**Phase 4: true 3D trunk rotation from both clips** (`pipeline/three_d.py`)
* With the clips synced and the second clip front-on or behind, the side-on clip gives each shoulder/hip line's
  component along the pitch and the second clip the component across it. Their angle in the horizontal plane is the
  real rotation, even where the shoulders overlap in one view.
* Hip–shoulder separation at BFC and shoulder counter-rotation (the two 30° load flags) now use 3D when it's
  available (view "3D (both clips)"; the single-camera value is kept for reference) and fall back to the flattened
  single-camera values (capped at Medium) otherwise.
* The second phone is sampled between its frames at the exact synced time (phones at 30 fps and trunk rotation at
  ~200°/s would otherwise cost degrees).
* How far the second camera is turned from the line of the run is measured from the run-up and undone: the bowler
  runs straight, so any sideways drift in the second view compared with the forward travel in the side view is the
  camera's turn.
* Checked against a 3D bowler with known rotations (30 fps second phone, perspective, pixel noise): within 3° with
  the camera square, 8° or 20° off square (before the turn correction, 8° off square cost 5°). A run-up that isn't
  straight biases this estimate; refused beyond 25° (that's a diagonal clip). Label flips between frames → refused
  with a reason.

**Phase 5: verdicts across several deliveries** (`analysis/consistency.py`, `GET /api/v1/bowlers/{name}/summary`)
* For every metric with a rule: in how many of the last 5 / 10 / 20 deliveries it was flagged. Only Medium/High
  confidence measurements count; Low ones are listed as not counted. A verdict needs at least 3 measurements.
  * **Consistent** ≥ 60 %, **occasional** 25–60 %, **one-off** < 25 %.
  * Ranked by consistency, then the rule's weight.
* Coach-style summary ("Front knee bends after landing in 5 of 6. Work on this first: <drill>"), drills and cues
  from the coaching notes for repeated issues only, a link to the clearest example delivery, "consistently good"
  metrics, ball-speed median and best, and personal trends (older vs newer half) for the trend-only measurements.
  With fewer than 5 deliveries it says these are early signs, not verdicts.
* On the Bowler profile page.
* Tests: 110 backend, 5 frontend.

# v2.6: Phase 3 (each camera angle measures what it can; the two clips combined)

* **Camera angle detection** for both clips: side-on, front-on (behind the stumps, facing the bowler), behind the bowler,
  or diagonal. It uses the run-up: how much the bowler crosses the frame, how much a posture-robust body length (thigh
  + shin + trunk) grows or shrinks, and how square the shoulders look. Calibrated on the real clips: side-on 0.97–1.00
  (size late/early), the 19°-oblique WhatsApp clip 0.57 (diagonal), the behind-the-bowler clip 0.33 (behind).
  * Main clip: side-on or diagonal accepted (diagonal → Medium quality at best, and the existing ground-tilt rules
    withhold distances along the pitch); front-on / behind rejected as before.
  * **Uploaded the wrong way round?** If the first clip is front-on/behind and the second side-on/diagonal, they are
    swapped automatically (with a note).
  * A second clip that is also side-on is reported as adding no new view, instead of producing wrong "front-on"
    numbers.
* **New side-on measurements (trend only, no research thresholds):** head ahead of the front foot at release, head bob
  in the last strides, back knee angle at BFC, front leg plant angle at FFC, release point ahead of the front foot,
  front foot landing angle (heel/toe; CPU engine), run-up speed kept at FFC, front-arm pull-down timing, bowling
  elbow straightening (always Low: a 2D trend, never a legality judgement).
* **New second-clip measurements (front-on or behind):** head falling away at release, front foot alignment
  (open/closed), bowling arm angle from vertical, front knee caving inward, run-up line drift.
  * Left/right comes from the model's anatomical hip labels, so values have the same sign from in front or behind.
  * Diagonal second clip: only side bend, head falling away and arm angle, at Low confidence. Rotation, foot
    alignment, knee caving and drift need the camera on the line of the run and say so.
  * Every metric has a method description and a "not measured" reason (hidden joint, overlapping hips/shoulders,
    moving camera, oblique view, missing heel/toe points…).
* **Combining the two clips.**
  * With a confident sound sync, the side-on key moments (timed best) are carried into the second clip exactly;
    second-clip detection from front/behind was the weak point (it found the gather instead of back foot contact on
    the real behind-view clip).
  * If the second clip's own detection agrees (within 2 frames), both clips' key moments go **up one confidence
    level** ("confirmed by both clips"). If it disagrees, the report says so and uses the side-on moments.
  * Without sync, the second clip's own moments are used only if they pass the timing cross-check.
* **"Show me" per video:** findings and measurements from the second clip open that clip's frame (with its skeleton)
  in "How we measured this" and jump Linked playback to that exact frame. The metrics table and dialogs name the
  view (Side-on / Front-on / Behind bowler / Diagonal). `frame/{n}?view=front` serves second-clip frames.
* The headline coverage ("N of M metrics measured") no longer counts second-clip metrics when no second clip was
  uploaded.
* **Validation status (honest):**
  * New measurements are tested on skeletons with known geometry: exact values, same result from in front and from
    behind, diagonal and overlap rules.
  * Checked against the picture on real clip 11.29.51: front leg 49° ahead of vertical in a long stride, toe-up heel
    strike ~27°, straight back leg 175° at BFC, all visible in the frames.
  * The roadmap's "within ~5° of careful manual measurements" for the second-clip metrics still needs real two-phone
    recordings of the same delivery; none exist yet.
* Tests: 96 backend, 5 frontend.

# v2.5: Phase 2 (the two clips lined up in time + linked playback)

* **Automatic sync from sound.** The two phones start recording at different moments; BowlIQ now finds the time shift
  between them from their soundtracks (`pipeline/sync.py`). Two independent matchers must agree within 4 ms:
  GCC-PHAT on the waveforms and cross-correlation of sound onsets. The rest of the soundtracks (footsteps,
  background) must also still line up with the matched sound blanked out, so two unrelated clips that each contain
  one clap can't be paired.
  * Calibration: 0 false syncs in 40 random unrelated pairs and on every pairing of your real clips. Correct within
    2 ms on 39 of 40 simulated two-phone recordings, and within 5 ms on 30 of 36 much harsher ones (room echo, AAC
    re-encode, real background noise). The rest fell back to the release moment.
  * Real-clip check: clip 11.29.51 paired with a copy that starts 0.80 s later and has a re-recorded soundtrack
    (echo, duller mic, noise) synced at −0.7999 s, and every frame of the delivery matched (0 frames off, by pixel
    comparison).
* **Backup: release moment** (±1–2 frames) when the sound can't be matched; **none** with a clear reason otherwise.
  If the sound sync is confident but the front clip's own release is > 0.1 s away from it, the front clip's key
  moments are not trusted (its metrics show "not measured" with the reason).
* **Linked playback** on the results page: side-on and front-on clips side by side, frame-locked through the sync,
  skeleton on each, play / frame step / jump to back foot, front foot, release, speed 0.1–1×. Shows which method
  synced the clips and how confident it is. **Nudge** the front clip ±1/±5 frames and **Save alignment**
  (`PUT /api/v1/analysis/{id}/sync`); "Use automatic" goes back.
* New media: `video?view=front` (overlay-free, fast-seeking) and `keypoints?view=front`. The result has
  `front_video` and `sync`.
* **Timing fix found while validating sync (affects every variable-frame-rate phone clip, i.e. most WhatsApp
  videos).**
  * Problem: preprocessing resampled them to their *average* rate (e.g. 58.2 fps). Measured by pixel-matching
    against the originals, frames during the delivery were shown 19–35 ms (1–2 frames) late.
  * Fix: they are now converted at their nominal rate (59.94), placing each frame at its true time. All real frames
    are kept; one is repeated only where the phone skipped one. Timing error during the delivery is now −1 ms
    (clip 11.29.51) and −7 ms (WhatsApp clip).
  * Side effect: front foot contact on clip 11.29.51 is now right (manual 248–249 in original frames; was 252, the
    old "3 frames late" limit). On the WhatsApp clip back foot and front foot each moved by one frame (now 69/74/83
    vs manual ~67/75/83).
  * Frame numbers of VFR clips analysed after this update differ from older analyses of the same clip (59.94 vs
    58.2 frames per second); times in seconds are comparable and now more accurate.
* Filming tip on the upload page: clap once, loudly, after both phones start recording, about the same distance from
  each (sound travels ~3 ms per metre).
* Tests: 87 backend (sync: offset recovery, unrelated soundtracks, disagreement, missing sound, audio/trim
  alignment, manual alignment API; VFR timing regression), 5 frontend.

# v2.4: Phase 1 + 1b (always the right bowler, correct legs and feet)

**Phase 1: only the bowler is analysed**
* **Tap the bowler** (upload step 2, "Who is the bowler?"): scrub to any frame and click the bowler. The tracker starts
  from the tapped person (searching ±0.3 s around the tapped frame) and follows only them. Tapping empty ground gives
  `BOWLER_NOT_FOUND_AT_TAP` instead of silently picking someone else. A front-on clip has its own picker.
* **Ignore zones:** drag boxes over the umpire, batter, fielders or spectators (up to 8). People whose body centre is
  inside a box are never considered.
* **Appearance memory:** every detection carries a colour signature of the torso and upper legs; the tracker refuses to
  jump to a person who looks different, even when they're right next to the bowler.
* **Ranked candidates:** without a tap, the three people most likely to be bowling (steady run across the frame) are
  tried in turn; the next one is used only if no bowling action is found for the first, and the report says so.
* **Must look like bowling:** the candidate needs a run-up, a front-foot landing and the hand clearly above the head at
  release. A clip filmed from behind/in front (footprint line tilted > 35°, or classified front-on) is rejected with
  `WRONG_CAMERA_ANGLE` instead of producing wrong numbers.
* API: `/analyze` accepts `side_tap` / `front_tap` (`t,x,y`: seconds into the uploaded clip, x/y as 0–1 fractions of the
  frame) and `side_ignore` / `front_ignore` (`x0,y0,x1,y1;...`). Values are range-checked; a tap outside the trimmed part
  or a zone covering the whole frame is refused with a clear message.

**Phase 1b: correct legs, feet and angle display**
* **Heel and toe points** (MediaPipe) are tracked as 4 extra joints (21 total) and drawn on the skeleton; they swap
  together with the knee and ankle when a left/right leg mix-up is fixed. (The GPU engine has no foot points; they show
  as missing there.)
* **Front-knee angle only where it means something:** from back foot contact to just after release, and not while the
  legs cross (when "front leg" is ambiguous). This stops the swapped/jumping knee numbers seen during the run-up.
  Applies to the results player, Compare and the rendered annotated video.
* **Tried and rejected (measured on the real clips, so not shipped):** using MediaPipe's depth estimate to decide which
  leg is nearer the camera (its leg depth flips sign side-on and created new swaps), and timing foot contact from the
  heel instead of the ankle (no improvement: front foot 75 vs ankle 76 vs manual 76; back foot got worse).
* Validation on real clips (manual frame counts): WA clip BFC/FFC/release 69/76/84 vs ~68/76/84 (also when the
  bowler is tapped or the bystander is shaded; tapping the bystander is correctly rejected); clip 11.29.51
  245/256/261 vs 245/252-253/260-261; rear-view clip 11.29.47 rejected with `WRONG_CAMERA_ANGLE`.
* Tests: 78 backend (10 new for tracking and guidance parsing), 5 frontend.

# v2.3: Compare playback fixed

* **Compare videos no longer freeze.** Playback used to fire a new seek on both videos ~15×/s without waiting, so seeks
  queued up and the videos and skeletons froze. Now: seeks are coalesced (a new target waits for the previous seek),
  the skeleton is drawn for the frame actually on screen, playback follows the clock and skips frames it can't
  decode in time, and clips are aligned on release in real time (different frame rates stay in step). Speed buttons
  0.1× / 0.25× / 0.5× / 1× added.
* **Fast seeking:** the interactive (overlay-free) video is encoded with a keyframe every 8 frames and no B-frames
  (was ~250), so frame-accurate seeks are near-instant (also helps frame stepping on the results page).
* `python scripts/reencode_videos.py` converts analyses made before this update (skips ones already done).
* Tests: 68 backend, 5 frontend.

# v2.2: fixes from two more real clips

* **Empty lead-in no longer fails the clip** (`POSE_NOT_FOUND … 19% of frames`): only the stretch where the bowler is
  on screen is analysed, visibility is judged inside that stretch, and the report says which seconds were used.
  The bowler is also found when nobody moves in the first half of the clip.
* **Clear error for the wrong camera angle** (`WRONG_CAMERA_ANGLE`) instead of a generic "couldn't find the back
  foot" when a clip is filmed from behind or in front of the bowler.
* **Front foot contact must come ≥ 40 ms before release** (the GPU engine had picked a "landing" 1 frame before
  release on one clip; it now finds 253 vs manual 252–253).
* **Stride is footprint-to-footprint** (settled ankle positions), so it no longer jumps when the contact frame is
  detected at heel strike vs a few frames later (engines now agree within ~3 % instead of ~11 %).
* Camera-angle classification uses only frames where the bowler is visible.
* **Second (front-on) clip safeguards**: shoulder/hip lines seen end-on (points overlapping in the picture) are not
  measured, rotation is folded to 0–90° (a line has no direction), and the second clip's back-foot→front-foot time
  must match the side-on clip or its measurements aren't reported. This removed an impossible 122° "counter-rotation"
  (and a false load flag) from a behind-the-bowler clip.
* Tests: 67.

# v2.1: NVIDIA GPU support

* **GPU pose engine** (optional): Ultralytics YOLO26x-pose on CUDA, batched FP16, behind the existing `PoseEstimator`
  interface (`app/pipeline/pose_yolo.py`). Pinned weights URL + SHA-256. `BOWLIQ_POSE_BACKEND=auto` uses it when a CUDA
  GPU is present; it falls back to MediaPipe automatically when it can't find the bowler in a clip.
* **NVENC encoding**: preprocessing and both rendered videos use `h264_nvenc` when a trial encode succeeds, else
  libx264 (`BOWLIQ_VIDEO_ENCODER`). ffmpeg inputs in preprocessing now also use the protocol whitelist.
* **Robustness fixes found while validating the GPU engine on a real clip** (these also improve the CPU engine):
  * foot contact compares against the approach speed over the previous 0.1 s instead of only 3 frames (gradual stops
    were missed), and allows realistic heel-to-flat settling (1.0 body height/s);
  * both bowling-arm hypotheses are evaluated; the arm that is vertical above its own shoulder at release wins (a
    dipped head in the follow-through or a high front arm in the coil fooled the old rule);
  * "hand above the head" uses eyes/ears/shoulders when the face is hidden;
  * oblique-camera limits tightened (Low confidence above 8° ground tilt, not measured above 15°, no ball speed above 8°),
    because a 19° tilt was close enough to the old 20° edge to flip results between runs.
* `/health` reports the pose engine, device and video encoder; Settings shows them.
* Tests: 62 backend tests (new: engine fallback, real gradual-stop ankle track, hidden-face release).

# Audit fixes (v2.0)

Every finding from `AUDIT_REPORT.md`, with how it was fixed and how it is verified.
Verification: **57 backend tests** (pytest), **5 frontend tests** (`npm test`), `ruff` clean, `bandit` 0 issues,
`npm audit` 0 vulnerabilities, a full real-browser run (sign-in → sample analysis → signed video/frame/PDF → profile →
compare → delete), and both real clips re-run.

| ID | Fix | Where | Verified by |
|---|---|---|---|
| C1 | Bearer-token auth on every route except `/health`; tokens via `BOWLIQ_API_TOKENS` or auto-generated; constant-time compare; sign-in page; short-lived HMAC-signed media links for `<video>`/`<img>`/PDF | `core/security.py`, `core/config.py`, `api/routes.py`, `frontend/app/login`, `lib/api.ts`, `lib/useMediaLinks.ts` | `test_auth_required`, `test_signed_media_links`, e2e |
| C2 | `owner_id` on jobs and analyses; every query scoped to the owner; 404 for others' ids | `models/db.py`, `api/deps.py`, `api/routes.py`, `core/jobs.py` | e2e owner-isolation asserts |
| H1 | ASGI body-size limit (Content-Length check + streaming counter) before multipart parsing; disk-free check | `core/security.py`, `core/storage.py` | `test_oversize_body_rejected_before_reading` |
| H2 | Per-user rate limits (analyze/read/feedback) and per-user queue cap → 429 | `core/security.py`, `api/routes.py` | `test_rate_limiter` |
| H3 | `TrustedHostMiddleware` (`BOWLIQ_ALLOWED_HOSTS`) | `main.py` | `test_untrusted_host_rejected` |
| H4 | Origin check + required `X-BowlIQ-Client` header on POST/PUT/PATCH/DELETE (plus Bearer auth) | `core/security.py` | `test_cross_origin_and_missing_client_header_blocked` |
| H5 | Sample saved under reserved "Sample (synthetic)"; never used for baselines | `lib/useSample.ts`, `core/jobs.py`, `orchestrator.py` | `test_sample_bowler_never_gets_a_baseline` |
| H6 | Blocking ffprobe/OpenCV/DB work moved to the threadpool; async file writes | `api/routes.py` | e2e |
| H7 | SQLite job queue with atomic claim + lease/heartbeat; standalone `python -m app.worker`; only expired leases become INTERRUPTED; queued jobs survive restarts | `core/jobs.py`, `app/worker.py` | `test_claim_is_exclusive`, `test_interrupted_only_when_lease_expired` |
| H8 | Scale = mean of upright-frame and posture-robust segment-sum estimates; uncertainty reported (`scale.rel_uncertainty`) and used to cap confidence | `pipeline/scale.py`, `pipeline/metrics.py` | sample clip: stride 87.3 → 81.3 % (true 80.6 %), ball 136 → 127 km/h (true ~122) |
| H9 | History returns `{value, unit}`; profile plots one unit per chart and says how many deliveries were skipped | `routes.history`, `schemas.MetricPoint`, `app/profile` | e2e history assert |
| M1 | Generic 500s with error ids; ffmpeg stderr only in logs; `/health` without paths | `main.py`, `core/ffmpeg.py`, `core/jobs.py`, `routes.health` | `test_health_is_public_and_has_no_paths` |
| M2 | Finite-number validation; `json.dumps(allow_nan=False)` | `routes.parse_float`, `quality.resolve_trim`, `core/jobs.py` | `test_nan_and_inf_rejected`, `test_params_persist_as_strict_json` |
| M3 | Versioned model URL + pinned SHA-256 (verified on load, https only) | `core/config.py`, `pipeline/pose.py` | health `model_file` check |
| M4 | `-protocol_whitelist file,pipe` on every ffprobe call; MP4/MOV signature check before parsing; Docker worker non-root/read-only | `core/ffmpeg.py`, `Dockerfile`, `docker-compose.yml` | `test_non_mp4_bytes_rejected_by_signature` |
| M5 | API: nosniff, DENY, no-referrer, CORP, CSP on JSON. Frontend: CSP, frame-ancestors, nosniff, Referrer/Permissions-Policy, no X-Powered-By | `core/security.py`, `frontend/next.config.ts` | `test_security_headers`, curl |
| M6 | Raw uploads deleted after preprocessing; retention setting; orphan cleanup; disk check; catch-all cleanup on failed uploads | `orchestrator.py`, `core/maintenance.py`, `core/storage.py` | folder listing after a run |
| M7 | Versioned migrations (`PRAGMA user_version`), FK `ON DELETE CASCADE`, `PRAGMA foreign_keys=ON`, unique feedback, composite indexes; `result.json` duplicate removed | `models/db.py` | migration of an old v1 database tested |
| M8 | Timezone-aware columns; `iso_utc()` always emits an offset | `models/db.py`, `api/routes.py` | e2e |
| M9 | Event detection tries the next landing when a plausibility gate fails | `pipeline/events.py` | events tests |
| M10 | Whole-clip leg check (front ankle must lead at FFC) → swap labels and re-detect | `pipeline/smooth.py`, `orchestrator.py` | pipeline runs |
| M11 | Gemini: 15 s timeout, one jittered retry on 429/5xx, pinned dated model | `report/generator.py` | generator tests |
| M12 | E2E now fails if the sample doesn't analyse; new hardening, range, recovery, verdict and quality tests; frontend tests | `tests/`, `frontend/tests` | itself |
| M13 | Player draws the canvas from refs in the frame callback; React state ≤ 10 Hz | `components/VideoPlayer.tsx` | browser run |
| M14 | Object URLs revoked; thumbnails as blob URLs (revoked on unmount) | `app/new/page.tsx` | code |
| M15 | Exact pins; dev split; SHA-256 lock files for Windows and Linux (installed with `--require-hashes` in a clean venv; 57/57 tests pass) | `requirements*.txt`, `*.lock` | clean-venv install |
| M16 | Request ids, structured key=value logs, access log, audit log for deletes, `/metrics` (Prometheus), pose device probed once at startup | `core/logging_setup.py`, `core/observability.py`, `main.py` | curl |
| M17 | Dockerfiles (non-root, pinned, healthcheck), docker-compose (read-only, cap_drop), Caddy TLS example, CI workflow, backup/restore tool | repo root, `deploy/`, `.github/`, `app/cli/backup.py` | — |
| M18 | No focusable element nested in buttons; `--fa` contrast ≥ 4.5:1; shortcuts only while the player is focused; dismissible toast; accessible confirm dialog | `components/*`, `globals.css` | browser run |
| M19 | Ball-speed ± includes the scale uncertainty (95 %) | `pipeline/ball_speed.py` | e2e: ± covers the true speed |
| M20 | Variable-frame-rate detection → warning, lower quality level (frames still resampled to CFR) | `pipeline/quality.py` | `test_quality_low_fps_and_vfr_warn_but_continue` |
| L1 | Unused imports/variables removed; strengths sort fixed; tilt computed once | various | ruff clean |
| L2 | Encoders always reaped; original error not masked; temp files closed | `report/render_video.py` | code |
| L3 | No `warnings.catch_warnings` (thread-safe masked reductions) | `pipeline/quality.py` | code |
| L4 | Signed numbers must match signed source values | `report/generator.py` | generator tests |
| L5 | Single `resolve_trim()` for route and pipeline | `pipeline/quality.py` | tests |
| L6 | `limit`/`offset` on lists; history limit; SQL-side history for baselines | `api/routes.py`, `core/jobs.py` | e2e |
| L7 | One feedback row per finding (unique constraint, update in place) | `models/db.py`, `routes.feedback` | e2e |
| L8 | Upload timeout + cancel, polling backoff, no theme flash, per-page titles, no API calls before sign-in | `lib/api.ts`, pages, `layout.tsx` | browser run |
| L9 | API versioned under `/api/v1` | `api/routes.py`, `lib/api.ts` | all tests |
| L10 | Tooltips open on tap | `components/Tooltip.tsx` | code |
| L11 | `--no-server-header` in all run commands and Docker; `poweredByHeader: false` | docs, Dockerfile, `next.config.ts` | curl |
| L12 | `Settings.from_env()` (constructible in tests) | `core/config.py` | tests |
| L13 | Script logic moved to `app/cli/`; `scripts/*.py` are thin wrappers; no top-level `scripts` package | `app/cli`, `scripts` | `test_validate_script` |
| L14 | Engine creation behind a lock | `models/db.py` | code |

## Behaviour changes to know about

* You must **sign in** (token printed on first start; saved in `backend/storage/.api_token`).
* The API moved to **`/api/v1`**.
* The bowling-arm, contact-settling, oblique-camera and planted-foot checks from the real-clip testing are kept.
* Distances and speeds are now about 7 % lower than before on the same clip (the old scale under-estimated body height).
* Databases from the previous version are migrated automatically on first start; existing deliveries belong to the
  owner id `owner` (the auto-generated single-user token). If you later switch to named tokens, include an
  `owner:<token>` entry in `BOWLIQ_API_TOKENS` to keep access to them.

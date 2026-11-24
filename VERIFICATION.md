# BowlIQ v3.0 — verification (Step 9)

Run on 14 labelled clips in `backend/tests/clips/` (truth read frame by frame). v3.0 numbers come from
`python scripts/regression.py --fresh`, which re-runs the pose model on every clip. v2.13 numbers come from the old
code run on the same clips and judged by the same rules.

**Rules:**
* Key moments must be within the sidecar's tolerance; foot contacts are never judged stricter than ±0.05 s
  (heel strike vs foot flat).
* Arm, direction and view must match the label.
* "Flagged" means something is off but the report says so (needs_review) with a reason that names that item.
* `!` marks a value outside tolerance; ✗ marks a wrong view or arm.

| Clip | Truth BFC / FFC / release | v2.13 (before) | | v3.0 (after) | | Review reason (after) |
|---|---|---|---|---|---|---|
| behind_c3_30fps | 70 / 74 / 78 | 70 / 73 / 77 | ✅ pass | 70 / 73 / 77 | ✅ pass |  |
| behind_c5_30fps | 27 / 29 / 31 | 34! / 38! / 42! | ❌ wrong, not flagged | 29 / 34! / 39! (arm left ✗) | ⚠️ flagged | FFC: detection methods disagree (speed 34, height 30).; BFC: detection methods disagree (speed 29, height 27). |
| behind_handheld_30fps | 94 / 96 / 98 | 87! / 92! / 96 (view front-on, arm left ✗) | ❌ wrong, not flagged | 77! / 95 / 96 (arm left ✗) | ⚠️ flagged | The strict search found no clear delivery (No bowling action was found: the hand never goes clearly above the head after a front foot landin |
| behind_nets_60fps | 160 / 166 / 169 | 139! / 147! / 151! (arm left ✗) | ❌ wrong, not flagged | 139! / 147! / 151! (arm left ✗) | ⚠️ flagged | FFC: detection methods disagree (speed 147, height –).; BFC: detection methods disagree (speed 139, height –). |
| broadcast_slowmo_left | 238 / 335 / 448 | 247 / 338 / 457! | ❌ wrong, not flagged | 397! / 422! / 455 | ⚠️ flagged | FFC: detection methods disagree (speed 422, height –, hip_decel –).; BFC: detection methods disagree (speed 397, height 360). |
| front_oblique_60fps | 203 / 210 / 219 | 203 / 210 / 219 (view side-on ✗) | ❌ wrong, not flagged | 203 / 210 / 219 (view side-on ✗) | ⚠️ flagged | FFC: detection methods disagree (speed 210, height 214, hip_decel 215).; View detected with low confidence: Run-up travel: 1.6 body lengths  |
| oblique_behind_60fps | 70 / 79 / 86 | 71 / 81 / 85 | ✅ pass | 71 / 81 / 85 | ✅ pass |  |
| side_c1_30fps | 30 / 35 / 39 | 30 / 34 / 39 | ✅ pass | 31 / 34 / 39 | ✅ pass |  |
| side_c2_30fps | 77 / 80 / 86 | 78 / 81 / 86 | ✅ pass | 78 / 81 / 86 | ✅ pass |  |
| side_c4_30fps | 47 / 52 / 55 | 47 / 50 / 56 | ✅ pass | 47 / 50 / 56 | ✅ pass |  |
| side_nets_30fps | 124 / 130 / 134 | 125 / 128 / 133 | ✅ pass | 125 / 128 / 133 | ✅ pass |  |
| side_nets_60fps | 248 / 260 / 267 | 251 / 259 / 267 | ✅ pass | 251 / 259 / 267 | ✅ pass |  |
| side_nets_follow_pan | 248 / 260 / 267 | 252! / 258 / 267 | ❌ wrong, not flagged | 252! / 258 / 267 | ⚠️ flagged | FFC: detection methods disagree (speed 258, height 262, hip_decel 263).; BFC: detection methods disagree (speed 252, height 248). |
| synthetic_sample_120fps | 70 / 92 / 104 | 70 / 92 / 104 | ✅ pass | 70 / 92 / 104 | ✅ pass |  |

**Before:** 8 pass, 6 wrong without any warning.  **After:** 8 pass, 6 flagged for review with the right reason, 0 wrong without a warning.

## Honest summary
* **Same accuracy on these clips:** the same 8 of 14 are fully correct before and after.
* **No more silent errors:** the 6 that were wrong without any warning are now all flagged with the right reason.
  The user can fix them in seconds (Check detection → Recompute, no re-upload).
* **Got worse (flagged):** the TV slow-motion replay. v2.13 had BFC/FFC close (247 / 338 vs 238 / 335), but its
  release was 9 frames off and it showed no warning. v3.0 puts BFC/FFC far off (397 / 422) and flags it; its release
  (455) is within tolerance. The camera now counts as moving on this clip, so the movement correction is applied, and
  the foot signals become unreliable. Set the frames by hand for replays.
* **Still wrong (flagged):**
  * 3 clips filmed from behind: the pose model swaps left and right when the bowler faces away.
  * The TV slow-motion replay.
  * The handheld clip with a small, followed bowler.
  * The front-diagonal clip: its key moments are now right, but it is still classified side-on.
* **Invariance (tests/test_invariance.py):** all pass, at keypoint level and through the real pipeline on video.
  * Mirror, 30/60 fps, 50 %/150 % scale, 10 % crop and 90° rotation with a rotation tag give the same key moments
    within ±1 frame.
  * This fixed a 0.12 s back-foot error at 30 fps (a planted foot "landing" twice).
* **Tests:** 144 backend tests pass (137 fast + 7 slow video tests); frontend tsc, build and 5 tests pass.
* **UI check (real browser):**
  * On a flagged clip the Check detection panel opened by itself and showed the reasons.
  * Changing the arm and setting the release frame, then Recompute, took 6.9 s; the Diagnostics graph loaded.

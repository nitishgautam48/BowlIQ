# BowlIQ — AI fast-bowling biomechanics analyzer

Upload a side-on clip of one delivery (plus an optional front-on clip). BowlIQ tracks the skeleton on every frame,
finds back foot contact (BFC), front foot contact (FFC) and release, measures the delivery with plain geometry, judges
it against published research or your personal baseline, and writes a ranked, calm coaching report with an annotated
video and a PDF.

* **AI does perception only** (MediaPipe pose). **Measurement is NumPy geometry.** The LLM (Gemini Flash, optional)
  only explains the numbers it is given, and its output is rejected if it invents a number.
* Every metric carries `value, unit, view, frame, confidence (High/Medium/Low), confidence_reason, status`.
  If something can't be measured reliably, it is `not_measured` with a reason, never a guess.
* 100 % free and local: no paid APIs, no Redis, no cloud. SQLite with a lease-based job queue, token sign-in,
  per-user data isolation, rate limits and signed media links (see CHANGES.md for the security hardening).

```
bowliq/
├── backend/            FastAPI + pipeline (Python 3.11)
│   ├── app/            main.py, api/, core/ (config, jobs, storage), models/ (schemas, db),
│   │                   pipeline/ (one module per stage), biomech/, analysis/, report/
│   ├── models/         pose_landmarker_heavy.task (downloaded automatically on first run)
│   ├── samples/        sample_delivery.mp4 (synthetic, see "Sample clip")
│   ├── scripts/        run_folder.py, validate.py, make_sample.py
│   ├── storage/        SQLite DB + per-analysis files (gitignored)
│   └── tests/          pytest suite
└── frontend/           Next.js 16 + React 19 + TypeScript (types/analysis.ts = API contract)
```

---

## 1. Windows 11 setup (project in `E:\bowliq`)

Unzip `BowlIQ.zip` into `E:\` so the folders are `E:\bowliq\backend` and `E:\bowliq\frontend`.
See **WINDOWS_COMMANDS.md** for the copy-paste command list; the short version:

```powershell
winget install Gyan.FFmpeg
winget install Python.Python.3.11
winget install OpenJS.NodeJS.LTS
```

Open a **new** PowerShell window (so PATH updates), then:

```powershell
E:
cd E:\bowliq\backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --require-hashes -r requirements-dev.win.lock
copy .env.example .env
python -m pytest -q
uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-server-header
```

If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (once), then activate again.

* **First start** downloads the pinned pose model (`pose_landmarker_heavy.task`, ~30 MB) into `backend\models\` and
  verifies its SHA-256. It then **prints your access token** in the console and saves it in
  `backend\storage\.api_token` (`type storage\.api_token` shows it).
* **Sign in:** open http://localhost:3000 → you are sent to the sign-in page → paste the token.
* **Several people:** set `BOWLIQ_API_TOKENS=name1:longtoken1,name2:longtoken2` in `.env`; each person only sees
  their own deliveries.
* **Lock files:** `requirements*.win.lock` / `*.linux.lock` pin every package with SHA-256 hashes (verified to install
  on Windows/Python 3.11). `requirements.txt` holds the same top-level pins without hashes.

Frontend (second PowerShell window):

```powershell
cd E:\bowliq\frontend
copy .env.example .env.local
npm ci
npm test
npm run build
npm start
```

(`npm run dev` instead of `build` + `start` for live reloading while editing.)

### `.env` reference

| Key | Default | Meaning |
|---|---|---|
| `BOWLIQ_API_TOKENS` | empty | `owner:token` pairs (tokens ≥ 16 chars). Empty = one auto-generated token |
| `GEMINI_API_KEY` | empty | Optional free key (https://aistudio.google.com/apikey). Empty = template report |
| `GEMINI_MODEL` | `gemini-2.5-flash` | A dated Gemini Flash model id |
| `BOWLIQ_CORS_ORIGINS` | `http://localhost:3000,…` | Frontend origins allowed to call the API |
| `BOWLIQ_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Host names the API answers to (DNS-rebinding protection) |
| `BOWLIQ_MAX_QUEUED_JOBS` | 5 | Per-user cap on waiting/running analyses |
| `BOWLIQ_RATE_ANALYZE_PER_MIN` / `_READ_PER_MIN` | 6 / 600 | Per-user rate limits |
| `BOWLIQ_RETENTION_DAYS` | 0 | Auto-delete analyses older than N days (0 = keep) |
| `BOWLIQ_EMBEDDED_WORKER` | true | false = run `python -m app.worker` as a separate process |
| `BOWLIQ_STORAGE_DIR` | `backend\storage` | Database + analysis files location |
| `BOWLIQ_POSE_BACKEND` | auto | auto (as in v3; see `perception.auto_primary`) / rtm (RTMDet + ByteTrack + RTMPose) / yolo / mediapipe / vitpose (placeholder) |
| `BOWLIQ_YOLO_IMGSZ` / `BOWLIQ_YOLO_BATCH` | 1280 / 8 | GPU engine input size and batch |
| `BOWLIQ_VIDEO_ENCODER` | auto | auto / nvenc / x264 |
| `BOWLIQ_POSE_DELEGATE` | auto | MediaPipe delegate: auto / cpu / gpu (GPU delegate isn't available on Windows) |

## 1b. Using an NVIDIA GPU (RTX 4060 etc.)

Two optional GPU accelerations; the app works the same without them.

| Stage | CPU (default) | GPU |
|---|---|---|
| Pose (largest share of the time) | MediaPipe Pose Landmarker heavy | Ultralytics **YOLO26x-pose** on CUDA, batched, FP16 |
| Video encoding (preprocess + 2 renders) | libx264 | **NVENC** (`h264_nvenc`) |

* **NVENC** needs nothing extra: the Gyan ffmpeg build includes it. At startup BowlIQ runs a 0.2 s trial encode and
  uses NVENC if it works, otherwise libx264 (`BOWLIQ_VIDEO_ENCODER=auto|nvenc|x264`).
* **GPU pose engine**: install PyTorch with CUDA and `requirements-gpu.txt` (commands in WINDOWS_COMMANDS.md → GPU).
  With `BOWLIQ_POSE_BACKEND=auto` (default) it's used whenever a CUDA GPU is available. The weights
  (`yolo26x-pose.pt`, 126 MB) download once from the official Ultralytics GitHub release and are SHA-256 verified.
  If it can't find the bowler in a clip (it ignores the cartoon sample, for example), MediaPipe is used for that clip
  automatically and the report says so.
* **Accuracy check**: on a real 58 fps phone delivery both engines gave the same events as frame-by-frame manual
  review (BFC 69/70, FFC 76/76, release 84/85; manual ≈ 67–70, 76, 84–85) and the same bowling arm. Joint angles differ
  by a few degrees between the engines (e.g. front knee at FFC 151° vs 161°), so don't mix engines within one
  bowler's baseline if you can avoid it.
* **Speed**: not measured on an RTX 4060 (no GPU in the build environment). YOLO26x-pose is ~1 s/frame on a laptop
  CPU, versus roughly 20–40 ms/frame expected on a 4060 in FP16 at 1280 px. `BOWLIQ_YOLO_IMGSZ=960` trades a little
  detail for about twice the speed. `/api/v1/health` (and Settings) shows which engine, device and encoder are active.
* Licence: Ultralytics is AGPL-3.0, which is fine for personal, non-commercial use.

## 2. Running, tests, backups, Docker

* Backend: `uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-server-header` (API docs at
  http://localhost:8000/api/v1/docs).
* Optional separate worker: set `BOWLIQ_EMBEDDED_WORKER=false` and run `python -m app.worker` in another window. Jobs
  are claimed atomically from SQLite with a lease, so any number of workers is safe.
* Tests: `python -m pytest -q` (57 backend tests) and `npm test` (frontend). Lint: `ruff check app scripts tests`,
  `bandit -q -r app -s B404,B603`, `pip-audit -r requirements.txt`.
* Backup (safe while running): `python scripts\backup.py backup E:\BowlIQBackups` (keeps 14 by default).
  Restore (server stopped): `python scripts\backup.py restore E:\BowlIQBackups\bowliq-YYYYMMDD-HHMMSS`.
* Docker / self-hosting: `docker compose up --build` (API, worker and web, non-root, read-only root filesystems,
  bound to 127.0.0.1). For internet exposure put `deploy/Caddyfile` (TLS, body-size limit, HSTS) in front and set
  `BOWLIQ_ALLOWED_HOSTS` / `BOWLIQ_CORS_ORIGINS` to your domains.
* CI: `.github/workflows/ci.yml` runs lint, bandit, pip-audit, pytest, tsc, frontend tests, npm audit and the build.

## 3. Filming guide

Side-on, square to the popping crease, about **10 m** away, camera at **hip height** on a **tripod**, **60 fps
minimum (120+ preferred)**, landscape, whole body in view from the last 3–4 run-up strides to follow-through.
Optional front-on clip from behind the stumps, recorded at the same time: **clap once, loudly, after both phones start recording**, about equally far from both phones, so the clips are synced automatically. If other people are in view, tap the bowler (and shade bystanders) in the upload's "Who is the bowler?" step. Under 50 fps the clip is accepted with Low quality and a warning.

## 4. Pipeline

| Stage | Module | What it does |
|---|---|---|
| Quality | `pipeline/quality.py` | fps, resolution, duration, rotation; camera motion (frame difference outside the person box); side/front view from run-up geometry. Hard-fails only unreadable files or < 1 s (and > 30 s after trim). |
| Preprocess | `pipeline/preprocess.py` | trim, auto-rotate, long side ≤ 1280, constant frame rate with each frame at its true time (variable-frame-rate phone clips are converted at their nominal rate, e.g. 59.94, repeating a frame only where the phone skipped one), H.264 temp MP4. |
| Sync (two clips) | `pipeline/sync.py` | front_time = side_time + offset. Sound first: GCC-PHAT and onset cross-correlation must agree within 4 ms and still agree with the matched sound blanked out; else the release moment in both clips (±1–2 frames); else not synced (align by hand in Linked playback). |
| Pose | `pipeline/pose.py` | MediaPipe Tasks PoseLandmarker (heavy), VIDEO mode with timestamps, every frame; 33 → 17 COCO joints + heel and toe of each foot (21). Bowler selection (`pipeline/tracking.py`): the person you tapped, else the top-3 candidates by steady horizontal travel tried in turn; people inside ignore zones are dropped; frame-to-frame tracking uses predicted position, body size and a torso/leg colour signature so it can't jump to a different-looking person. |
| Smooth | `pipeline/smooth.py` | frame-level left/right label swaps of knee+ankle and elbow+wrist undone by constant-velocity continuity (MediaPipe often swaps legs side-on); gaps (visibility < 0.5) ≤ 6 frames interpolated, longer gaps left missing; Savitzky–Golay (~30 ms window, odd, ≥ 5; lighter on the bowling arm); bowling arm = the wrist that rises furthest above the nose anywhere in the clip (2nd-highest frame, so one bad frame can't decide; whole clip because release isn't always in the 2nd half); direction detected and mirrored so the maths always runs left→right. |
| Events | `pipeline/events.py` | FFC: after the last airborne phase, first frame where the front ankle's horizontal speed drops > 60 % vs its approach speed (peak over the previous 0.1 s, ≥ 3 frames; the original "previous 3 frames" rule missed gradual real-world stops) and its vertical speed is ≈ 0 (allowing 60 ms for heel-to-flat settling). Both bowling-arm hypotheses are tried; the one whose wrist is highest above its own shoulder at release wins. "Hand above the head" falls back to eyes/ears/shoulders when the face is hidden. BFC: same on the back ankle before FFC. Release: highest bowling wrist within 150 ms after FFC. Plausibility gates (a run-up; hand clearly above the head at release; BFC < FFC < release). Otherwise `EVENTS_NOT_FOUND` with a filming fix. |
| Adapting to imperfect video | `pipeline/adapt.py` | camera-movement correction (background tracked, people masked, kept only if the planted foot gets stiller), enlarged-quarters search and zoomed second pass for small/dark bowlers, refusal under 70 px tall. `BOWLIQ_ADAPT_VIDEO=false` turns it off. |
| Camera angle | `pipeline/quality.py` | side-on / front-on / behind / diagonal from the run-up (sideways travel, growth or shrink of a posture-robust body length, shoulder width). Main clip: side-on or diagonal; second clip: front-on or behind (diagonal at Low). Clips uploaded the wrong way round are swapped. |
| 3D rotation | `pipeline/three_d.py` | with the clips synced and the second clip front-on/behind: shoulder and hip lines in the horizontal plane from both views (side-on → along the pitch, second → across it), second phone interpolated to the exact synced time, its turn from the line of the run measured from the run-up and undone. Replaces the flattened hip–shoulder separation and counter-rotation. |
| Multi-delivery verdicts | `analysis/consistency.py` | consistent / occasional / one-off issues across the last 5–20 deliveries (Medium/High confidence only), drills for repeated issues, personal trends. |
| Combining clips | `pipeline/orchestrator.py` | with a sound sync the side-on key moments are carried into the second clip; when the second clip's own detection agrees, both go up one confidence level. |
| Real-footage checks | `pipeline/metrics.py` | **Oblique camera:** if the planted back foot (BFC) and front foot (FFC) aren't on one image line (> 10° tilt) the run isn't square to the camera, so stride and run-up speed drop to Low confidence (> 20°: not measured) and ball speed isn't reported. **Planted front foot:** if the tracked front ankle moves > 12 % of body height between FFC and release, the leg estimate is unreliable (blur/occlusion), so front-knee-at-release, knee flexion and release height are not measured. |
| Scale | `pipeline/scale.py` | body height px = average of (nose → mean ankle in the most upright frame × 1.07) and (shank + thigh + trunk + neck→nose segment lengths × 1.07, posture-robust). Their disagreement is the scale uncertainty, which caps confidence of distance/speed metrics and widens the ball-speed ±. With your height: metres per pixel; without: body-relative units. |
| Metrics | `pipeline/metrics.py`, `biomech/angles.py` | see below. |
| Ball speed | `pipeline/ball_speed.py` | needs your height. MOG2 background subtraction + ball-sized round/streaked blob filter + nearest-to-predicted-path association for up to 20 frames after release; straight-line fit; `± ` from fit residual and pixel quantisation (×1.5 under 50 fps). Not reported when unscaled, the camera moves, or fewer than 4 ball positions are found. |
| Baseline | `analysis/baseline.py` | ≥ 10 deliveries: mean (and SD) of the top 25 % by ball speed, or the most recent 10 when speed isn't available. |
| Rules | `analysis/rules.py`, `analysis/benchmarks.yaml` | severity = normalised deviation × weight × confidence factor (High 1.0 / Medium 0.7 / Low 0.4); top 3 findings (never padded), up to 3 strengths, kinetic chain, injury flags. |
| Report | `report/generator.py`, `report/coaching_notes.md` | Gemini Flash gets only the findings JSON and the allowed cues/drills; structured JSON output; every number must appear in the input (rounding tolerance) and cues/drills must be verbatim from the notes, else the deterministic template is used. API errors and 429s also fall back. |
| Video | `report/render_video.py` | skeleton, knee and trunk angle arcs with labels, BFC/FFC/RELEASE labels, frame counter, 4× repeated frames (0.25× slow motion) for release ± 0.3 s; H.264 yuv420p faststart. A matching overlay-free video is served for the interactive player. |
| PDF | `report/pdf.py` | headline, ball speed, top fixes with sources, load patterns, key frames (BFC/FFC/release), metrics table, sources, limits. |

### Generalisation (v3.0): no per-clip tuning

* **One config file.** Every threshold lives in `backend/config/pipeline.yaml`, in seconds, body heights or fractions
  of the clip's own signal (units in the comments). Override the path with `BOWLIQ_PIPELINE_CONFIG`.
* **Clip context** (`pipeline/clip_context.py`): rotation (verified with the pose), direction, bowling arm, view and
  bowler, each with a confidence and the reasons, shown in the report under "Check detection".
* **Key moments by consensus** (`pipeline/consensus.py`): 3 methods for release and FFC and 2 for BFC. On
  disagreement the result is Low confidence with `needs_review` and the reason.
* **Stages degrade, they don't fail.** A job fails only for an unreadable video, no person, or a clip under 1 s.
  `result.stages` says which steps ran.
* **Corrections:** `POST /api/v1/analysis/{id}/recompute` reuses the cached keypoints (`cache.pkl`) and takes seconds.
  Corrections are logged in `storage/corrections.jsonl`.
* **Diagnostics:** `storage/debug/{id}/` (signals.png, context.json, key frames, roles.mp4, pose3d.json), shown in Coach view, with a rotatable 3D skeleton synced to the roles video.
* **Regression:** `backend/tests/clips/` (video + sidecar JSON). Run `python scripts/regression.py`; it must end with
  `OK` before any change to the pipeline or config is accepted. `scripts/label_clip.py` labels new clips.
  `tests/test_invariance.py` checks mirror, frame rate, scale, crop and rotation.

### Perception (v4.0): two pose models, limb identity, 3D

* **Detection + tracking:** RTMDet-m finds people in every frame and ByteTrack keeps their identities; the bowler is
  the track with the run-up pattern (sustained travel, sharp deceleration, big arm swing), never the biggest person.
* **2D pose:** with `BOWLIQ_POSE_BACKEND=rtm`, RTMPose-m Halpe-26 (top-down, with heels and toes) is the primary
  estimator and MediaPipe runs on the bowler's crop as the secondary. It isn't the default yet: on the 14 labelled
  clips it scored 5 pass / 3 flagged / 5 fail vs MediaPipe's 8 / 6 / 0. Per frame and joint their disagreement (body heights) lowers that joint's
  confidence, and joints the primary misses are filled from the secondary. Models: OpenMMLab's official ONNX
  exports, run with ONNX Runtime (CUDA on an NVIDIA GPU). `BOWLIQ_POSE_BACKEND=auto|rtm|yolo|mediapipe|vitpose`
  (vitpose is a placeholder for later).
* **Identity** (`pipeline/identity.py`):
  * Left/right leg and arm swaps are undone per frame with a Hungarian assignment against motion-predicted limbs,
    with hysteresis so labels can't flicker.
  * Limbs then get roles: the front leg is the foot that lands at FFC, the back leg the one landing at BFC, and the
    bowling arm comes from the clip context.
  * Every metric uses roles, never raw labels.
  * Corrections are logged in the diagnostics and drawn in `roles.mp4`.
* **3D lifting:** MotionBERT (Human3.6M) lifts the corrected 2D sequence. Knee angles, trunk flexion, lateral
  flexion, hip–shoulder separation and shoulder counter-rotation are computed in 3D.
* **View-aware gating:**
  * A 2D angle is used only if the limb's plane is within 20° of the picture; otherwise the 3D value is used.
  * Confidence is High only when 3D and a valid 2D agree within 7°, Medium with one source, Low when they disagree.
  * Per metric, `angles.source` in `config/pipeline.yaml` is `auto` (the rule above), `2d`, `3d` or `off` (ignore
    3D). Knees ship as `off`: on the side-on regression clips MotionBERT's knees were 12–42° more bent than the
    reliable side-on 2D, while trunk flexion agreed within 4°.
  * `config/pipeline.yaml → angles` holds these limits and the per-metric source.
* **Foot contacts:** heel and toe keypoints add two methods to the BFC/FFC consensus: the foot's lowest point
  settling, and heel-strike speed collapse.
* **Validation:** `python scripts/validate.py your_kinovea.csv --sources --clips-dir tests/clips` prints the mean
  absolute error per metric for MediaPipe-2D, RTMPose-2D and 3D-lifted. `--write-config` keeps the most accurate
  source per metric.
* **Install:** `pip install -r requirements-rtm.txt` (ONNX Runtime GPU) plus PyTorch (for MotionBERT). Without them,
  BowlIQ runs as before with MediaPipe and 2D angles.

## 5. Metrics and sources

| Key | View | Definition | Rule (benchmarks.yaml) | Source |
|---|---|---|---|---|
| `stride_length_pct_height` | side | back ankle at BFC → front ankle at FFC, horizontal, ÷ body height × 100 | band 75–85; pace 0.6, injury 0.6 | Spratford et al., 2016, Int J Sports Sci & Coaching |
| `front_knee_angle_ffc` | side | hip–knee–ankle angle at FFC (180° = straight) | monitor | Portus et al. (context) |
| `front_knee_angle_release` | side | same, at release | higher is better vs baseline; pace 0.7 | Ramachandran et al., 2024 meta-analysis |
| `front_knee_flexion_ffc_to_release` | side | max bend after FFC relative to the FFC angle (+ = bending) | flag ≥ 10° (flexor); pace 1.0 | Portus et al.; Worthington, King & Ranson, 2013, J Appl Biomech |
| `trunk_forward_flexion_release` | side | mid-hip→mid-shoulder vs vertical at release (+ = towards batter) | higher is better vs baseline; pace 0.5 | Worthington et al., 2013 |
| `release_height_pct_height` | side | wrist height above the planted front ankle at release ÷ body height × 100 | monitor | — |
| `runup_speed` | side | mid-hip horizontal speed, line fit over 0.3 s before BFC (m/s with height, else % height/s) | higher is better vs baseline; pace 1.0 | Worthington et al., 2013; Ramachandran et al., 2024 |
| `hip_shoulder_separation_bfc` | front | angle between shoulder and hip lines in the image plane at BFC | flag ≥ 30°; max confidence Medium | Portus et al., 2004 |
| `shoulder_counter_rotation` | front | (max shoulder-line angle BFC→FFC) − (angle at BFC) | injury flag ≥ 30° (mixed action); injury 1.0; max confidence Medium | Portus et al., 2004; Alway et al. |
| `trunk_lateral_flexion_release` | front | mid-hip→mid-shoulder sideways angle from vertical at release | monitor | — |

Front-on metrics without a front clip are `not_measured` with reason "Add a front-on clip". "Higher is better"
metrics are `info` ("Personal baseline unlocks after 10 deliveries") until the baseline exists, then `outside` when
below baseline − 1 SD. Load-pattern flags read "Load pattern worth checking with a coach or physio" — never a diagnosis.

## 6. API reference (v1)

Base path `/api/v1`. All errors are JSON `{ "code", "message", "fix" }` (internal errors carry only an error id).
Every route except `/health` needs `Authorization: Bearer <token>`; state-changing requests also need
`X-BowlIQ-Client: web` and an allowed `Origin`. Media routes (`video`, `frame`, `pdf`) also accept the short-lived
signed query from `/links` because `<video>`/`<img>` can't send headers. Users only ever see their own data
(other ids return 404).

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | public: ffmpeg, model (verified), database, pose device, Gemini key present; no paths |
| GET | `/me` | the signed-in owner |
| POST | `/analyze` | multipart: `side_video` (required), `front_video`, `bowler_name`, `bowler_height_cm` (120–230), `trim_start_s`, `trim_end_s`, `side_tap` / `front_tap` (`t,x,y`: seconds into the uploaded clip and 0–1 frame fractions), `side_ignore` / `front_ignore` (`x0,y0,x1,y1;…`, up to 8 zones), `slow_motion` (×1–×60; omitted = detect), `camera_view` (side-on / front-on / behind; omitted = detect), `bowling_arm` (right / left; omitted = detect). With two clips, both phones should hear one loud clap (they are synced from sound). MP4/MOV (signature-checked), ≤ 200 MB (enforced before reading the body), 1–30 s after trim, finite numbers only. Rate-limited, per-user queue cap → `{ job_id }` |
| GET | `/jobs/{job_id}` | `{ job_id, status, stage, progress, error?, result? }` |
| GET | `/analysis/{id}` | `AnalysisResult` (`frontend/types/analysis.ts` = `backend/app/models/schemas.py`) |
| GET | `/analysis/{id}/links` | `{ query, expires_at }` signed media query (6 h) |
| GET | `/analysis/{id}/video` | annotated H.264 MP4, HTTP range support, immutable caching; `?clean=true` overlay-free; `?view=front` the front-on clip (overlay-free) |
| GET | `/analysis/{id}/keypoints` | `{ fps, width, height, joints[21], frames }` normalised 0–1, smoothed (17 COCO joints + heel and toe of each foot). `?view=front` for the front-on clip |
| GET | `/jobs/{id}/video`, `/jobs/{id}/links` | the processed clip of a failed job (kept 24 h when the failure can be fixed by marking) |
| POST | `/jobs/{id}/mark` | `{ bfc, ffc, release, slow_motion? }` → finish a failed analysis with key moments marked by hand |
| POST | `/analysis/{id}/reanalyze` | `{ bfc, ffc, release }` frame numbers set by hand → new job re-analysed with exactly those key moments (same video, same bowler) |
| POST | `/analysis/{id}/recompute` | `{ view?, bowling_arm?, direction?, bowler_track_id?, bfc_frame?, ffc_frame?, release_frame?, height_cm?, clear? }` → recomputes this delivery in place from the cached keypoints (seconds); logged to `storage/corrections.jsonl` |
| GET | `/analysis/{id}/debug/{name}` | diagnostics: `signals.png`, `context.json`, `bfc.jpg`, `ffc.jpg`, `release.jpg`, `roles.mp4`, `pose3d.json` |
| GET | `/analysis/{id}/person/{cid}` | thumbnail of candidate person `cid` (from `context.candidates`) for "Choose bowler" |
| GET | `/bowlers/{name}/summary?last=10` | consistent vs one-off issues across recent deliveries, strengths, trends, coach summary |
| PUT | `/analysis/{id}/sync` | `{ "offset_s": number \| null }`: save a hand-adjusted alignment of the two clips (front_time = side_time + offset), or null for the automatic one |
| GET | `/analysis/{id}/frame/{n}` | JPEG with overlay (`?joint=` highlights a joint), cached on disk |
| GET | `/analysis/{id}/pdf` | PDF report |
| DELETE | `/analysis/{id}` | removes files and DB rows (feedback cascades) |
| POST | `/analysis/{id}/feedback` | `{ finding_rank, helpful }`, one vote per finding (updates in place) |
| GET | `/deliveries?bowler=&limit=&offset=` | paginated list |
| GET | `/bowlers` | bowlers with counts |
| GET | `/bowlers/{name}/history?limit=` | `{ deliveries[{…, metrics: {key: {value, unit}}}], baseline_active, deliveries_until_baseline }` |
| GET | `/compare?a=&b=` | both results + diffs (better/worse by each metric's direction; unit mismatches flagged) |
| GET | `/benchmarks` | rules, methods and sources |
| GET | `/samples`, `/samples/{name}` | sample clips |
| GET | `/metrics` | Prometheus text: requests, job outcomes, stage time, queue depth |

Jobs live in SQLite and are claimed atomically by a worker (embedded thread or `python -m app.worker`) that renews a
lease while working. On startup, jobs left `processing` whose lease has expired are marked failed with code
`INTERRUPTED`; queued jobs are kept and processed. The schema is versioned (`PRAGMA user_version`) and migrated
automatically, with foreign keys and cascades enabled.

## 7. Sample clip

`backend/samples/sample_delivery.mp4` is a **synthetic** side-on clip drawn with OpenCV (`scripts/make_sample.py`): a
shaded figure 1.80 m tall at 200 px/m, 120 fps, ground-truth BFC 70 / FFC 92 / release 104, and a ball released at
~122 km/h. MediaPipe detects the figure, so it exercises the full real pipeline (the detected events match the
ground truth exactly; ball speed reads about 127 ± 9 km/h, stride 81 % vs 80.6 % true). It is **not** a real bowler,
so its technique numbers mean nothing.

To replace it with a real clip: put any `.mp4`/`.mov` in `backend/samples/` (the first file alphabetically is used)
and delete the synthetic one. Use footage you own or a Creative Commons clip (e.g. search Wikimedia Commons for
"fast bowling" videos with a CC BY / CC BY-SA licence, and keep the attribution in `backend/samples/ATTRIBUTION.txt`).
"Analyze a sample clip" sends height 180 cm (the synthetic figure's height) and saves under the reserved bowler
"Sample (synthetic)", which never enters your history or personal baseline. Edit `SAMPLE_HEIGHT_CM` in
`frontend/lib/useSample.ts` to match a real clip.

## 8. Validating accuracy (Kinovea + validate.py)

1. Analyse your clips in bulk: `python scripts/run_folder.py D:\clips --bowler "Alex" --height 175`
   (writes `D:\clips\bowliq_runs.csv` mapping clip → analysis id).
2. Open each clip in [Kinovea](https://www.kinovea.org/) (free). Step frame by frame to find BFC, FFC and release and
   note their frame numbers (Kinovea counts from 0 if you set the time format to "frame number"; BowlIQ frames are
   counted on the trimmed, constant-frame-rate clip, so analyse untrimmed files with run_folder.py). Use the angle tool
   on the hip–knee–ankle at FFC and release, and the line tool against vertical for the trunk.
3. Write a CSV:
   ```csv
   clip,metric,manual_value,event,manual_frame
   nets1.mp4,front_knee_angle_ffc,158,,
   nets1.mp4,trunk_forward_flexion_release,34,,
   nets1.mp4,,,FFC,57
   nets1.mp4,,,release,66
   ```
4. Run `python scripts/validate.py manual.csv --runs D:\clips\bowliq_runs.csv` → MAE per metric and mean / max frame
   error (and ms) per event.

## 9. Known limits

* **2D video.** One camera can't measure rotation towards or away from it. Front-on rotation metrics are image-plane
  projections of 3D rotations and are capped at Medium confidence. Side-on angles assume the camera is square to the
  crease.
* **Frame rate.** At 30 fps an event can be 33 ms off, release may fall between frames and ball speed is less precise.
  Use 120 fps or more.
* **Scale.** Body height is nose-to-ankle × 1.07 in the most upright frame; errors there scale every metre value and
  the ball speed. Without your height, stride and release height are % of body height and speeds are % height/s.
* **Camera angle.** Distances along the pitch are only reported when both foot contacts sit on one image line
  (square-on camera); a phone filming from behind the bowler's shoulder or at an angle will get angles but no stride,
  run-up speed or ball speed.
* **Heel-first landings.** The 17-point skeleton has no heel or toe points, so foot contact is found from the ankle.
  When the bowler lands heel-first, the CPU engine can mark front foot contact 2–3 frames (~50 ms at 60 fps) after
  heel strike (seen on a real clip: manual 252–253, MediaPipe 256, GPU engine 253). Stride is measured between the
  settled foot positions (footprints), so it is not affected by this.
* **Clip framing.** Only the stretch where the bowler is on screen is analysed (empty lead-in/out is ignored and the
  report says which seconds were used). Clips filmed from in front of or behind the bowler are rejected with
  `WRONG_CAMERA_ANGLE`: side-on is required for the main clip.
* **Pose model.** Loose clothing, low light, motion blur and left/right leg swaps in side-on views reduce joint
  confidence; low-visibility joints are interpolated only across ≤ 6 frames.
* **Ball tracking** needs a steady camera, a ball visible against the background and ≥ 4 positions; otherwise it is
  reported as not measured.
* **Targets** are population ranges from the cited research; your coach may set different ones. BowlIQ never diagnoses
  injuries.

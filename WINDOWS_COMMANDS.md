# BowlIQ: Windows commands (project at `E:\bowliq`)

Unzip `BowlIQ.zip` into `E:\` first, so you have `E:\bowliq\backend` and `E:\bowliq\frontend`.
Use **PowerShell**. Every block is meant to be run in order.

> **Already installed an earlier version?** Skip to "Updating an existing install" at the end. Your analyses, `.env`,
> `.venv`, `node_modules` and downloaded models are kept.

## 0. One-time installs (any PowerShell window)

```powershell
winget install Gyan.FFmpeg
winget install Python.Python.3.11
winget install OpenJS.NodeJS.LTS
```

Close PowerShell and open a new one, then check:

```powershell
ffmpeg -version
py -3.11 --version
node --version
```

## 1. Backend setup (once)

```powershell
E:
cd E:\bowliq\backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run `Activate.ps1`, run this once, then the activate line again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

```powershell
python -m pip install --upgrade pip
pip install --require-hashes -r requirements-dev.win.lock
copy .env.example .env
notepad .env
```

(In `.env` you can paste a free Gemini key into `GEMINI_API_KEY=`. Leaving it empty is fine.)

Run the test suite (57 tests, about 1 minute; the first run also downloads the pose model):

```powershell
python -m pytest -q
```

## 2. Start the backend (every time; Terminal 1)

```powershell
E:
cd E:\bowliq\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-server-header
```

The first start prints **your access token**. To see it again later:

```powershell
type E:\bowliq\backend\storage\.api_token
```

Health check (should show `"ok":true`):

```powershell
curl.exe http://127.0.0.1:8000/api/v1/health
```

## 3. Frontend setup (once; Terminal 2)

```powershell
E:
cd E:\bowliq\frontend
copy .env.example .env.local
npm ci
npm test
npm run build
```

## 4. Start the frontend (every time; Terminal 2)

```powershell
E:
cd E:\bowliq\frontend
npm start
```

Open http://localhost:3000 → sign in with the access token → **Analyze a sample clip** or **Start an analysis**.

## GPU (RTX 4060): optional, for faster analysis

Update the NVIDIA driver first (GeForce Experience / NVIDIA App, or nvidia.com). Then, with the backend venv active:

```powershell
cd E:\bowliq\backend
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

If pip says no matching version, use the CUDA 12.8 build instead:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```powershell
pip install -r requirements-gpu.txt
pip install --force-reinstall --no-deps opencv-contrib-python==5.1.0.95
```

(The last line restores the OpenCV build MediaPipe needs, because Ultralytics pulls in a second OpenCV package.)

Check the GPU is visible to PyTorch (should print `True NVIDIA GeForce RTX 4060…`):

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Restart the backend. The console should say `Pose engine: ultralytics_yolo26x_pose on gpu; video encoder: h264_nvenc`,
and Settings → Backend status shows the same. To go back to the CPU engine, set `BOWLIQ_POSE_BACKEND=mediapipe` in
`.env`.

### Perception upgrade (v4.0): RTMPose + MotionBERT on the RTX 4060

RTMDet and RTMPose run on ONNX Runtime; MotionBERT (3D) runs on PyTorch. Use the **CUDA 12.8** PyTorch build: it
ships the CUDA 12 libraries that `onnxruntime-gpu` 1.23 uses.

```powershell
cd E:\bowliq\backend
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```powershell
pip install -r requirements-rtm.txt
```

Check that both use the GPU. The first command should print `True`:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

The second loads the RTMDet model on the GPU (downloading it once) and should print a list starting with
`CUDAExecutionProvider`. If it prints only `CPUExecutionProvider`, the CUDA libraries didn't load:

```powershell
python -c "from app.pipeline import rtm; print(rtm.session('rtmdet_m_person').get_providers())"
```

(`onnxruntime.preload_dlls()` on its own prints "Failed to load cublas…" lines when the CUDA libraries come with
PyTorch rather than separate NVIDIA pip packages. That's harmless: BowlIQ imports PyTorch first, which loads them.)

MotionBERT (3D angles) is now used automatically. To also try the RTMPose engine, add `BOWLIQ_POSE_BACKEND=rtm` to
`backend\.env`. It isn't the default yet: on the 14 labelled clips it beat MediaPipe from behind the bowler but lost
elsewhere. Restart the backend. The first analysis downloads the models once (about 230 MB, checked by SHA-256); the
console then says `Pose engine: rtmpose_m_halpe26 on gpu`. Each report's Coach view → Diagnostics → context.json shows the
time per stage (`stages[].seconds`) and `timings.pose_ms_per_frame`.

* No NVIDIA GPU: `pip install onnxruntime==1.23.2 torch` instead (CPU; slower but works).
* Back to MediaPipe only: set `BOWLIQ_POSE_BACKEND=mediapipe` in `.env`.

## Optional

Separate worker process (instead of the one built into the API): set `BOWLIQ_EMBEDDED_WORKER=false` in `.env`, restart
the backend, then in a third window:

```powershell
E:
cd E:\bowliq\backend
.\.venv\Scripts\Activate.ps1
python -m app.worker
```

Backup (safe while running) and restore (stop the backend first):

```powershell
python scripts\backup.py backup E:\BowlIQBackups
```

```powershell
python scripts\backup.py restore E:\BowlIQBackups\bowliq-20261121-093000
```

Analyse a whole folder and check accuracy against Kinovea measurements:

```powershell
python scripts\run_folder.py E:\clips --bowler "Alex" --height 175
```

```powershell
python scripts\validate.py E:\clips\manual.csv --runs E:\clips\bowliq_runs.csv
```

Code checks:

```powershell
ruff check app scripts tests
```

```powershell
bandit -q -r app -s B404,B603
```

```powershell
pip-audit -r requirements.txt
```

After updating from an older version, make old analyses seek smoothly on the Compare page (one time):

```powershell
python scripts\reencode_videos.py
```

Regenerate the synthetic sample clip:

```powershell
python scripts\make_sample.py
```

## Troubleshooting

* `ffmpeg` not found → reopen PowerShell after `winget install Gyan.FFmpeg`.
* `pip install --require-hashes` fails → make sure you're on Python 3.11 (`python --version` inside the venv). As a
  fallback: `pip install -r requirements-dev.txt` (same versions, without hash checking).
* Sign-in says the token is wrong → copy it again from `storage\.api_token` (no spaces).
* "Backend offline" banner → Terminal 1 must be running.
* Port in use → `netstat -ano | findstr :8000`, then close that program, or start with `--port 8001` and set
  `NEXT_PUBLIC_API_URL=http://localhost:8001` in `frontend\.env.local` (then `npm run build` again).

## Updating an existing install (v4.0)

Stop both terminals first (Ctrl+C). Then, in a new PowerShell window:

```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\BowlIQ.zip" -DestinationPath E:\ -Force
```

```powershell
E:
cd E:\bowliq\backend
.\.venv\Scripts\Activate.ps1
python -m pytest -q -m "not slow"
```

Expect `166 passed, 7 deselected`. The slow invariance tests run the real pose model on augmented copies of the sample clip
(about 2–4 minutes):

```powershell
python -m pytest -q -m slow
```

Then the regression suite over the 14 labelled clips in `tests\clips` (the first run caches the skeletons, about
5–10 minutes on CPU; later runs take seconds per clip):

```powershell
$env:POSE_BACKEND="mediapipe"; python scripts\regression.py --jobs 2
```

It should end with `OK: no clip failed or got worse.` Label a new clip with:

```powershell
python scripts\label_clip.py tests\clips\my_clip.mp4
```

Rebuild the frontend:

```powershell
cd E:\bowliq\frontend
npm install
npm run build
```

Start the backend (Terminal 1):

```powershell
E:
cd E:\bowliq\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-server-header
```

Start the frontend (Terminal 2):

```powershell
E:
cd E:\bowliq\frontend
npm start
```

Open http://localhost:3000/new.

* Every report now says what it detected about the clip (camera view, bowling arm, direction, which person is the
  bowler, each key moment) with a confidence. If anything is uncertain, a **"Check detection"** banner appears and
  the panel under the video opens: fix the dropdowns, step to a frame and press **Set as BFC / FFC / Release**, or
  pick the bowler from the thumbnails, then **Recompute**. The report, video and PDF are redone in a few seconds,
  without re-uploading.
* Switch to **Coach** view and open **Diagnostics** to see the signals graph (ankle speeds, wrist height, hip speed
  and each detection method's guess) and the key frames.
* All analysis thresholds are in `backend\config\pipeline.yaml` (commented). If you change one, re-run
  `python scripts\regression.py`: accept the change only if it still says OK.
* Analyses made before v3.0 can't use Recompute (no cached skeleton); "Correct them" still works for those.

### Optional: Gemini video check (v3.2)

Gemini watches each clip first and points out the bowler, camera view, bowling arm and slow motion; the
measurements are still made on your PC. The clip is sent to Google. First measure whether it helps on your 14
labelled clips. Copy your clips into `E:\bowliq\backend\tests\clips` under the names in the `.json` files, make sure
`GEMINI_API_KEY` is set in `backend\.env`, then run the two checks below.

Normal run:

```powershell
cd E:\bowliq\backend
.\.venv\Scripts\Activate.ps1
$env:POSE_BACKEND="mediapipe"; python scripts\regression.py --jobs 2
```

With Gemini:

```powershell
$env:POSE_BACKEND="mediapipe"; python scripts\regression.py --jobs 2 --gemini
```

If the Gemini run shows more `PASS` rows (and no `FAIL`), switch it on: open `backend\.env`, set
`BOWLIQ_GEMINI_VIDEO=1`, save, and restart the backend.

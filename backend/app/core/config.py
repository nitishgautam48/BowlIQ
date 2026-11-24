"""Application settings loaded from environment / .env (see .env.example)."""
from __future__ import annotations

import contextlib
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    backend_dir: Path
    storage_dir: Path
    models_dir: Path
    samples_dir: Path
    pose_delegate: str
    gemini_api_key: str
    gemini_model: str
    gemini_video: bool  # optional first pass: Gemini watches the clip and gives hints (sends the clip to Google)
    gemini_video_model: str
    cors_origins: list[str]
    allowed_hosts: list[str]
    api_tokens_raw: str
    embedded_worker: bool
    retention_days: int
    max_queued_jobs: int
    rate_analyze_per_min: int
    rate_read_per_min: int
    pose_backend: str
    yolo_imgsz: int
    yolo_batch: int
    video_encoder: str
    adapt_video: bool  # Phase 1c: camera-movement correction, zoom on small bowlers, contrast boost, sub-frame timing
    pose_model_name: str = "pose_landmarker_heavy.task"
    # Pinned, versioned model + checksum (supply-chain integrity).
    pose_model_url: str = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                           "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
    pose_model_sha256: str = "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"
    # GPU pose engine (Ultralytics YOLO26x-pose, AGPL-3.0), pinned release asset + checksum.
    yolo_model_name: str = "yolo26x-pose.pt"
    yolo_model_url: str = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt"
    yolo_model_sha256: str = "08ed9e01d22a6f248b04f2f9992016aca9a32250b9ab57057d886a09d026700d"
    max_upload_bytes: int = 200 * 1024 * 1024
    min_duration_s: float = 1.0
    max_duration_s: float = 30.0
    max_long_side: int = 1280
    baseline_min_deliveries: int = 10
    media_link_ttl_s: int = 6 * 3600
    sample_bowler: str = "Sample (synthetic)"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        storage = Path(os.getenv("BOWLIQ_STORAGE_DIR", str(BACKEND_DIR / "storage")))
        return cls(
            backend_dir=BACKEND_DIR,
            storage_dir=storage,
            models_dir=BACKEND_DIR / "models",
            samples_dir=BACKEND_DIR / "samples",
            pose_delegate=os.getenv("BOWLIQ_POSE_DELEGATE", "auto").lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            gemini_video=_bool("BOWLIQ_GEMINI_VIDEO", False),
            gemini_video_model=os.getenv("GEMINI_VIDEO_MODEL", "").strip(),
            cors_origins=_list("BOWLIQ_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"),
            allowed_hosts=_list("BOWLIQ_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver"),
            api_tokens_raw=os.getenv("BOWLIQ_API_TOKENS", "").strip(),
            embedded_worker=_bool("BOWLIQ_EMBEDDED_WORKER", True),
            retention_days=int(os.getenv("BOWLIQ_RETENTION_DAYS", "0") or 0),
            max_queued_jobs=int(os.getenv("BOWLIQ_MAX_QUEUED_JOBS", "5") or 5),
            rate_analyze_per_min=int(os.getenv("BOWLIQ_RATE_ANALYZE_PER_MIN", "6") or 6),
            rate_read_per_min=int(os.getenv("BOWLIQ_RATE_READ_PER_MIN", "600") or 600),
            pose_backend=os.getenv("BOWLIQ_POSE_BACKEND", "auto").strip().lower(),
            yolo_imgsz=int(os.getenv("BOWLIQ_YOLO_IMGSZ", "1280") or 1280),
            yolo_batch=int(os.getenv("BOWLIQ_YOLO_BATCH", "8") or 8),
            video_encoder=os.getenv("BOWLIQ_VIDEO_ENCODER", "auto").strip().lower(),
            adapt_video=os.getenv("BOWLIQ_ADAPT_VIDEO", "true").strip().lower() not in ("0", "false", "no", "off"),
        )

    @property
    def pose_model_path(self) -> Path:
        return self.models_dir / self.pose_model_name

    @property
    def yolo_model_path(self) -> Path:
        return self.models_dir / self.yolo_model_name

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "bowliq.db"

    @property
    def token_file(self) -> Path:
        return self.storage_dir / ".api_token"

    @property
    def secret_file(self) -> Path:
        return self.storage_dir / ".signing_key"

    def analysis_dir(self, analysis_id: str) -> Path:
        return self.storage_dir / "analyses" / analysis_id


settings = Settings.from_env()


def _read_or_create(path: Path, nbytes: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        val = path.read_text(encoding="utf-8").strip()
        if val:
            return val
    val = secrets.token_urlsafe(nbytes)
    path.write_text(val, encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return val


def api_tokens() -> dict[str, str]:
    """token → owner id. From BOWLIQ_API_TOKENS ("owner:token,owner2:token2"), else one auto-generated owner token."""
    if settings.api_tokens_raw:
        out = {}
        for item in settings.api_tokens_raw.split(","):
            owner, _, tok = item.strip().partition(":")
            if owner and len(tok) >= 16:
                out[tok] = owner
        if out:
            return out
    return {_read_or_create(settings.token_file, 24): "owner"}


def signing_key() -> bytes:
    return _read_or_create(settings.secret_file, 32).encode()

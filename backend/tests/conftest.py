import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# isolate test storage, fixed test credentials, never call Gemini from tests
os.environ["BOWLIQ_STORAGE_DIR"] = tempfile.mkdtemp(prefix="bowliq_test_")
os.environ["GEMINI_API_KEY"] = ""
os.environ["BOWLIQ_API_TOKENS"] = "tester:test-token-0123456789abcdef,other:other-token-0123456789abcdef"
os.environ["BOWLIQ_RATE_ANALYZE_PER_MIN"] = "1000"
os.environ["BOWLIQ_MAX_QUEUED_JOBS"] = "50"
# Pin every setting the tests depend on, so a developer's backend/.env (e.g. a narrower BOWLIQ_ALLOWED_HOSTS without
# the test client's "testserver" host) can't change test behaviour. load_dotenv never overrides existing variables.
os.environ["BOWLIQ_ALLOWED_HOSTS"] = "localhost,127.0.0.1,testserver"
os.environ["BOWLIQ_CORS_ORIGINS"] = "http://localhost:3000,http://127.0.0.1:3000"
os.environ["BOWLIQ_RATE_READ_PER_MIN"] = "100000"
os.environ["BOWLIQ_RETENTION_DAYS"] = "0"
os.environ["BOWLIQ_EMBEDDED_WORKER"] = "true"
os.environ["BOWLIQ_POSE_BACKEND"] = "mediapipe"  # deterministic results whether or not a GPU engine is installed
os.environ["BOWLIQ_VIDEO_ENCODER"] = "x264"

TOKEN = "test-token-0123456789abcdef"  # noqa: S105 (test-only credential)
OTHER = "other-token-0123456789abcdef"

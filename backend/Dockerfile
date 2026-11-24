# BowlIQ backend: non-root, hash-locked dependencies, pinned model baked in.
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libegl1 libgles2 curl \
 && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements.linux.lock .
RUN pip install --require-hashes -r requirements.linux.lock
COPY app ./app
COPY config ./config
COPY scripts ./scripts
COPY samples ./samples
RUN mkdir -p models /data && chown -R app:app /app /data \
 && curl -fsSL -o models/pose_landmarker_heavy.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task \
 && echo "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b  models/pose_landmarker_heavy.task" | sha256sum -c -
USER app
ENV BOWLIQ_STORAGE_DIR=/data
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fsS -H "Host: localhost" http://127.0.0.1:8000/api/v1/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header", "--proxy-headers"]

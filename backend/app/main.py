"""BowlIQ FastAPI application."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.routes import public, router, set_runtime_info
from app.core import jobs
from app.core.config import api_tokens, settings
from app.core.errors import BowlIQError
from app.core.ffmpeg import FFMPEG_MISSING_FIX, ffmpeg_available
from app.core.logging_setup import request_id_var, setup_logging
from app.core.maintenance import cleanup
from app.core.observability import count_request
from app.core.security import BodySizeLimitMiddleware, GuardMiddleware
from app.models.db import get_engine

setup_logging()
log = logging.getLogger("bowliq")


def _startup() -> None:
    if not ffmpeg_available():
        log.error("ffmpeg/ffprobe NOT FOUND on PATH. %s", FFMPEG_MISSING_FIX)
    get_engine()
    n = jobs.recover_interrupted()
    if n:
        log.warning("Marked %d interrupted job(s) as failed (INTERRUPTED).", n)
    cleanup()
    from app.core.ffmpeg import video_encoder
    from app.pipeline.pose import describe_engine, ensure_model

    try:
        ensure_model()  # MediaPipe model is always kept as the fallback engine
        engine, device = describe_engine()  # probed once here, so /health has no side effects
        if engine.startswith("ultralytics"):
            from app.pipeline.pose_yolo import ensure_yolo_model

            ensure_yolo_model()
        set_runtime_info(engine, device, video_encoder())
        log.info("Pose engine: %s on %s; video encoder: %s", engine, device, video_encoder())
    except BowlIQError as exc:
        log.error("%s %s", exc.message, exc.fix)
    if not settings.gemini_api_key:
        log.info("GEMINI_API_KEY not set: reports will use the deterministic template.")
    if not settings.api_tokens_raw:
        token = next(iter(api_tokens()))
        log.warning("Sign in to BowlIQ with this access token (also saved in storage/.api_token): %s", token)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(_startup)
    if settings.embedded_worker:
        jobs.start_embedded_worker()
    yield
    jobs.shutdown()


app = FastAPI(title="BowlIQ API", version="2.0.0", lifespan=lifespan, docs_url="/api/v1/docs",
              openapi_url="/api/v1/openapi.json", redoc_url=None)
app.include_router(public)
app.include_router(router)


class AccessLogMiddleware:
    """Request id propagation, access log line and request counters."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = (scope.get("state") or {}).get("request_id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        status = {"code": 500}

        async def _send(msg):
            if msg["type"] == "http.response.start":
                status["code"] = msg["status"]
            await send(msg)

        try:
            await self.app(scope, receive, _send)
        finally:
            route = getattr(scope.get("route"), "path", "unmatched")
            count_request(scope["method"], route, status["code"])
            if not route.endswith("/health"):
                logging.getLogger("bowliq.access").info("%s %s %s", scope["method"], route, status["code"])
            request_id_var.reset(token)


# Order (outermost first): Guard → TrustedHost → BodySizeLimit → AccessLog → CORS → app
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                   allow_headers=["Authorization", "Content-Type", "Range", "X-BowlIQ-Client", "X-Request-Id"],
                   expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "X-Request-Id"])
app.add_middleware(AccessLogMiddleware)
app.add_middleware(BodySizeLimitMiddleware, max_bytes=2 * settings.max_upload_bytes + 1_000_000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
app.add_middleware(GuardMiddleware)


@app.exception_handler(BowlIQError)
async def _bowliq_error(_: Request, exc: BowlIQError) -> JSONResponse:
    return JSONResponse(exc.to_dict(), status_code=exc.http_status)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    errs = exc.errors()
    first = errs[0] if errs else {}
    field = ".".join(str(x) for x in first.get("loc", []) if x not in ("body", "query", "path"))
    msg = f"{field}: {first.get('msg', 'invalid value')}" if field else str(first.get("msg", "Invalid request."))
    return JSONResponse({"code": "INVALID_REQUEST", "message": msg[:300], "fix": "Check the form fields and try again."},
                        status_code=422)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    msg = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse({"code": f"HTTP_{exc.status_code}", "message": msg,
                         "fix": "Check the URL." if exc.status_code == 404 else "Try again."}, status_code=exc.status_code)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    err_id = uuid.uuid4().hex[:8]
    log.exception("Unhandled error %s on %s %s", err_id, request.method, request.url.path)
    return JSONResponse({"code": "INTERNAL_ERROR", "message": f"Something went wrong (error {err_id}).",
                         "fix": "Try again. If it keeps happening, report this error id with the backend log."},
                        status_code=500)

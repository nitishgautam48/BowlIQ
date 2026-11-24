"""Authentication, signed media links, request guards (host/origin/body size/rate limits) and security headers."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import Header, Query, Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import api_tokens, settings, signing_key
from app.core.errors import BowlIQError

CLIENT_HEADER = "x-bowliq-client"


# ───────────────────────────── authentication ─────────────────────────────

def owner_for_token(token: str) -> str | None:
    for tok, owner in api_tokens().items():
        if secrets.compare_digest(token.encode(), tok.encode()):
            return owner
    return None


def require_user(authorization: str | None = Header(default=None)) -> str:
    token = (authorization or "").removeprefix("Bearer ").strip()
    owner = owner_for_token(token) if token else None
    if owner is None:
        raise BowlIQError("UNAUTHORIZED", "You are not signed in, or your access token is wrong.",
                          "Open the BowlIQ app and sign in with the access token shown in the backend console.", 401)
    return owner


# ───────────────────────────── signed media links ─────────────────────────────
# <video>/<img>/PDF links can't send an Authorization header, so media endpoints also accept a short-lived HMAC
# signature bound to the analysis id and owner.

def sign_media(analysis_id: str, owner: str, ttl_s: int | None = None) -> dict:
    exp = int(time.time()) + (ttl_s or settings.media_link_ttl_s)
    msg = f"{analysis_id}:{owner}:{exp}".encode()
    sig = hmac.new(signing_key(), msg, hashlib.sha256).hexdigest()
    return {"exp": exp, "sig": sig, "owner": owner}


def verify_media(analysis_id: str, owner: str, exp: int, sig: str) -> bool:
    if exp < time.time():
        return False
    msg = f"{analysis_id}:{owner}:{exp}".encode()
    good = hmac.new(signing_key(), msg, hashlib.sha256).hexdigest()
    return secrets.compare_digest(good, sig)


def media_user(request: Request, analysis_id: str, exp: int | None = Query(default=None),
               sig: str | None = Query(default=None), u: str | None = Query(default=None)) -> str:
    """Owner from a Bearer token, or from a valid signed media link."""
    auth = request.headers.get("authorization")
    if auth:
        return require_user(auth)
    if exp is not None and sig and u and verify_media(analysis_id, u, int(exp), sig):
        return u
    raise BowlIQError("UNAUTHORIZED", "This media link is missing or has expired.", "Reload the report page.", 401)


# ───────────────────────────── rate limiting ─────────────────────────────

class RateLimiter:
    """Sliding-window limiter, in memory (per process)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_s: float = 60.0) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                retry = int(window_s - (now - q[0])) + 1
                raise BowlIQError("RATE_LIMITED", "Too many requests.", f"Wait {retry} s and try again.", 429)
            q.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()


# ───────────────────────────── ASGI middleware ─────────────────────────────

def _json(status: int, code: str, message: str, fix: str) -> JSONResponse:
    return JSONResponse({"code": code, "message": message, "fix": fix}, status_code=status)


class BodySizeLimitMiddleware:
    """Rejects bodies above max_bytes from Content-Length up front, and aborts streams that exceed it."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length")
        too_big = _json(413, "FILE_TOO_LARGE", "The upload is larger than the 200 MB limit.",
                        "Trim the clip to one delivery or export at a lower bitrate, then upload again.")
        try:
            if cl is not None and int(cl) > self.max:
                await too_big(scope, receive, send)
                return
        except ValueError:
            await _json(400, "INVALID_REQUEST", "Bad Content-Length header.", "Retry the upload.")(scope, receive, send)
            return
        seen = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max:
                    raise _BodyTooLarge()
            return msg

        async def tracking_send(msg: Message) -> None:
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await too_big(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


class GuardMiddleware:
    """Origin check + client header on state-changing requests (CSRF / drive-by POST protection),
    request ids, and security headers on every response."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
        method = scope["method"]
        rid = headers.get("x-request-id") or uuid.uuid4().hex[:12]
        scope.setdefault("state", {})["request_id"] = rid
        if method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = headers.get("origin")
            if origin and origin not in settings.cors_origins:
                await _json(403, "FORBIDDEN_ORIGIN", "Cross-site request blocked.",
                            "Use the BowlIQ web app.")(scope, receive, send)
                return
            if headers.get(CLIENT_HEADER) != "web":
                await _json(403, "MISSING_CLIENT_HEADER", "Request is missing the BowlIQ client header.",
                            "Use the BowlIQ web app or send 'X-BowlIQ-Client: web'.")(scope, receive, send)
                return

        async def send_with_headers(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                h = list(msg.get("headers") or [])
                ctype = next((v for k, v in h if k.lower() == b"content-type"), b"")
                h += [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                      (b"referrer-policy", b"no-referrer"),
                      (b"cross-origin-resource-policy", b"same-site"), (b"x-request-id", rid.encode())]
                if ctype.startswith(b"application/json"):  # a strict CSP would break the browser's PDF viewer
                    h.append((b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"))
                msg["headers"] = h
            await send(msg)

        await self.app(scope, receive, send_with_headers)

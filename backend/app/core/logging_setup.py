"""Structured (key=value) logging with request ids."""
from __future__ import annotations

import contextvars
import logging
import os

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    level = os.getenv("BOWLIQ_LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if any(getattr(h, "_bowliq", False) for h in root.handlers):
        return
    h = logging.StreamHandler()
    h._bowliq = True  # type: ignore[attr-defined]
    h.setFormatter(logging.Formatter('ts=%(asctime)s level=%(levelname)s logger=%(name)s rid=%(request_id)s msg="%(message)s"'))
    h.addFilter(_RequestIdFilter())
    root.addHandler(h)
    root.setLevel(level)

"""Standalone worker process: `python -m app.worker`. Use with BOWLIQ_EMBEDDED_WORKER=false on the API."""
from __future__ import annotations

import logging
import signal

from app.core import jobs
from app.core.logging_setup import setup_logging
from app.models.db import get_engine
from app.pipeline.pose import ensure_model


def main() -> None:
    setup_logging()
    get_engine()
    ensure_model()
    n = jobs.recover_interrupted()
    if n:
        logging.getLogger("bowliq").warning("Marked %d interrupted job(s) as failed.", n)
    signal.signal(signal.SIGTERM, lambda *_: jobs.shutdown())
    try:
        jobs.worker_loop()
    except KeyboardInterrupt:
        jobs.shutdown()


if __name__ == "__main__":
    main()

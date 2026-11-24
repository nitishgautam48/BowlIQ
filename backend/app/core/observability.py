"""Minimal Prometheus-style counters (no extra dependency)."""
from __future__ import annotations

import threading
from collections import Counter

_lock = threading.Lock()
_requests: Counter = Counter()
_job_outcomes: Counter = Counter()
_stage_seconds: Counter = Counter()


def count_request(method: str, route: str, status: int) -> None:
    with _lock:
        _requests[(method, route, status)] += 1


def count_job(outcome: str) -> None:
    with _lock:
        _job_outcomes[outcome] += 1


def add_stage_time(stage: str, seconds: float) -> None:
    with _lock:
        _stage_seconds[stage] += seconds


def render_metrics(queue: dict[str, int]) -> str:
    lines = ["# TYPE bowliq_http_requests_total counter"]
    with _lock:
        for (m, r, s), n in sorted(_requests.items()):
            lines.append(f'bowliq_http_requests_total{{method="{m}",route="{r}",status="{s}"}} {n}')
        lines.append("# TYPE bowliq_jobs_total counter")
        for o, n in sorted(_job_outcomes.items()):
            lines.append(f'bowliq_jobs_total{{outcome="{o}"}} {n}')
        lines.append("# TYPE bowliq_stage_seconds_total counter")
        for st, sec in sorted(_stage_seconds.items()):
            lines.append(f'bowliq_stage_seconds_total{{stage="{st}"}} {sec:.3f}')
    lines.append("# TYPE bowliq_jobs gauge")
    for st, n in sorted(queue.items()):
        lines.append(f'bowliq_jobs{{status="{st}"}} {n}')
    return "\n".join(lines) + "\n"

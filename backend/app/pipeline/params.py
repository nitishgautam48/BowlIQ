"""Pipeline tunables, loaded once from config/pipeline.yaml (see the comments there for units and meaning).

Every threshold in the analysis code comes from here, so nothing is tuned to one clip inside the code. Helpers turn
the clip-independent units of the file into this clip's frames:

    frames(0.1, fps)        → round(0.1 × fps), at least general.min_frames
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"


class Section(dict):
    """A config section readable as attributes: P.contacts.drop_ratio."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"pipeline.yaml has no setting '{name}'") from exc


def _wrap(v: Any) -> Any:
    if isinstance(v, dict):
        return Section({k: _wrap(x) for k, x in v.items()})
    return v


@lru_cache(maxsize=4)
def load(path: str | None = None) -> Section:
    p = Path(path or os.getenv("BOWLIQ_PIPELINE_CONFIG") or DEFAULT_PATH)
    with open(p, encoding="utf-8") as fh:
        return _wrap(yaml.safe_load(fh))


P = load()


def frames(seconds: float, fps: float, minimum: int | None = None) -> int:
    """Seconds → frames at this clip's rate: round(seconds × fps), never below `minimum` (default general.min_frames)."""
    lo = P.general.min_frames if minimum is None else minimum
    return max(int(lo), int(round(float(seconds) * float(fps))))


def odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

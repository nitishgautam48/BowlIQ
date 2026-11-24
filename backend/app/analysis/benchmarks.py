"""Loads benchmarks.yaml (the single source of metric rules and citations)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PATH = Path(__file__).with_name("benchmarks.yaml")


@lru_cache(maxsize=1)
def benchmarks() -> dict[str, dict]:
    with open(PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["metrics"]


def rule(key: str) -> dict:
    return benchmarks()[key]


def target_text(key: str, baseline: float | None = None, unit: str | None = None) -> str | None:
    r = rule(key)
    u = unit or r["unit"]
    sep = "" if u == "°" else " "
    if r["type"] == "band":
        lo, hi = r["band"]
        return f"{lo:g}–{hi:g}{sep}{u}"
    if r["type"] == "max":
        return f"Under {r['threshold']:g}{sep}{u}"
    if r["type"] == "higher_better":
        if baseline is None:
            return "Personal baseline unlocks after 10 deliveries"
        return f"At or above your baseline ({baseline:.1f}{sep}{u})"
    return None

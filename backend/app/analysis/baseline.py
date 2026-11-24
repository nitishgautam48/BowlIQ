"""Personal baseline: with ≥ 10 previous deliveries, the mean (and SD) of the bowler's top 25 % by ball speed,
or of the most recent 10 when ball speed isn't available."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.config import settings


@dataclass
class MetricStat:
    mean: float
    sd: float
    n: int
    unit: str


@dataclass
class Baseline:
    active: bool
    deliveries: int
    deliveries_until_baseline: int
    based_on: Optional[str] = None
    stats: dict[str, MetricStat] = field(default_factory=dict)


def compute_baseline(history: list[dict]) -> Baseline:
    """history: previous deliveries (newest first), each {"ball_speed_kmh": float|None, "metrics": {key: {"value", "unit"}}}."""
    n = len(history)
    need = settings.baseline_min_deliveries
    if n < need:
        return Baseline(False, n, need - n)
    with_speed = [h for h in history if h.get("ball_speed_kmh") is not None]
    if len(with_speed) >= need:
        k = max(3, int(np.ceil(0.25 * len(with_speed))))
        chosen = sorted(with_speed, key=lambda h: h["ball_speed_kmh"], reverse=True)[:k]
        based_on = f"your fastest {k} of {len(with_speed)} deliveries by ball speed"
    else:
        chosen = history[:need]
        based_on = f"your most recent {need} deliveries"
    stats: dict[str, MetricStat] = {}
    keys = {k for h in chosen for k in (h.get("metrics") or {})}
    for key in keys:
        by_unit: dict[str, list[float]] = {}
        for h in chosen:
            m = (h.get("metrics") or {}).get(key)
            if m and m.get("value") is not None:
                by_unit.setdefault(m["unit"], []).append(float(m["value"]))
        if not by_unit:
            continue
        unit, vals = max(by_unit.items(), key=lambda kv: len(kv[1]))
        if len(vals) < 3:
            continue
        arr = np.array(vals)
        stats[key] = MetricStat(float(arr.mean()), float(arr.std(ddof=1)), len(vals), unit)
    return Baseline(True, n, 0, based_on, stats)

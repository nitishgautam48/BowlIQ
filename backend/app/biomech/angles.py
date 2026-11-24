"""Plain 2D geometry used by every metric. All functions are vectorised over leading axes.

Coordinate convention inside the pipeline: pixels, x to the right (bowling direction after mirroring),
Y *up* (image y flipped), so "vertical" is (0, 1).
"""
from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def joint_angle(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> np.ndarray:
    """Angle ABC in degrees (at vertex b) = arccos(BA·BC / |BA||BC|). 180 = straight line."""
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    u = a - b
    v = c - b
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * v, axis=-1) / (nu * nv)
    cos = np.clip(cos, -1.0, 1.0)
    out = np.degrees(np.arccos(cos))
    return np.where((nu == 0) | (nv == 0), np.nan, out)


def segment_angle_vs_vertical(lower: ArrayLike, upper: ArrayLike, signed: bool = True) -> np.ndarray:
    """Angle (deg) of the segment lower→upper from vertical (Y-up).

    Signed: positive when the upper point is ahead (+x, the bowling direction) of the lower point.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    d = upper - lower
    ang = np.degrees(np.arctan2(d[..., 0], d[..., 1]))  # 0 = straight up, +90 = pointing +x
    return ang if signed else np.abs(ang)


def line_angle(p1: ArrayLike, p2: ArrayLike) -> np.ndarray:
    """Orientation (deg) of the line p1→p2 relative to the +x axis, in (-180, 180]."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    return np.degrees(np.arctan2(d[..., 1], d[..., 0]))


def line_to_line_angle(a1: ArrayLike, a2: ArrayLike, b1: ArrayLike, b2: ArrayLike) -> np.ndarray:
    """Smallest angle (deg, 0–90) between two undirected lines."""
    da = np.asarray(a2, float) - np.asarray(a1, float)
    db = np.asarray(b2, float) - np.asarray(b1, float)
    na = np.linalg.norm(da, axis=-1)
    nb = np.linalg.norm(db, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.abs(np.sum(da * db, axis=-1)) / (na * nb)
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def wrap_deg(a: ArrayLike) -> np.ndarray:
    """Wrap angles to (-180, 180]."""
    a = np.asarray(a, float)
    return (a + 180.0) % 360.0 - 180.0


def angular_velocity(angle_deg: ArrayLike, fps: float) -> np.ndarray:
    """Δangle × fps (deg/s), forward difference, same length (last sample repeated)."""
    a = np.asarray(angle_deg, float)
    if a.size < 2:
        return np.zeros_like(a)
    unwrapped = np.degrees(np.unwrap(np.radians(a)))
    d = np.diff(unwrapped) * fps
    return np.append(d, d[-1])

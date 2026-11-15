"""Validation-only selective prediction and confidence interval helpers."""

from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = correct / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def selective_metrics(labels: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict[str, float | int | None]:
    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have equal length")
    mask = (p >= threshold) | (p <= 1.0 - threshold)
    signals = int(mask.sum())
    correct = int(((p[mask] >= 0.5).astype(int) == y[mask]).sum()) if signals else 0
    lower, upper = wilson_interval(correct, signals)
    return {
        "threshold": float(threshold), "signals": signals,
        "coverage": float(signals / len(y)) if len(y) else 0.0,
        "accuracy": float(correct / signals) if signals else None,
        "accuracy_ci95_lower": lower, "accuracy_ci95_upper": upper,
    }


def meta_label(probabilities: Iterable[float], threshold: float, *, session_allowed: Iterable[bool] | None = None) -> np.ndarray:
    p = np.asarray(list(probabilities), dtype=float)
    act = (p >= threshold) | (p <= 1.0 - threshold)
    if session_allowed is not None:
        act &= np.asarray(list(session_allowed), dtype=bool)
    return act.astype(int)


__all__ = ["meta_label", "selective_metrics", "wilson_interval"]

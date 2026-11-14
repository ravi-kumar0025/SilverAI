"""Timestamp-aware target generation shared by research and live validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_timestamp_targets(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
    horizon_minutes: int,
    threshold_pct: float | None = None,
) -> pd.DataFrame:
    """Generate returns only when the exact future timestamp is observed.

    Rows crossing a missing interval, session break, or other irregular gap are
    left unresolved instead of treating the next row as a fixed duration.
    """

    required = {"datetime_et", "close"}
    if not required.issubset(frame.columns):
        raise ValueError("Target generation requires datetime_et and close columns.")
    if interval_minutes <= 0 or horizon_minutes <= 0 or horizon_minutes % interval_minutes:
        raise ValueError("horizon_minutes must be a positive multiple of interval_minutes.")
    ordered = frame.copy().sort_values("datetime_et", kind="mergesort").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["datetime_et"], utc=True)
    closes = pd.to_numeric(ordered["close"], errors="coerce")
    step = pd.Timedelta(minutes=interval_minutes)
    horizon = pd.Timedelta(minutes=horizon_minutes)
    bars = horizon_minutes // interval_minutes
    index = pd.DatetimeIndex(timestamps)
    future_positions = index.get_indexer(index + horizon)
    positions = np.arange(len(ordered))
    valid = (future_positions == positions + bars)
    intervals = index.to_series().diff().to_numpy()
    for offset in range(1, bars + 1):
        valid[: len(ordered) - offset] &= intervals[offset:] == step
    valid &= np.isfinite(closes.to_numpy(dtype=float))
    close_values = closes.to_numpy(dtype=float)
    valid &= close_values != 0
    returns = np.full(len(ordered), np.nan, dtype=float)
    valid_indices = np.flatnonzero(valid)
    returns[valid_indices] = (close_values[future_positions[valid_indices]] - close_values[valid_indices]) / close_values[valid_indices] * 100.0
    ordered["future_return_pct"] = returns
    ordered["target_valid"] = valid
    if threshold_pct is not None:
        if threshold_pct < 0:
            raise ValueError("threshold_pct must be nonnegative.")
        ordered["target_direction"] = np.select(
            [returns > threshold_pct, returns < -threshold_pct], [1, -1], default=0
        ).astype("int64")
        ordered.loc[~valid, "target_direction"] = pd.NA
    return ordered


__all__ = ["generate_timestamp_targets"]

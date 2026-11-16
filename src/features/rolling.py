"""Restartable, closed-candle rolling feature construction for inference."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from src.project_config import CONFIG
from src.data.market_data_contracts import Candle
from src.features.market_features import SPOT_FEATURE_COLUMNS, generate_spot_features


class RollingFeatureBuilder:
    """Maintain causal OHLCV context and emit one complete feature row."""

    def __init__(self, *, interval_minutes: int = CONFIG.candle_minutes, max_history: int = 256):
        if interval_minutes <= 0 or max_history < 201:
            raise ValueError("interval_minutes must be positive and max_history must cover the 200-bar warm-up.")
        self.interval = pd.Timedelta(minutes=interval_minutes)
        self.max_history = int(max_history)
        self._candles: list[dict[str, Any]] = []

    @staticmethod
    def _row(candle: Candle | dict[str, Any]) -> dict[str, Any]:
        if isinstance(candle, Candle):
            return {"timestamp": candle.timestamp, "silver_open": candle.open, "silver_high": candle.high, "silver_low": candle.low, "silver_close": candle.close}
        required = {"timestamp", "silver_open", "silver_high", "silver_low", "silver_close", "xauusd_close", "eurusd_close"}
        if not required.issubset(candle):
            raise ValueError("Spot rolling history requires timestamp, silver OHLC, XAU/USD close, and EUR/USD close.")
        return {key: candle[key] for key in required}

    def append(self, candle: Candle | dict[str, Any]) -> pd.DataFrame | None:
        row = self._row(candle)
        timestamp = pd.Timestamp(row["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(CONFIG.timezone)
        timestamp = timestamp.tz_convert(CONFIG.timezone)
        row["timestamp"] = timestamp
        if self._candles:
            previous = pd.Timestamp(self._candles[-1]["timestamp"])
            if timestamp == previous:
                raise ValueError("Duplicate candle timestamp in rolling history.")
            if timestamp < previous:
                raise ValueError("Rolling candle history requires chronological timestamps.")
            if timestamp - previous != self.interval:
                # A session gap or missing candle invalidates the rolling window.
                self._candles = []
        self._candles.append(row)
        self._candles = self._candles[-self.max_history :]
        if len(self._candles) < 201:
            return None
        frame = generate_spot_features(pd.DataFrame(self._candles))
        if any(column not in frame.columns for column in SPOT_FEATURE_COLUMNS):
            return None
        values = frame.iloc[-1][list(SPOT_FEATURE_COLUMNS)]
        if values.isna().any() or not np.isfinite(values.astype(float)).all():
            return None
        return frame.iloc[[-1]][list(SPOT_FEATURE_COLUMNS)].copy()

    def restore_history(self, candles: Iterable[Candle | dict[str, Any]]) -> None:
        self._candles = []
        for candle in candles:
            self.append(candle)

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._candles]

    @property
    def history_length(self) -> int:
        return len(self._candles)


__all__ = ["RollingFeatureBuilder"]

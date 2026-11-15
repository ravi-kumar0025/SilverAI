"""Automatic closed-candle prediction orchestration.

This module is provider-neutral.  A real authenticated provider can feed
normalized updates into it without changing model, storage, or API code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from src.config import CONFIG
from src.data.market_data import Candle, ClosedCandleBuilder, MarketUpdate


class ClosedCandleOrchestrator:
    def __init__(
        self,
        feature_builder: Callable[[list[Candle]], pd.DataFrame],
        predict_closed_candle: Callable[[pd.DataFrame, Candle], dict[str, Any]],
        *,
        interval_minutes: int = CONFIG.candle_minutes,
    ) -> None:
        self.builder = ClosedCandleBuilder(interval_minutes)
        self.feature_builder = feature_builder
        self.predict_closed_candle = predict_closed_candle
        self._seen: set[tuple[str, str, str]] = set()

    def on_update(self, update: MarketUpdate) -> dict[str, Any] | None:
        closed = self.builder.update(update)
        if closed is None:
            return None
        return self._predict(closed)

    def _predict(self, candle: Candle) -> dict[str, Any] | None:
        key = (candle.instrument, candle.contract, candle.timestamp)
        if key in self._seen:
            return None
        append = getattr(self.feature_builder, "append", None)
        features = append(candle) if callable(append) else self.feature_builder([candle])
        if features is None:
            return None
        if not isinstance(features, pd.DataFrame) or len(features) != 1:
            raise ValueError("feature_builder must return exactly one feature row per closed candle.")
        result = self.predict_closed_candle(features, candle)
        self._seen.add(key)
        return result

    def restore(self, records: Iterable[dict[str, Any]]) -> None:
        """Restore deduplication keys after a process restart."""
        self._seen = {
            (str(row["instrument"]), str(row["contract"]), str(row["candle_timestamp"]))
            for row in records
            if row.get("instrument") and row.get("contract") and row.get("candle_timestamp")
        }


__all__ = ["ClosedCandleOrchestrator"]

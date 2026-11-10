"""Provider-neutral market-data contracts and the checked-in historical provider.

The historical provider is intentionally explicit: it is useful for replay and
development, but it never reports itself as a live feed. A live vendor adapter
can implement the same protocol without changing the chart or prediction code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from src.data.loader import load_5m_data
from src.data.preprocessing import validate_ohlcv


@dataclass(frozen=True)
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(Protocol):
    name: str
    is_live: bool

    def candles(self, limit: int = 288) -> list[Candle]: ...


def validate_candles(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate, and validate OHLCV without fabricating rows."""
    required = {"datetime_et", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {', '.join(sorted(missing))}.")
    ordered = frame.copy().sort_values("datetime_et", kind="stable")
    ordered = ordered.drop_duplicates("datetime_et", keep="last").reset_index(drop=True)
    valid = validate_ohlcv(ordered[list(required - {"datetime_et"})])
    if not valid["valid"].all():
        raise ValueError("Market data contains invalid OHLCV rows.")
    return ordered


class HistoricalCsvProvider:
    name = "silver_5m.csv"
    is_live = False

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = data_dir

    def frame(self) -> pd.DataFrame:
        return validate_candles(load_5m_data(self.data_dir))

    def candles(self, limit: int = 288) -> list[Candle]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        frame = self.frame().tail(limit)
        return [
            Candle(
                timestamp=row.datetime_et.isoformat(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for row in frame.itertuples(index=False)
        ]


def provider_status(provider: MarketDataProvider) -> dict[str, object]:
    return {
        "provider": provider.name,
        "is_live": provider.is_live,
        "status": "historical" if not provider.is_live else "connected",
        "last_update": datetime.now().astimezone().isoformat(),
    }


__all__ = ["Candle", "HistoricalCsvProvider", "MarketDataProvider", "provider_status", "validate_candles"]

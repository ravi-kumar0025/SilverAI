"""Provider-neutral market-data contracts and the checked-in historical provider.

The historical provider is intentionally explicit: it is useful for replay and
development, but it never reports itself as a live feed. A live vendor adapter
can implement the same protocol without changing the chart or prediction code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

import pandas as pd

from src.data.loader import load_5m_data
from src.data.preprocessing import validate_ohlcv
from src.config import CONFIG


@dataclass(frozen=True)
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    instrument: str = CONFIG.instrument
    contract: str = CONFIG.symbol


class MarketDataProvider(Protocol):
    name: str
    is_live: bool

    def candles(self, limit: int = 288) -> list[Candle]: ...

    def status(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MarketUpdate:
    """Normalized trade/quote update consumed by a candle builder."""

    timestamp: pd.Timestamp
    price: float
    volume: float = 0.0
    instrument: str = CONFIG.instrument
    contract: str = CONFIG.symbol


class ClosedCandleBuilder:
    """Build time-aligned candles and emit only completed intervals.

    The builder never emits the currently open interval.  A caller must pass
    the next update (or call ``flush`` at shutdown) before a completed candle
    is returned, which prevents unfinished data from reaching inference.
    """

    def __init__(self, interval_minutes: int = CONFIG.candle_minutes):
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive.")
        self.interval = pd.Timedelta(minutes=interval_minutes)
        self._current: dict[str, Any] | None = None

    def _bucket(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        value = pd.Timestamp(timestamp)
        if value.tzinfo is None:
            value = value.tz_localize(CONFIG.timezone)
        return value.tz_convert(CONFIG.timezone).floor(self.interval)

    def update(self, update: MarketUpdate) -> Candle | None:
        timestamp = pd.Timestamp(update.timestamp)
        bucket = self._bucket(timestamp)
        price = float(update.price)
        volume = float(update.volume)
        if not all(pd.notna([price, volume])) or price <= 0 or volume < 0:
            raise ValueError("Market updates require finite positive price and nonnegative volume.")
        if self._current is None:
            self._current = {"timestamp": bucket, "open": price, "high": price, "low": price, "close": price, "volume": volume, "instrument": update.instrument, "contract": update.contract}
            return None
        if bucket == self._current["timestamp"]:
            self._current["high"] = max(self._current["high"], price)
            self._current["low"] = min(self._current["low"], price)
            self._current["close"] = price
            self._current["volume"] += volume
            return None
        if bucket < self._current["timestamp"]:
            raise ValueError("Market updates must arrive in chronological order.")
        closed = Candle(**{key: value.isoformat() if key == "timestamp" else value for key, value in self._current.items()})
        self._current = {"timestamp": bucket, "open": price, "high": price, "low": price, "close": price, "volume": volume, "instrument": update.instrument, "contract": update.contract}
        return closed

    def flush(self) -> None:
        """Discard an open candle; incomplete intervals must not be predicted."""
        self._current = None


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
                instrument=CONFIG.instrument,
                contract=CONFIG.symbol,
            )
            for row in frame.itertuples(index=False)
        ]


def provider_status(provider: MarketDataProvider) -> dict[str, object]:
    if hasattr(provider, "status"):
        return provider.status()
    last_update = None
    if not provider.is_live:
        try:
            frame = provider.frame()  # type: ignore[attr-defined]
            last_update = frame["datetime_et"].iloc[-1].isoformat()
        except (AttributeError, IndexError, KeyError, OSError, ValueError):
            last_update = None
    return {
        "provider": provider.name,
        "is_live": provider.is_live,
        "status": "historical" if not provider.is_live else "connected",
        "last_update": last_update,
        "data_mode": "historical" if not provider.is_live else "live",
        "instrument": CONFIG.instrument,
        "contract": CONFIG.symbol,
    }


__all__ = ["Candle", "ClosedCandleBuilder", "HistoricalCsvProvider", "MarketDataProvider", "MarketUpdate", "provider_status", "validate_candles"]

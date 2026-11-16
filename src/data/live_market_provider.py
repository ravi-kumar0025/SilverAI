"""Dukascopy live-bridge state and deterministic replay source.

The network poller intentionally lives in ``scripts/live_feed_logger.mjs``.
This module is the server-side contract: it accepts already-normalized closed
one-minute candles, rejects forming/duplicate data, and exposes freshness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from src.project_config import CONFIG
from src.market_model_config import SPOT_CONFIG


LIVE_SYMBOLS = {"xagusd", "xauusd", "eurusd"}


def _utc(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def _minute(value: Any) -> datetime:
    return _utc(value).replace(second=0, microsecond=0)


@dataclass
class LiveFeedState:
    provider: str = "dukascopy-realtime"
    interval: str = "m1"
    price_type: str = "bid"
    volumes: bool = False
    connected: bool = False
    last_update: str | None = None
    last_heartbeat: str | None = None
    reconnect_attempts: int = 0
    forming_discarded: int = 0
    duplicate_discarded: int = 0
    gap_events: int = 0
    outage_events: int = 0
    accepted_candles: int = 0
    stale_after_seconds: int = SPOT_CONFIG.live_stale_after_seconds
    _last_by_symbol: dict[str, datetime] = field(default_factory=dict, repr=False)
    _seen: set[tuple[str, str]] = field(default_factory=set, repr=False)

    def heartbeat(self, now: Any = None) -> None:
        self.last_heartbeat = _utc(now or datetime.now(timezone.utc)).isoformat()

    def freshness(self, now: Any = None) -> dict[str, Any]:
        current = _utc(now or datetime.now(timezone.utc))
        last = _utc(self.last_update) if self.last_update else None
        age = None if last is None else max(0.0, (current - last).total_seconds())
        return {
            "is_live": self.connected,
            "fresh": bool(self.connected and age is not None and age <= self.stale_after_seconds),
            "last_update": self.last_update,
            "last_heartbeat": self.last_heartbeat,
            "age_seconds": age,
            "stale_after_seconds": self.stale_after_seconds,
        }

    def as_dict(self, now: Any = None) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "status": "connected" if self.connected else "disconnected",
            "data_mode": "live",
            "instrument": CONFIG.instrument,
            "contract": CONFIG.symbol,
            "interval": self.interval,
            "price_type": self.price_type,
            "volumes": self.volumes,
            "reconnect_attempts": self.reconnect_attempts,
            "forming_discarded": self.forming_discarded,
            "duplicate_discarded": self.duplicate_discarded,
            "gap_events": self.gap_events,
            "outage_events": self.outage_events,
            "accepted_candles": self.accepted_candles,
            "companion_missing": sorted(LIVE_SYMBOLS - set(self._last_by_symbol)),
            **self.freshness(now),
        }


class DukascopyRealtimeSource:
    """Server-side receiver for the Node Dukascopy realtime bridge."""

    name = "dukascopy-realtime"
    is_live = True

    def __init__(self, on_candle: Callable[[dict[str, Any]], Any] | None = None, *, stale_after_seconds: int | None = None):
        self.state = LiveFeedState(stale_after_seconds=stale_after_seconds or SPOT_CONFIG.live_stale_after_seconds)
        self._on_candle = on_candle

    def status(self, now: Any = None) -> dict[str, Any]:
        return self.state.as_dict(now)

    def heartbeat(self, *, now: Any = None, connected: bool = True) -> dict[str, Any]:
        self.state.heartbeat(now)
        if connected:
            self.state.connected = True
        else:
            if self.state.connected:
                self.state.outage_events += 1
            self.state.connected = False
        return self.status(now)

    def ingest(self, payload: Mapping[str, Any], *, now: Any = None) -> dict[str, Any]:
        symbol = str(payload.get("symbol") or payload.get("instrument") or "").lower()
        if symbol not in LIVE_SYMBOLS:
            raise ValueError(f"Unsupported live symbol: {symbol or 'missing'}")
        timestamp = _minute(payload.get("timestamp"))
        current = _utc(now or datetime.now(timezone.utc))
        if bool(payload.get("forming")) or timestamp >= _minute(current):
            self.state.forming_discarded += 1
            return {"accepted": False, "reason": "forming_candle", "status": self.status(current)}
        key = (symbol, timestamp.isoformat())
        if key in self.state._seen:
            self.state.duplicate_discarded += 1
            return {"accepted": False, "reason": "duplicate", "status": self.status(current)}
        previous = self.state._last_by_symbol.get(symbol)
        if previous is not None and timestamp - previous > timedelta(minutes=1):
            self.state.gap_events += 1
        self.state._seen.add(key)
        self.state._last_by_symbol[symbol] = timestamp
        self.state.connected = True
        self.state.last_update = timestamp.isoformat()
        self.state.heartbeat(current)
        self.state.accepted_candles += 1
        candle = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "open": float(payload["open"]),
            "high": float(payload["high"]),
            "low": float(payload["low"]),
            "close": float(payload["close"]),
            "source": self.name,
        }
        if self._on_candle:
            self._on_candle(candle)
        return {"accepted": True, "candle": candle, "status": self.status(current)}


class ReplaySource:
    """Deterministic source for the same receiver path used by live candles."""

    def __init__(self, events: Iterable[Mapping[str, Any]], *, now: Any = None, stale_after_seconds: int | None = None, on_candle: Callable[[dict[str, Any]], Any] | None = None):
        self.events = list(events)
        self.source = DukascopyRealtimeSource(on_candle=on_candle, stale_after_seconds=stale_after_seconds)
        self.now = _utc(now or datetime.now(timezone.utc))

    def run(self) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        for event in self.events:
            result = self.source.ingest(event, now=self.now)
            if result["accepted"]:
                accepted.append(result["candle"])
        return accepted

    def status(self) -> dict[str, Any]:
        return self.source.status(self.now)


# Compatibility name for callers that imported the old stub.
ConfiguredLiveProvider = DukascopyRealtimeSource


__all__ = ["ConfiguredLiveProvider", "DukascopyRealtimeSource", "LiveFeedState", "ReplaySource", "LIVE_SYMBOLS"]

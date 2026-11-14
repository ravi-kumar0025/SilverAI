"""Credential-gated live-provider boundary.

The adapter normalizes provider configuration and status without pretending to
be connected. A vendor transport can call ``ingest`` with normalized candles
once credentials and entitlements are supplied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from src.config import CONFIG
from src.data.market_data import Candle


@dataclass(frozen=True)
class LiveProviderConfig:
    provider: str
    api_key: str | None
    websocket_url: str | None
    symbol: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.websocket_url)


class ConfiguredLiveProvider:
    name = "configured-live-provider"
    is_live = False

    def __init__(self, config: LiveProviderConfig | None = None):
        self.config = config or LiveProviderConfig(
            provider=os.getenv("SILVERAI_LIVE_PROVIDER", "unset"),
            api_key=os.getenv("SILVERAI_LIVE_API_KEY"),
            websocket_url=os.getenv("SILVERAI_LIVE_WS_URL"),
            symbol=os.getenv("SILVERAI_LIVE_SYMBOL", CONFIG.symbol),
        )
        self._connected = False
        self._last_update: str | None = None
        self._on_candle: Callable[[Candle], Any] | None = None

    def connect(self, on_candle: Callable[[Candle], Any] | None = None) -> None:
        if not self.config.configured:
            raise RuntimeError("Live provider requires SILVERAI_LIVE_API_KEY and SILVERAI_LIVE_WS_URL.")
        self._on_candle = on_candle
        raise RuntimeError("Live transport is configured but not connected; supply the authorized vendor transport adapter.")

    def ingest(self, candle: Candle) -> Any:
        if candle.instrument != CONFIG.instrument or candle.contract != self.config.symbol:
            raise ValueError("Live candle instrument or contract does not match the SilverAI contract.")
        self._last_update = candle.timestamp
        return self._on_candle(candle) if self._on_candle else None

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.config.provider,
            "is_live": self._connected,
            "status": "connected" if self._connected else "configuration_required" if not self.config.configured else "disconnected",
            "data_mode": "live",
            "instrument": CONFIG.instrument,
            "contract": self.config.symbol,
            "last_update": self._last_update,
            "credentials_configured": self.config.configured,
            "message": "Provide authorized provider credentials and transport setup before live use.",
        }


__all__ = ["ConfiguredLiveProvider", "LiveProviderConfig"]

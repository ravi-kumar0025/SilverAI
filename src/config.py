"""Central runtime configuration for the SilverAI evaluation contract."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SilverAIConfig:
    instrument: str = "COMEX Silver Futures"
    symbol: str = "SI"
    candle_minutes: int = 5
    forecast_horizon_bars: int = 3
    timezone: str = "America/New_York"
    evaluation_days: int = 5

    @property
    def forecast_horizon_minutes(self) -> int:
        return self.candle_minutes * self.forecast_horizon_bars


def load_config() -> SilverAIConfig:
    """Load safe, integer runtime overrides without scattering constants."""

    candle_minutes = int(os.getenv("SILVERAI_CANDLE_MINUTES", "5"))
    horizon_bars = int(os.getenv("SILVERAI_HORIZON_BARS", "3"))
    evaluation_days = int(os.getenv("SILVERAI_EVALUATION_DAYS", "5"))
    if candle_minutes <= 0 or horizon_bars <= 0 or evaluation_days <= 0:
        raise ValueError("SilverAI time configuration must contain positive integers.")
    return SilverAIConfig(
        candle_minutes=candle_minutes,
        forecast_horizon_bars=horizon_bars,
        evaluation_days=evaluation_days,
    )


CONFIG = load_config()

__all__ = ["CONFIG", "SilverAIConfig", "load_config"]

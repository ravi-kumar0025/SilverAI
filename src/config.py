"""Central runtime configuration for the SilverAI evaluation contract."""

from __future__ import annotations

import os
import hashlib
import json


APPROVED_FEATURES = (
    "return_1", "return_3", "return_6", "volatility_5", "volatility_20", "volatility_50",
    "sma_distance_20", "sma_distance_50", "candle_return", "candle_range_pct",
    "upper_wick_ratio", "lower_wick_ratio", "relative_volume", "volume_change",
    "hour_sin", "hour_cos", "day_of_week",
)
from dataclasses import dataclass


@dataclass(frozen=True)
class SilverAIConfig:
    instrument: str = "COMEX Silver Futures"
    symbol: str = "SI"
    candle_minutes: int = 5
    forecast_horizon_bars: int = 3
    timezone: str = "America/New_York"
    evaluation_days: int = 5
    neutral_quantile: float = 0.40
    calibration_fraction: float = 0.20
    random_seed: int = 42
    classification_target: str = "target_direction_15m"
    regression_target: str = "future_return_15m"
    threshold_definition: str = "40th percentile of absolute training returns"
    feature_names: tuple[str, ...] = APPROVED_FEATURES

    @property
    def forecast_horizon_minutes(self) -> int:
        return self.candle_minutes * self.forecast_horizon_bars

    def as_dict(self) -> dict[str, object]:
        return {
            "instrument": self.instrument, "symbol": self.symbol,
            "candle_minutes": self.candle_minutes, "forecast_horizon_bars": self.forecast_horizon_bars,
            "forecast_horizon_minutes": self.forecast_horizon_minutes, "timezone": self.timezone,
            "evaluation_days": self.evaluation_days, "neutral_quantile": self.neutral_quantile,
            "calibration_fraction": self.calibration_fraction, "random_seed": self.random_seed,
            "classification_target": self.classification_target, "regression_target": self.regression_target,
            "threshold_definition": self.threshold_definition,
            "feature_names": list(self.feature_names),
        }

    @property
    def version(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def load_config() -> SilverAIConfig:
    """Load safe, integer runtime overrides without scattering constants."""

    candle_minutes = int(os.getenv("SILVERAI_CANDLE_MINUTES", "5"))
    horizon_bars = int(os.getenv("SILVERAI_HORIZON_BARS", "3"))
    evaluation_days = int(os.getenv("SILVERAI_EVALUATION_DAYS", "5"))
    neutral_quantile = float(os.getenv("SILVERAI_NEUTRAL_QUANTILE", "0.40"))
    calibration_fraction = float(os.getenv("SILVERAI_CALIBRATION_FRACTION", "0.20"))
    if candle_minutes <= 0 or horizon_bars <= 0 or evaluation_days <= 0:
        raise ValueError("SilverAI time configuration must contain positive integers.")
    if not 0 < neutral_quantile < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("SilverAI quantile and calibration fraction must be between zero and one.")
    return SilverAIConfig(
        candle_minutes=candle_minutes,
        forecast_horizon_bars=horizon_bars,
        evaluation_days=evaluation_days,
        neutral_quantile=neutral_quantile,
        calibration_fraction=calibration_fraction,
    )


CONFIG = load_config()

__all__ = ["APPROVED_FEATURES", "CONFIG", "SilverAIConfig", "load_config"]

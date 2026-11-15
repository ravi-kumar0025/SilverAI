"""Central runtime configuration for the SilverAI evaluation contract."""

from __future__ import annotations

import os
import hashlib
import json
from src.features.spot_features import SPOT_FEATURE_COLUMNS


APPROVED_FEATURES = SPOT_FEATURE_COLUMNS
from dataclasses import dataclass


@dataclass(frozen=True)
class SilverAIConfig:
    instrument: str = "XAG/USD Spot"
    symbol: str = "XAG/USD"
    candle_minutes: int = 5
    forecast_horizon_bars: int = 12
    timezone: str = "UTC"
    evaluation_days: int = 5
    neutral_quantile: float = 0.40
    calibration_fraction: float = 0.20
    random_seed: int = 42
    classification_target: str = "binary_atr_scaled_direction"
    regression_target: str | None = None
    threshold_definition: str = "Up/Down outside k*ATR_pct; no-move rows dropped"
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
    horizon_bars = int(os.getenv("SILVERAI_HORIZON_BARS", "12"))
    evaluation_days = int(os.getenv("SILVERAI_EVALUATION_DAYS", "5"))
    neutral_quantile = float(os.getenv("SILVERAI_NEUTRAL_QUANTILE", "0.40"))
    calibration_fraction = float(os.getenv("SILVERAI_CALIBRATION_FRACTION", "0.20"))
    if candle_minutes <= 0 or horizon_bars <= 0 or evaluation_days <= 0:
        raise ValueError("SilverAI time configuration must contain positive integers.")
    if not 0 < neutral_quantile < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("SilverAI quantile and calibration fraction must be between zero and one.")
    return SilverAIConfig(
        instrument=os.getenv("SILVERAI_INSTRUMENT", "XAGUSD_SPOT"),
        symbol=os.getenv("SILVERAI_SYMBOL", "XAG/USD"),
        candle_minutes=candle_minutes,
        forecast_horizon_bars=horizon_bars,
        evaluation_days=evaluation_days,
        neutral_quantile=neutral_quantile,
        calibration_fraction=calibration_fraction,
    )


CONFIG = load_config()

__all__ = ["APPROVED_FEATURES", "CONFIG", "SilverAIConfig", "load_config"]

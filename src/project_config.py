"""Compatibility view of the single active configuration contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from src.features.market_features import SPOT_FEATURE_COLUMNS
from src.market_model_config import SPOT_CONFIG, load_spot_config


APPROVED_FEATURES = SPOT_FEATURE_COLUMNS


@dataclass(frozen=True)
class SilverAIConfig:
    instrument: str
    symbol: str
    candle_minutes: int
    forecast_horizon_bars: int
    timezone: str
    evaluation_days: int
    neutral_quantile: float
    calibration_fraction: float
    random_seed: int
    classification_target: str
    regression_target: str | None
    threshold_definition: str
    feature_names: tuple[str, ...]

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
            "threshold_definition": self.threshold_definition, "feature_names": list(self.feature_names),
        }

    @property
    def version(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def load_config() -> SilverAIConfig:
    spot = load_spot_config()
    return SilverAIConfig(
        instrument=spot.instrument, symbol=spot.symbol, candle_minutes=spot.candle_minutes,
        forecast_horizon_bars=spot.forecast_horizon_bars, timezone=spot.timezone,
        evaluation_days=spot.evaluation_days, neutral_quantile=spot.neutral_quantile,
        calibration_fraction=0.20, random_seed=spot.random_seed,
        classification_target="15-minute_down_neutral_up", regression_target="log(close[t+3]/close[t])",
        threshold_definition="Calibrated top-class confidence with validation-selected neutral emission threshold",
        feature_names=APPROVED_FEATURES,
    )


CONFIG = load_config()

__all__ = ["APPROVED_FEATURES", "CONFIG", "SilverAIConfig", "load_config"]

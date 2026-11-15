"""Configuration for the isolated Dukascopy XAG/USD spot pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


@dataclass(frozen=True)
class SpotConfig:
    instrument: str
    symbol: str
    data_source: str
    price_side: str
    data_root: Path
    dataset_manifest: Path
    timezone: str
    candle_minutes: int
    candidate_horizons_bars: tuple[int, ...]
    target_k_values: tuple[float, ...]
    target_k: float
    forecast_horizon_bars: int
    purge_bars: int
    embargo_bars: int
    holdout_fraction: float
    confidence_threshold: float
    feature_groups: tuple[str, ...]
    feature_set_version: str
    model_version: str
    final_holdout_start: str
    final_holdout_end: str
    spread_bps: float
    slippage_bps: float

    @property
    def forecast_horizon_minutes(self) -> int:
        return self.candle_minutes * self.forecast_horizon_bars

    @property
    def root_data(self) -> Path:
        return ROOT / self.data_root

    @property
    def dataset_id(self) -> str:
        manifest = ROOT / self.dataset_manifest
        if not manifest.is_file():
            return "manifest-missing"
        return str(json.loads(manifest.read_text(encoding="utf-8")).get("dataset_id", "manifest-invalid"))


def load_spot_config(path: str | Path = CONFIG_PATH) -> SpotConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SpotConfig(
        instrument=str(raw["instrument"]), symbol=str(raw["symbol"]), data_source=str(raw["data_source"]), price_side=str(raw["price_side"]),
        data_root=Path(raw["data_root"]), dataset_manifest=Path(raw["dataset_manifest"]), timezone=str(raw["timezone"]), candle_minutes=int(raw["candle_minutes"]),
        candidate_horizons_bars=tuple(int(x) for x in raw["candidate_horizons_bars"]),
        target_k_values=tuple(float(x) for x in raw["target_k_values"]), target_k=float(raw["target_k"]),
        forecast_horizon_bars=int(raw["forecast_horizon_bars"]), purge_bars=int(raw["purge_bars"]),
        embargo_bars=int(raw["embargo_bars"]), holdout_fraction=float(raw["holdout_fraction"]),
        confidence_threshold=float(raw["confidence_threshold"]), feature_groups=tuple(raw["feature_groups"]),
        feature_set_version=str(raw["feature_set_version"]), model_version=str(raw["model_version"]),
        final_holdout_start=str(raw["final_holdout_start"]), final_holdout_end=str(raw["final_holdout_end"]),
        spread_bps=float(raw["costs"]["spread_bps"]), slippage_bps=float(raw["costs"]["slippage_bps"]),
    )


SPOT_CONFIG = load_spot_config()

__all__ = ["CONFIG_PATH", "SPOT_CONFIG", "SpotConfig", "load_spot_config"]

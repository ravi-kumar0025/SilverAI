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
    candidate_timeframes_minutes: tuple[int, ...]
    target_k_values: tuple[float, ...]
    target_k: float
    forecast_horizon_bars: int
    purge_bars: int
    embargo_bars: int
    validation_folds: int
    label_methods: tuple[str, ...]
    holdout_fraction: float
    confidence_threshold: float
    confidence_thresholds: tuple[float, ...]
    random_seed: int
    evaluation_days: int
    neutral_quantile: float
    model_root: Path
    regression_model_root: Path
    selection_path: Path
    prediction_db_path: Path
    frozen_evaluation_path: Path
    holdout_status: str
    holdout_ledger_path: Path
    live_stale_after_seconds: int
    price_columns: dict[str, str]
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
    def root_model(self) -> Path:
        return ROOT / self.model_root

    @property
    def root_regression_model(self) -> Path:
        return ROOT / self.regression_model_root

    @property
    def root_selection(self) -> Path:
        return ROOT / self.selection_path

    @property
    def root_prediction_db(self) -> Path:
        return ROOT / self.prediction_db_path

    @property
    def root_frozen_evaluation(self) -> Path:
        return ROOT / self.frozen_evaluation_path

    @property
    def root_holdout_ledger(self) -> Path:
        return ROOT / self.holdout_ledger_path

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
        candidate_timeframes_minutes=tuple(int(x) for x in raw["candidate_timeframes_minutes"]),
        target_k_values=tuple(float(x) for x in raw["target_k_values"]), target_k=float(raw["target_k"]),
        forecast_horizon_bars=int(raw["forecast_horizon_bars"]), purge_bars=int(raw["purge_bars"]),
        embargo_bars=int(raw["embargo_bars"]), validation_folds=int(raw["validation_folds"]),
        label_methods=tuple(str(x) for x in raw["label_methods"]), holdout_fraction=float(raw["holdout_fraction"]),
        confidence_threshold=float(raw["confidence_threshold"]), feature_groups=tuple(raw["feature_groups"]),
        feature_set_version=str(raw["feature_set_version"]), model_version=str(raw["model_version"]),
        final_holdout_start=str(raw["final_holdout_start"]), final_holdout_end=str(raw["final_holdout_end"]),
        spread_bps=float(raw["costs"]["spread_bps"]), slippage_bps=float(raw["costs"]["slippage_bps"]),
        confidence_thresholds=tuple(float(x) for x in raw["confidence_thresholds"]),
        random_seed=int(raw["random_seed"]), evaluation_days=int(raw["evaluation_days"]),
        neutral_quantile=float(raw["neutral_quantile"]), model_root=Path(raw["model_root"]), regression_model_root=Path(raw["regression_model_root"]),
        selection_path=Path(raw["selection_path"]), prediction_db_path=Path(raw["prediction_db_path"]),
        frozen_evaluation_path=Path(raw["frozen_evaluation_path"]), holdout_status=str(raw["holdout_status"]),
        holdout_ledger_path=Path(raw["holdout_ledger_path"]),
        live_stale_after_seconds=int(raw["live_stale_after_seconds"]),
        price_columns={str(key): str(value) for key, value in raw["price_columns"].items()},
    )


SPOT_CONFIG = load_spot_config()

__all__ = ["CONFIG_PATH", "SPOT_CONFIG", "SpotConfig", "load_spot_config"]

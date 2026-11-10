"""Manifest and guardrails for the submitted, no-change live evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CONFIG


@dataclass(frozen=True)
class FrozenEvaluationManifest:
    asset: str
    symbol: str
    option: str
    candle_minutes: int
    horizon_bars: int
    horizon_minutes: int
    evaluation_trading_days: int
    model_version: str
    configuration_version: str
    frozen_at: str


def configuration_version(metadata: dict[str, Any]) -> str:
    payload = {
        "feature_names": metadata.get("feature_names"),
        "threshold_value_pct": metadata.get("threshold_value_pct"),
        "prediction_horizon_minutes": metadata.get("prediction_horizon_minutes"),
        "prediction_horizon_bars": metadata.get("prediction_horizon_bars"),
        "classification_target": metadata.get("classification_target"),
        "regression_target": metadata.get("regression_target"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_manifest(metadata: dict[str, Any], model_version: str) -> FrozenEvaluationManifest:
    horizon_bars = int(metadata.get("prediction_horizon_bars", CONFIG.forecast_horizon_bars))
    horizon_minutes = int(metadata.get("prediction_horizon_minutes", CONFIG.forecast_horizon_minutes))
    if horizon_minutes != CONFIG.candle_minutes * horizon_bars:
        raise ValueError("Model horizon does not match the configured candle interval.")
    return FrozenEvaluationManifest(
        asset=CONFIG.instrument,
        symbol=CONFIG.symbol,
        option="B_live_web_app",
        candle_minutes=CONFIG.candle_minutes,
        horizon_bars=horizon_bars,
        horizon_minutes=horizon_minutes,
        evaluation_trading_days=CONFIG.evaluation_days,
        model_version=model_version,
        configuration_version=configuration_version(metadata),
        frozen_at=datetime.now(timezone.utc).isoformat(),
    )


def write_manifest(manifest: FrozenEvaluationManifest, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != asdict(manifest):
            raise FileExistsError("A different frozen evaluation manifest already exists.")
        return destination
    destination.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return destination


def load_manifest(path: str | Path) -> FrozenEvaluationManifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return FrozenEvaluationManifest(**value)


__all__ = ["FrozenEvaluationManifest", "build_manifest", "configuration_version", "load_manifest", "write_manifest"]

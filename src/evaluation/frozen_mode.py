"""Manifest and guardrails for the submitted, no-change live evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.project_config import CONFIG


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
    artifact_paths: dict[str, str | None]
    artifact_hashes: dict[str, str | None]
    configuration: dict[str, Any]
    configuration_hash: str
    feature_schema: list[str]
    target_definition: dict[str, Any]
    data_metadata: dict[str, Any]
    status: str = "frozen"


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


def _sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    metadata: dict[str, Any],
    model_version: str,
    *,
    artifact_paths: dict[str, str | Path | None] | None = None,
    data_metadata: dict[str, Any] | None = None,
) -> FrozenEvaluationManifest:
    horizon_bars = int(metadata.get("prediction_horizon_bars", CONFIG.forecast_horizon_bars))
    horizon_minutes = int(metadata.get("prediction_horizon_minutes", CONFIG.forecast_horizon_minutes))
    if horizon_minutes != CONFIG.candle_minutes * horizon_bars:
        raise ValueError("Model horizon does not match the configured candle interval.")
    paths = {name: (str(path) if path is not None else None) for name, path in (artifact_paths or {}).items()}
    paths.setdefault("classifier", None)
    paths.setdefault("regressor", None)
    paths.setdefault("calibration", None)
    configuration = CONFIG.as_dict()
    configuration["horizon_bars"] = horizon_bars
    configuration["horizon_minutes"] = horizon_minutes
    configuration_encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
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
        artifact_paths=paths,
        artifact_hashes={name: _sha256(path) for name, path in paths.items()},
        configuration=configuration,
        configuration_hash=hashlib.sha256(configuration_encoded).hexdigest(),
        feature_schema=list(metadata.get("feature_names") or []),
        target_definition={
            "classification_target": metadata.get("classification_target"),
            "regression_target": metadata.get("regression_target"),
            "threshold_value_pct": metadata.get("threshold_value_pct"),
        },
        data_metadata=data_metadata or {"status": "not_recorded"},
    )


def write_manifest(manifest: FrozenEvaluationManifest, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        existing_immutable = dict(existing)
        requested_immutable = asdict(manifest)
        existing_immutable.pop("frozen_at", None)
        requested_immutable.pop("frozen_at", None)
        if existing_immutable != requested_immutable:
            raise FileExistsError("A different frozen evaluation manifest already exists.")
        return destination
    destination.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return destination


def load_manifest(path: str | Path) -> FrozenEvaluationManifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return FrozenEvaluationManifest(**value)


def verify_manifest(manifest: FrozenEvaluationManifest) -> dict[str, Any]:
    current_hashes = {name: _sha256(path) for name, path in manifest.artifact_paths.items()}
    artifacts_match = current_hashes == manifest.artifact_hashes
    configuration_encoded = json.dumps(manifest.configuration, sort_keys=True, separators=(",", ":")).encode()
    configuration_match = hashlib.sha256(configuration_encoded).hexdigest() == manifest.configuration_hash
    return {
        "valid": artifacts_match and configuration_match,
        "artifacts_match": artifacts_match,
        "configuration_match": configuration_match,
        "current_artifact_hashes": current_hashes,
    }


__all__ = ["FrozenEvaluationManifest", "build_manifest", "configuration_version", "load_manifest", "verify_manifest", "write_manifest"]

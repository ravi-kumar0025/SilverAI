"""Inference helpers for the final binary XAG/USD spot artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.spot_features import SPOT_FEATURE_COLUMNS
from src.spot_config import SPOT_CONFIG


DEFAULT_SPOT_MODEL_ROOT = Path(__file__).resolve().parents[2] / "reports" / "spot_dukascopy_20261115"


def load_spot_artifacts(root: str | Path = DEFAULT_SPOT_MODEL_ROOT) -> dict[str, Any]:
    root = Path(root)
    model_path = root / "spot_lightgbm.joblib"
    calibrator_path = root / "spot_sigmoid_calibrator.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Spot model artifact not found: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact.get("model") if isinstance(artifact, dict) else artifact
    features = tuple(artifact.get("features", SPOT_FEATURE_COLUMNS)) if isinstance(artifact, dict) else SPOT_FEATURE_COLUMNS
    if features != SPOT_FEATURE_COLUMNS:
        raise ValueError("Spot model feature order does not match the final 34-feature schema.")
    calibrator = joblib.load(calibrator_path) if calibrator_path.is_file() else None
    return {"model": model, "calibrator": calibrator, "features": features, "metadata": artifact if isinstance(artifact, dict) else {}, "dataset_id": SPOT_CONFIG.dataset_id, "model_version": SPOT_CONFIG.model_version}


def validate_spot_features(frame: pd.DataFrame, features: tuple[str, ...] = SPOT_FEATURE_COLUMNS) -> pd.DataFrame:
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required spot features: {', '.join(missing)}")
    numeric = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric)
    if invalid.any().any():
        bad = ", ".join(invalid.any(axis=0)[lambda values: values].index)
        raise ValueError(f"Prediction features contain missing, infinite, or invalid values: {bad}")
    return numeric


def _calibrate(calibrator: Any, probability: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return probability
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def predict_spot(frame: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    features = validate_spot_features(frame, tuple(artifacts["features"]))
    model = artifacts["model"]
    probability_up = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    calibrated_up = _calibrate(artifacts.get("calibrator"), probability_up)
    result = pd.DataFrame(index=frame.index)
    if "timestamp" in frame.columns:
        result["timestamp"] = frame["timestamp"].to_numpy()
    result["probability_down"] = 1 - probability_up
    result["probability_up"] = probability_up
    result["calibrated_probability_down"] = 1 - calibrated_up
    result["calibrated_probability_up"] = calibrated_up
    result["predicted_direction"] = (probability_up >= 0.5).astype(int)
    result["predicted_direction_name"] = np.where(probability_up >= 0.5, "UP", "DOWN")
    result["prediction_horizon_minutes"] = SPOT_CONFIG.forecast_horizon_minutes
    result["model_name"] = "spot_lightgbm"
    result["confidence_threshold"] = SPOT_CONFIG.confidence_threshold
    return result.reset_index(drop=True)


__all__ = ["DEFAULT_SPOT_MODEL_ROOT", "load_spot_artifacts", "predict_spot", "validate_spot_features"]

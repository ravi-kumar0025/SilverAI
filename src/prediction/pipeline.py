"""End-to-end prediction orchestration for SilverAI."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.predict import (
    DEFAULT_MODEL_ROOT,
    DEFAULT_SELECTION_PATH,
    load_model_artifacts,
    predict,
    validate_prediction_features,
)
from src.prediction.scenarios import generate_scenarios
from src.prediction.signals import interpret_signal
from src.config import CONFIG
from src.evaluation.frozen_mode import configuration_version


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _model_metadata(artifacts: dict[str, Any]) -> dict[str, Any]:
    metadata = artifacts.get("metadata", {})
    return {
        "classifier_name": artifacts.get("classifier_name"),
        "regressor_name": artifacts.get("regressor_name"),
        "feature_names": metadata.get("feature_names"),
        "prediction_horizon_minutes": metadata.get("prediction_horizon_minutes", CONFIG.forecast_horizon_minutes),
        "prediction_horizon_bars": metadata.get("prediction_horizon_bars", CONFIG.forecast_horizon_bars),
        "classification_target": metadata.get("classification_target"),
        "regression_target": metadata.get("regression_target"),
        "threshold_value_pct": metadata.get("threshold_value_pct"),
        "configuration_version": configuration_version(metadata),
        "probability_calibration_status": "available" if artifacts.get("calibrator") is not None else "unavailable",
    }


def _probabilities(row: Mapping[str, Any]) -> dict[str, float] | None:
    names = ("down", "neutral", "up")
    values = {}
    for name in names:
        value = row.get(f"probability_{name}")
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        values[name] = numeric
    return values if np.isclose(sum(values.values()), 1.0, atol=1e-6) else None


def _calibrated_probabilities(row: Mapping[str, Any]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for name in ("down", "neutral", "up"):
        value = row.get(f"calibrated_probability_{name}")
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or not 0 <= numeric <= 1:
            return None
        values[name] = numeric
    return values if np.isclose(sum(values.values()), 1.0, atol=1e-6) else None


def _reliability(
    row: Mapping[str, Any], scenarios: dict[str, Any], confidence: Mapping[str, Any] | None
) -> dict[str, Any]:
    probabilities = _probabilities(row)
    limitations: list[str] = []
    if probabilities is None:
        limitations.append("Classifier probabilities are unavailable or unreliable.")
    if scenarios.get("model_disagreement"):
        limitations.append("Classifier and regression outputs disagree.")
    if confidence is None:
        limitations.append("No calibrated confidence module or assessment was supplied.")
    else:
        supplied = confidence.get("limitations", [])
        if isinstance(supplied, str):
            supplied = [supplied]
        if isinstance(supplied, (list, tuple)):
            limitations.extend(str(item) for item in supplied)
        if confidence.get("status") is not None:
            limitations.append(f"Supplied confidence status: {confidence['status']}.")
    limitations = list(dict.fromkeys(limitations))
    if scenarios.get("model_disagreement"):
        status = "mixed"
    elif probabilities is None or confidence is None:
        status = "limited"
    else:
        status = "available"
    return {
        "status": status,
        "probabilities_available": probabilities is not None,
        "models_agree": not bool(scenarios.get("model_disagreement")),
        "limitations": limitations,
    }


def _row_response(
    row: pd.Series,
    artifacts: dict[str, Any],
    *,
    threshold_pct: float | None,
    confidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row_mapping = row.to_dict()
    scenarios = generate_scenarios(
        row_mapping,
        threshold_pct=threshold_pct,
        confidence=confidence,
    )
    reliability = _reliability(row_mapping, scenarios, confidence)
    signal = interpret_signal(row_mapping, scenarios, confidence)
    warnings = list(reliability["limitations"])
    if signal["signal"] == "INSUFFICIENT_DATA":
        warnings.append("Final analytical signal is unavailable because required evidence is incomplete.")
    return {
        "prediction_timestamp": scenarios.get("timestamp"),
        "forecast_horizon_minutes": row_mapping.get("prediction_horizon_minutes", CONFIG.forecast_horizon_minutes),
        "predicted_direction": row_mapping.get("predicted_direction"),
        "predicted_direction_name": row_mapping.get("predicted_direction_name"),
        "predicted_return_15m": row_mapping.get("predicted_return_15m"),
        "class_probabilities": _probabilities(row_mapping),
        "calibrated_probabilities": _calibrated_probabilities(row_mapping),
        "reliability": reliability,
        "scenarios": scenarios["scenarios"],
        "signal": signal,
        "model_metadata": _model_metadata(artifacts),
        "warnings": list(dict.fromkeys(warnings)),
    }


def generate_prediction(
    features: pd.DataFrame,
    *,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    selection_path: str | Path | None = DEFAULT_SELECTION_PATH,
    artifacts: dict[str, Any] | None = None,
    confidence: Mapping[str, Any] | None = None,
    threshold_pct: float | None = None,
) -> dict[str, Any]:
    """Run prediction, scenario, reliability, and signal generation.

    A one-row input returns a response with direct prediction fields and a
    one-item ``predictions`` list. A multi-row input returns the same list in a
    batch envelope. No stage retrains, recalibrates, or modifies a model.
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.empty:
        raise ValueError("features must contain at least one row.")
    validate_prediction_features(features)
    artifacts = artifacts or load_model_artifacts(model_root, selection_path)
    metadata = artifacts["metadata"]
    if threshold_pct is None:
        threshold_pct = metadata.get("threshold_value_pct")
    if threshold_pct is not None:
        threshold_pct = float(threshold_pct)
    prediction_frame = predict(features, artifacts)
    responses = [
        _row_response(
            row,
            artifacts,
            threshold_pct=threshold_pct,
            confidence=confidence,
        )
        for _, row in prediction_frame.iterrows()
    ]
    safe_responses = [_json_safe(response) for response in responses]
    safe_metadata = _json_safe(_model_metadata(artifacts))
    if len(safe_responses) == 1:
        result = dict(safe_responses[0])
        result["batch"] = False
        result["predictions"] = safe_responses
        return result
    return {
        "batch": True,
        "predictions": safe_responses,
        "model_metadata": safe_metadata,
        "warnings": list(
            dict.fromkeys(
                warning
                for response in safe_responses
                for warning in response["warnings"]
            )
        ),
    }


__all__ = ["generate_prediction"]

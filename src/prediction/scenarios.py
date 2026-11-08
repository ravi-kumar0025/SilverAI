"""Deterministic, descriptive scenario summaries from SilverAI predictions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


_DIRECTIONS = {-1: "Bearish", 0: "Neutral", 1: "Bullish"}
_PROBABILITY_ALIASES = {
    -1: ("probability_down", "probability_bearish"),
    0: ("probability_neutral",),
    1: ("probability_up", "probability_bullish"),
}


def _single_prediction(prediction: Mapping[str, Any] | pd.Series | pd.DataFrame) -> dict[str, Any]:
    if isinstance(prediction, pd.DataFrame):
        if len(prediction) != 1:
            raise ValueError("Scenario generation requires exactly one prediction row.")
        return prediction.iloc[0].to_dict()
    if isinstance(prediction, pd.Series):
        return prediction.to_dict()
    if isinstance(prediction, Mapping):
        return dict(prediction)
    raise TypeError("prediction must be a mapping, pandas Series, or one-row DataFrame.")


def _direction(value: Any, field: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be one of -1, 0, or 1.") from exc
    if numeric not in _DIRECTIONS:
        raise ValueError(f"{field} must be one of -1, 0, or 1.")
    return numeric


def _finite_return(row: dict[str, Any]) -> float:
    value = row.get("predicted_return_15m")
    if value is None:
        value = row.get("predicted_15m_return")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction is missing a numeric predicted 15-minute return.") from exc
    if not np.isfinite(value):
        raise ValueError("Predicted 15-minute return must be finite.")
    return value


def _probabilities(row: dict[str, Any]) -> dict[int, float] | None:
    values: dict[int, float] = {}
    for label, aliases in _PROBABILITY_ALIASES.items():
        value = next((row[name] for name in aliases if name in row), None)
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or not 0 <= numeric <= 1:
            return None
        values[label] = numeric
    if not np.isclose(sum(values.values()), 1.0, atol=1e-6):
        return None
    return values


def _regression_direction(predicted_return: float, threshold_pct: float | None) -> int:
    if threshold_pct is None:
        return 1 if predicted_return > 0 else -1 if predicted_return < 0 else 0
    if threshold_pct < 0:
        raise ValueError("threshold_pct must be nonnegative.")
    return 1 if predicted_return > threshold_pct else -1 if predicted_return < -threshold_pct else 0


def _confidence_limitations(confidence: Mapping[str, Any] | None) -> list[str]:
    if confidence is None:
        return ["No calibrated confidence or reliability assessment was supplied."]
    limitations = confidence.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations]
    if not isinstance(limitations, (list, tuple)):
        limitations = []
    result = [str(item) for item in limitations]
    status = confidence.get("status")
    if status is not None and str(status).lower() not in {"calibrated", "reliable"}:
        result.append(f"Supplied confidence status: {status}.")
    return result or ["Reliability information was supplied but no limitations were specified."]


def _evidence(
    scenario_direction: int,
    classifier_direction: int,
    regression_direction: int,
    predicted_return: float,
    probabilities: dict[int, float] | None,
    threshold_pct: float | None,
) -> tuple[list[str], list[str], list[str]]:
    supporting: list[str] = []
    contradicting: list[str] = []
    limitations: list[str] = []
    if classifier_direction == scenario_direction:
        supporting.append(f"Classifier predicts {_DIRECTIONS[scenario_direction].upper()}.")
    else:
        contradicting.append(f"Classifier predicts {_DIRECTIONS[classifier_direction].upper()}.")
    if probabilities is None:
        limitations.append("Classifier probabilities are unavailable or unreliable.")
    else:
        supporting.append(
            f"Classifier {_DIRECTIONS[scenario_direction].lower()} probability: "
            f"{probabilities[scenario_direction]:.6f}."
        )
    if regression_direction == scenario_direction:
        supporting.append(f"Regression forecasts {predicted_return:.6f}% over 15 minutes.")
    else:
        contradicting.append(f"Regression forecasts {predicted_return:.6f}% over 15 minutes.")
    if classifier_direction != regression_direction:
        limitations.append(
            "Classifier and regression forecasts disagree; neither output was forced to agree."
        )
    if threshold_pct is None:
        limitations.append(
            "No regression neutral-band threshold was supplied; regression direction uses sign only."
        )
    if not supporting:
        supporting.append("No direct model evidence supports this scenario.")
    return supporting, contradicting, limitations


def generate_scenarios(
    prediction: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    threshold_pct: float | None = None,
    confidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate Bullish, Bearish, and Neutral summaries for one prediction.

    ``prediction`` should be one row returned by :func:`src.models.predict.predict`.
    Probabilities and confidence information are reported only when supplied;
    this function does not calculate confidence, signals, targets, or market data.
    """

    row = _single_prediction(prediction)
    classifier_direction = _direction(row.get("predicted_direction"), "predicted_direction")
    predicted_return = _finite_return(row)
    probabilities = _probabilities(row)
    regression_direction = _regression_direction(predicted_return, threshold_pct)
    common_limitations = _confidence_limitations(confidence)
    scenarios: dict[str, dict[str, Any]] = {}
    for scenario_direction, scenario_name in ((1, "Bullish"), (-1, "Bearish"), (0, "Neutral")):
        supporting, contradicting, limitations = _evidence(
            scenario_direction,
            classifier_direction,
            regression_direction,
            predicted_return,
            probabilities,
            threshold_pct,
        )
        limitations = common_limitations + limitations
        forecast = {
            "predicted_direction": classifier_direction,
            "predicted_direction_name": _DIRECTIONS[classifier_direction],
            "predicted_return_15m": predicted_return,
            "regression_implied_direction": regression_direction,
            "regression_implied_direction_name": _DIRECTIONS[regression_direction],
            "classifier_name": row.get("classifier_name"),
            "regressor_name": row.get("regressor_name"),
            "prediction_horizon_minutes": row.get("prediction_horizon_minutes", 15),
        }
        scenarios[scenario_name] = {
            "scenario_name": scenario_name,
            "probability": None if probabilities is None else probabilities[scenario_direction],
            "expected_direction": scenario_direction,
            "expected_direction_name": scenario_name,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "forecast": forecast,
            "reliability_limitations": limitations,
        }
    return {
        "timestamp": row.get("datetime_et", row.get("timestamp")),
        "scenarios": scenarios,
        "model_disagreement": classifier_direction != regression_direction,
        "classifier_direction": classifier_direction,
        "regression_implied_direction": regression_direction,
    }


__all__ = ["generate_scenarios"]

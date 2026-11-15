"""Analytical signal interpretation for SilverAI prediction scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


_SCENARIO_NAMES = ("Bullish", "Bearish", "Neutral")
_DIRECTION_NAMES = {-1: "BEARISH", 0: "NEUTRAL", 1: "BULLISH"}
_LOW_RELIABILITY = {
    "low",
    "unreliable",
    "insufficient",
    "insufficient_data",
    "not_calibrated",
    "not calibrated",
}


def _row(prediction: Mapping[str, Any] | pd.Series | pd.DataFrame) -> dict[str, Any]:
    if isinstance(prediction, pd.DataFrame):
        if len(prediction) != 1:
            raise ValueError("Signal interpretation requires exactly one prediction row.")
        return prediction.iloc[0].to_dict()
    if isinstance(prediction, pd.Series):
        return prediction.to_dict()
    if isinstance(prediction, Mapping):
        return dict(prediction)
    raise TypeError("prediction must be a mapping, pandas Series, or one-row DataFrame.")


def _valid_direction(value: Any) -> bool:
    try:
        return int(value) in _DIRECTION_NAMES
    except (TypeError, ValueError):
        return False


def _valid_return(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _reliability_limitations(confidence: Mapping[str, Any] | None) -> tuple[bool, list[str]]:
    if confidence is None:
        return False, ["No calibrated historical reliability assessment was supplied."]
    limitations = confidence.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations]
    if not isinstance(limitations, (list, tuple)):
        limitations = []
    result = [str(item) for item in limitations]
    reliable = confidence.get("reliable")
    status = str(confidence.get("status", "")).strip().lower()
    reliability = str(confidence.get("reliability", "")).strip().lower()
    insufficient = reliable is False or status in _LOW_RELIABILITY or reliability in _LOW_RELIABILITY
    if insufficient:
        result.append("Supplied historical reliability information is insufficient for a directional signal.")
    elif status:
        result.append(f"Supplied reliability status: {confidence['status']}.")
    return insufficient, result


def _valid_scenarios(scenarios: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(scenarios, Mapping) or not isinstance(scenarios.get("scenarios"), Mapping):
        return False, ["Scenario output is missing its scenarios mapping."]
    entries = scenarios["scenarios"]
    missing = [name for name in _SCENARIO_NAMES if name not in entries]
    if missing:
        return False, [f"Scenario output is missing: {', '.join(missing)}."]
    limitations: list[str] = []
    for name in _SCENARIO_NAMES:
        entry = entries[name]
        if not isinstance(entry, Mapping):
            return False, [f"Scenario '{name}' is not a structured mapping."]
        for field in ("supporting_evidence", "contradicting_evidence", "reliability_limitations"):
            if field not in entry or not isinstance(entry[field], (list, tuple)):
                return False, [f"Scenario '{name}' has incomplete {field}."]
        limitations.extend(str(item) for item in entry["reliability_limitations"])
    return True, list(dict.fromkeys(limitations))


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def interpret_signal(
    prediction: Mapping[str, Any] | pd.Series | pd.DataFrame,
    scenarios: Mapping[str, Any],
    confidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one prediction and its scenarios into an analytical label.

    Agreement between classifier and regression directions produces a
    directional label; disagreement produces ``MIXED``. Missing or explicitly
    unreliable inputs produce ``INSUFFICIENT_DATA``. No confidence threshold is
    invented here.
    """

    row = _row(prediction)
    valid_scenarios, scenario_limitations = _valid_scenarios(scenarios)
    insufficient_reliability, reliability_limitations = _reliability_limitations(confidence)
    timestamp = scenarios.get("timestamp", row.get("datetime_et", row.get("timestamp")))
    base = {
        "signal": "INSUFFICIENT_DATA",
        "actionable_signal": "NO_CLEAR_SIGNAL",
        "directional_interpretation": "No reliable directional interpretation is available.",
        "supporting_evidence": [],
        "conflicting_evidence": [],
        "reliability_limitations": _unique(scenario_limitations + reliability_limitations),
        "timestamp": timestamp,
        "explanation": "Required prediction, scenario, or reliability inputs are incomplete.",
        "trade_decision": "WAIT",
        "decision_reason": "A directional forecast is not sufficient evidence for a trade decision.",
    }
    if not valid_scenarios:
        return base
    if insufficient_reliability:
        base["explanation"] = "Historical reliability information explicitly marks the input as insufficient."
        return base
    if not _valid_direction(row.get("predicted_direction")):
        base["reliability_limitations"].append("Predicted classifier direction is missing or invalid.")
        return base
    predicted_return = row.get("predicted_return_15m", row.get("predicted_15m_return"))
    if not _valid_return(predicted_return):
        base["reliability_limitations"].append("Predicted 15-minute return is missing or invalid.")
        return base
    classifier_direction = int(row["predicted_direction"])
    regression_direction = scenarios.get("regression_implied_direction")
    if not _valid_direction(regression_direction):
        base["reliability_limitations"].append("Regression-implied direction is missing or invalid.")
        return base
    regression_direction = int(regression_direction)

    if classifier_direction != regression_direction:
        signal = "MIXED"
        interpretation = "Classifier and regression outputs disagree; directional evidence is mixed."
        explanation = (
            f"The classifier indicates {_DIRECTION_NAMES[classifier_direction]}, while the regression "
            f"forecast implies {_DIRECTION_NAMES[regression_direction]}; the models disagree. "
            "No directional signal was forced."
        )
        evidence_entries = [scenarios["scenarios"][_SCENARIO_NAMES[0]], scenarios["scenarios"][_SCENARIO_NAMES[1]]]
        supporting = [item for entry in evidence_entries for item in entry["supporting_evidence"]]
        conflicting = [item for entry in evidence_entries for item in entry["contradicting_evidence"]]
    elif classifier_direction == 0:
        signal = "NEUTRAL"
        interpretation = "Both model outputs support limited or indecisive movement."
        explanation = "The classifier predicts NEUTRAL and the regression forecast is not directionally meaningful under the supplied scenario threshold."
        entry = scenarios["scenarios"]["Neutral"]
        supporting = entry["supporting_evidence"]
        conflicting = entry["contradicting_evidence"]
    else:
        signal = _DIRECTION_NAMES[classifier_direction]
        interpretation = f"Both model outputs support {_DIRECTION_NAMES[classifier_direction].lower()} movement."
        explanation = f"Classifier and regression outputs agree on {_DIRECTION_NAMES[classifier_direction].lower()} direction."
        entry = scenarios["scenarios"]["Bullish" if classifier_direction == 1 else "Bearish"]
        supporting = entry["supporting_evidence"]
        conflicting = entry["contradicting_evidence"]

    if scenarios.get("model_disagreement") is True and signal != "MIXED":
        signal = "MIXED"
        explanation += " Scenario metadata also reports model disagreement."
    base.update(
        {
            "signal": signal,
            "directional_interpretation": interpretation,
            "supporting_evidence": _unique(list(supporting)),
            "conflicting_evidence": _unique(list(conflicting)),
            "explanation": explanation,
            "actionable_signal": "UP" if signal == "BULLISH" else "DOWN" if signal == "BEARISH" else "NO_CLEAR_SIGNAL",
        }
    )
    probabilities = []
    for name in ("probability_down", "probability_neutral", "probability_up"):
        try:
            probabilities.append(float(row.get(name)))
        except (TypeError, ValueError):
            probabilities = []
            break
    margin = None
    if len(probabilities) == 3 and all(np.isfinite(probabilities)):
        ordered = sorted(probabilities, reverse=True)
        margin = ordered[0] - ordered[1]
    threshold = confidence.get("threshold_pct") if isinstance(confidence, Mapping) else None
    try:
        threshold = abs(float(threshold))
    except (TypeError, ValueError):
        threshold = 0.0776397515528126
    return_margin = margin is not None and margin >= 0.05
    return_size = abs(float(predicted_return)) >= threshold
    if signal in {"BULLISH", "BEARISH"} and not scenarios.get("model_disagreement") and return_margin and return_size:
        base["trade_decision"] = "ACTIONABLE_LONG" if signal == "BULLISH" else "ACTIONABLE_SHORT"
        base["decision_reason"] = "Classifier and regression agree, the probability margin is at least 5 points, and the forecast clears the neutral band."
    else:
        base["trade_decision"] = "WAIT"
        base["decision_reason"] = "Wait: require model agreement, a 5-point probability margin, and a forecast beyond the neutral band."
    base["probability_margin"] = margin
    base["coverage_policy"] = "selective_actionable_signals"
    return base


__all__ = ["interpret_signal"]

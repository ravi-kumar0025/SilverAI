import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.scenarios import generate_scenarios
from src.prediction.signals import interpret_signal


def _prediction(direction, predicted_return):
    return {
        "datetime_et": "2026-01-01 10:00:00-05:00",
        "predicted_direction": direction,
        "predicted_return_15m": predicted_return,
        "probability_down": 0.1,
        "probability_neutral": 0.2,
        "probability_up": 0.7,
    }


def _scenarios(direction, predicted_return):
    return generate_scenarios(
        _prediction(direction, predicted_return), threshold_pct=0.1
    )


def test_directional_and_neutral_signals():
    assert interpret_signal(_prediction(1, 0.25), _scenarios(1, 0.25))["signal"] == "BULLISH"
    assert interpret_signal(_prediction(-1, -0.25), _scenarios(-1, -0.25))["signal"] == "BEARISH"
    assert interpret_signal(_prediction(0, 0.0), _scenarios(0, 0.0))["signal"] == "NEUTRAL"


def test_disagreement_is_mixed():
    result = interpret_signal(_prediction(1, -0.25), _scenarios(1, -0.25))

    assert result["signal"] == "MIXED"
    assert "disagree" in result["explanation"]


def test_missing_and_low_reliability_are_insufficient():
    incomplete = _scenarios(1, 0.25)
    incomplete["scenarios"].pop("Neutral")
    assert interpret_signal(_prediction(1, 0.25), incomplete)["signal"] == "INSUFFICIENT_DATA"

    low = interpret_signal(
        _prediction(1, 0.25), _scenarios(1, 0.25), {"status": "low"}
    )
    assert low["signal"] == "INSUFFICIENT_DATA"


def test_actionable_decision_is_selective():
    result = interpret_signal(
        _prediction(1, 0.25),
        _scenarios(1, 0.25),
    )
    assert result["trade_decision"] == "ACTIONABLE_LONG"
    assert result["coverage_policy"] == "selective_actionable_signals"


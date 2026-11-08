import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.scenarios import generate_scenarios


def _prediction(direction=1, predicted_return=0.25, probabilities=True):
    row = {
        "datetime_et": "2026-01-01 10:00:00-05:00",
        "predicted_direction": direction,
        "predicted_return_15m": predicted_return,
        "classifier_name": "classifier",
        "regressor_name": "regressor",
        "prediction_horizon_minutes": 15,
    }
    if probabilities:
        row.update(
            {
                "probability_down": 0.1,
                "probability_neutral": 0.2,
                "probability_up": 0.7,
            }
        )
    return row


def test_all_three_scenarios_are_returned_with_probabilities():
    result = generate_scenarios(_prediction(), threshold_pct=0.1)

    assert list(result["scenarios"]) == ["Bullish", "Bearish", "Neutral"]
    assert result["scenarios"]["Bullish"]["probability"] == 0.7
    assert result["scenarios"]["Bullish"]["expected_direction"] == 1
    assert result["timestamp"] == "2026-01-01 10:00:00-05:00"


def test_model_disagreement_is_explicit():
    result = generate_scenarios(_prediction(direction=1, predicted_return=-0.25), threshold_pct=0.1)

    assert result["model_disagreement"] is True
    limitations = result["scenarios"]["Bullish"]["reliability_limitations"]
    assert any("disagree" in item for item in limitations)
    assert any("Regression forecasts -0.250000%" in item for item in result["scenarios"]["Bullish"]["contradicting_evidence"])


def test_missing_probabilities_and_neutral_forecast_are_reported():
    result = generate_scenarios(
        _prediction(direction=0, predicted_return=0.0, probabilities=False),
        threshold_pct=0.1,
    )

    assert result["model_disagreement"] is False
    assert result["scenarios"]["Neutral"]["probability"] is None
    assert any("unavailable" in item for item in result["scenarios"]["Neutral"]["reliability_limitations"])


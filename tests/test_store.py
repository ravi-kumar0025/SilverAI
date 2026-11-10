from datetime import datetime, timedelta

import pandas as pd

from src.prediction.store import PredictionStore


def test_prediction_store_deduplicates_and_preserves_signal(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    payload = {
        "instrument": "COMEX Silver Futures",
        "contract": "SI",
        "candle_timestamp": "2026-01-01T09:30:00-05:00",
        "close_price": 50.0,
        "predicted_direction_name": "UP",
        "class_probabilities": {"down": 0.1, "neutral": 0.2, "up": 0.7},
        "model_metadata": {"classifier_name": "frozen-v1"},
        "forecast_horizon_minutes": 15,
        "forecast_horizon_bars": 3,
    }
    first = store.record(payload)
    second = store.record(payload)
    assert first["prediction_id"] == second["prediction_id"]
    assert len(store.list()) == 1


def test_prediction_outcome_is_written_without_changing_forecast(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    timestamp = pd.Timestamp("2026-01-01 09:30", tz="America/New_York")
    store.record(
        {
            "instrument": "COMEX Silver Futures",
            "contract": "SI",
            "candle_timestamp": timestamp.isoformat(),
            "close_price": 50.0,
            "predicted_direction_name": "UP",
            "class_probabilities": {"down": 0.1, "neutral": 0.2, "up": 0.7},
            "model_metadata": {"classifier_name": "frozen-v1"},
            "forecast_horizon_minutes": 15,
            "forecast_horizon_bars": 3,
        }
    )
    candles = pd.DataFrame(
        {
            "datetime_et": [timestamp + timedelta(minutes=5 * i) for i in range(4)],
            "close": [50.0, 50.1, 50.2, 50.5],
        }
    )
    assert store.evaluate(candles) == 1
    saved = store.list()[0]
    assert saved["predicted_direction"] == "UP"
    assert saved["actual_direction"] == "UP"
    assert saved["outcome_status"] == "evaluated"

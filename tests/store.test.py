from datetime import datetime, timedelta

import pandas as pd

from backend.services.prediction_pipeline_service import PredictionService
from src.prediction.store import PredictionStore, is_finalized_signal_record


def test_prediction_store_deduplicates_and_preserves_signal(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    payload = {
        "instrument": "XAGUSD_SPOT",
        "contract": "XAG/USD",
        "candle_timestamp": "2026-01-01T09:30:00Z",
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


def test_prediction_store_records_only_explicitly_closed_candles_as_finalized(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    row = store.record({
        "candle_timestamp": "2026-01-01T09:30:00Z",
        "close_price": 50.0,
        "predicted_direction_name": "UP",
        "class_probabilities": {"down": 0.1, "neutral": 0.2, "up": 0.7},
        "signal": {"signal": "BULLISH"},
        "candle_finalized": True,
        "model_metadata": {"classifier_name": "frozen-v1"},
        "forecast_horizon_minutes": 15,
        "forecast_horizon_bars": 3,
    })
    assert row["final_signal"] == "BULLISH"
    assert row["candle_finalized"] == 1


def test_finalized_signal_eligibility_excludes_newer_unfinalized_and_raw_rows(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    base = {
        "close_price": 50.0,
        "predicted_direction_name": "UP",
        "class_probabilities": {"down": 0.1, "neutral": 0.2, "up": 0.7},
        "model_metadata": {"classifier_name": "frozen-v1"},
        "forecast_horizon_minutes": 15,
        "forecast_horizon_bars": 3,
    }
    store.record({**base, "candle_timestamp": "2026-01-01T09:30:00Z", "signal": {"signal": "BULLISH"}, "candle_finalized": True})
    store.record({**base, "candle_timestamp": "2026-01-01T09:35:00Z", "predicted_direction_name": "DOWN", "candle_finalized": False})
    store.record({**base, "candle_timestamp": "2026-01-01T09:40:00Z", "candle_finalized": True})
    latest = store.list_finalized_signals(limit=1)
    assert latest[0]["candle_timestamp"].startswith("2026-01-01T09:30")
    assert is_finalized_signal_record(latest[0])


def test_latest_retrieval_uses_finalized_store_record_after_restart(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    base = {
        "close_price": 50.0,
        "predicted_direction_name": "UP",
        "class_probabilities": {"down": 0.1, "neutral": 0.2, "up": 0.7},
        "model_metadata": {"classifier_name": "frozen-v1"},
        "forecast_horizon_minutes": 15,
        "forecast_horizon_bars": 3,
    }
    store.record({**base, "candle_timestamp": "2026-01-01T09:30:00Z", "signal": {"signal": "BULLISH"}, "candle_finalized": True})
    store.record({**base, "candle_timestamp": "2026-01-01T09:35:00Z", "predicted_direction_name": "DOWN", "candle_finalized": False})
    service = PredictionService.__new__(PredictionService)
    service.store = store
    service._latest = None
    assert service.latest()["final_signal"] == "BULLISH"


def test_prediction_outcome_is_written_without_changing_forecast(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    timestamp = pd.Timestamp("2026-01-01 09:30", tz="UTC")
    store.record(
        {
            "instrument": "XAGUSD_SPOT",
            "contract": "XAG/USD",
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


def test_prediction_remains_pending_when_horizon_contains_a_gap(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    timestamp = pd.Timestamp("2026-01-01 09:30", tz="UTC")
    store.record({
        "candle_timestamp": timestamp.isoformat(),
        "close_price": 50.0,
        "predicted_direction_name": "UP",
        "class_probabilities": {"down": 0.1, "neutral": 0.2, "up": 0.7},
        "model_metadata": {"classifier_name": "frozen-v1"},
        "forecast_horizon_minutes": 15,
        "forecast_horizon_bars": 3,
    })
    candles = pd.DataFrame({
        "datetime_et": [timestamp, timestamp + timedelta(minutes=5), timestamp + timedelta(minutes=15)],
        "close": [50.0, 50.1, 50.5],
    })
    assert store.evaluate(candles) == 0
    assert store.list()[0]["outcome_status"] == "pending"

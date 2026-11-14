from datetime import timedelta

import pandas as pd
import pytest

from src.evaluation.paper_trading import evaluate_paper_trades
from src.prediction.store import PredictionStore
from src.targets import generate_timestamp_targets


def test_timestamp_target_rejects_gap_instead_of_using_next_row():
    start = pd.Timestamp("2026-01-01 09:30", tz="America/New_York")
    frame = pd.DataFrame({
        "datetime_et": [start, start + timedelta(minutes=5), start + timedelta(minutes=15)],
        "close": [50.0, 50.1, 50.5],
    })
    result = generate_timestamp_targets(frame, interval_minutes=5, horizon_minutes=15)
    assert result.loc[0, "target_valid"] is False or not bool(result.loc[0, "target_valid"])
    assert pd.isna(result.loc[0, "future_return_pct"])


def test_prediction_forecast_fields_are_database_immutable(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    saved = store.record({
        "candle_timestamp": "2026-01-01T09:30:00-05:00",
        "close_price": 50.0,
        "predicted_direction_name": "UP",
        "class_probabilities": {"down": 0.1, "neutral": 0.2, "up": 0.7},
        "model_metadata": {"classifier_name": "frozen-v1"},
    })
    with store._connect() as db:
        with pytest.raises(Exception, match="immutable"):
            db.execute("UPDATE predictions SET predicted_direction='DOWN' WHERE prediction_id=?", (saved["prediction_id"],))


def test_paper_trading_uses_non_actionable_final_signal_as_no_trade():
    result = evaluate_paper_trades([
        {"outcome_status": "evaluated", "final_signal": "MIXED", "predicted_direction": "UP", "close_price": 50.0, "actual_price": 51.0, "candle_timestamp": "2026-01-01T09:30:00Z"},
    ])
    assert result["trade_count"] == 0

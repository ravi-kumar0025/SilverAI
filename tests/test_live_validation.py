import pandas as pd

from src.data.market_data import HistoricalCsvProvider, validate_candles
from src.evaluation.baselines import classification_baselines
from src.evaluation.paper_trading import evaluate_paper_trades


def test_historical_provider_is_explicitly_not_live():
    provider = HistoricalCsvProvider()
    candles = provider.candles(12)
    assert provider.is_live is False
    assert provider.name == "silver_5m.csv"
    assert len(candles) == 12
    assert candles[-1].high >= max(candles[-1].open, candles[-1].close)


def test_validation_rejects_invalid_ohlc():
    invalid = pd.DataFrame({
        "datetime_et": pd.to_datetime(["2026-01-01T09:30:00-05:00"]),
        "open": [50.0], "high": [49.0], "low": [48.0], "close": [50.0], "volume": [1.0],
    })
    try:
        validate_candles(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid OHLC data must be rejected")


def test_baselines_and_paper_trading_are_explicit():
    baselines = classification_baselines(pd.Series([-1, 0, 1, 1]), pd.Series([50, 49, 50, 51]))
    assert {"always_up_baseline", "always_down_baseline", "majority_class_baseline", "rsi_14_baseline"} <= set(baselines.model)
    result = evaluate_paper_trades([
        {"prediction_id": "p1", "outcome_status": "evaluated", "predicted_direction": "UP", "close_price": 50, "actual_price": 51, "candle_timestamp": "2026-01-01T09:30:00", "outcome_evaluated_at": "2026-01-01T09:45:00"}
    ])
    assert result["paper_only"] is True
    assert result["trade_count"] == 1
    assert result["net_pnl"] < result["gross_pnl"]

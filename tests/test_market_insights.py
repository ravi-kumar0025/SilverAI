import pandas as pd

from src.market_insights import summarize_market


def test_market_insights_uses_closed_candles_and_exposes_levels():
    timestamps = pd.date_range("2026-01-01 09:00", periods=60, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "datetime_et": timestamps,
            "open": 50.0,
            "high": 50.2,
            "low": 49.8,
            "close": [50.0 + index * 0.01 for index in range(60)],
        }
    )
    result = summarize_market(frame)
    assert result["timestamp"] == str(timestamps[-1])
    assert result["levels"]["session_high"] >= result["levels"]["session_low"]
    assert result["trend"]["state"] == "UPTREND"

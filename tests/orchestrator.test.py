from datetime import timedelta

import pandas as pd

from src.data.market_data_contracts import MarketUpdate
from src.prediction.orchestrator import ClosedCandleOrchestrator


def test_orchestrator_predicts_only_after_a_candle_closes():
    calls = []

    def features(candles):
        return pd.DataFrame({"close": [candles[0].close]})

    def predict(frame, candle):
        calls.append(candle.timestamp)
        return {"candle_timestamp": candle.timestamp}

    orchestrator = ClosedCandleOrchestrator(features, predict)
    start = pd.Timestamp("2026-01-01 09:30", tz="UTC")
    assert orchestrator.on_update(MarketUpdate(start, 50.0)) is None
    assert orchestrator.on_update(MarketUpdate(start + timedelta(minutes=4), 50.1)) is None
    result = orchestrator.on_update(MarketUpdate(start + timedelta(minutes=5), 50.2))
    assert result["candle_timestamp"].startswith("2026-01-01T09:30:00+00:00")
    assert len(calls) == 1


def test_orchestrator_deduplicates_after_restore():
    def features(candles):
        return pd.DataFrame({"close": [candles[0].close]})

    calls = []
    orchestrator = ClosedCandleOrchestrator(features, lambda frame, candle: calls.append(candle.timestamp) or {})
    start = pd.Timestamp("2026-01-01 09:30", tz="UTC")
    orchestrator.restore([{"instrument": "XAGUSD_SPOT", "contract": "XAG/USD", "candle_timestamp": start.isoformat()}])
    orchestrator.on_update(MarketUpdate(start, 50.0))
    orchestrator.on_update(MarketUpdate(start + timedelta(minutes=5), 50.2))
    assert calls == []

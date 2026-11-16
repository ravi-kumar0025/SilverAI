from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.rolling import RollingFeatureBuilder
from src.features.market_features import SPOT_FEATURE_COLUMNS, generate_spot_features


def _candles(count=205):
    start = pd.Timestamp("2026-01-05 09:30", tz="UTC")
    rows = []
    for index in range(count):
        close = 50.0 + index * 0.01 + float(np.sin(index / 4.0) * 0.08)
        rows.append({
            "timestamp": (start + timedelta(minutes=5 * index)).isoformat(),
            "silver_open": close - 0.01,
            "silver_high": close + 0.02,
            "silver_low": close - 0.02,
            "silver_close": close,
            "xauusd_close": 2000.0 + index * 0.02,
            "eurusd_close": 1.1 + index * 0.0001,
        })
    return rows


def test_rolling_features_match_offline_generation_after_warmup():
    candles = _candles()
    builder = RollingFeatureBuilder()
    online = None
    for candle in candles:
        online = builder.append(candle)
    expected = generate_spot_features(pd.DataFrame(candles)).iloc[[-1]][list(SPOT_FEATURE_COLUMNS)]
    assert online is not None
    pd.testing.assert_frame_equal(online.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False, atol=1e-12)


def test_rolling_builder_rejects_duplicates_and_resets_on_gap():
    candles = _candles()
    builder = RollingFeatureBuilder()
    for candle in candles[:202]:
        builder.append(candle)
    with pytest.raises(ValueError, match="Duplicate"):
        builder.append(candles[201])
    gap = dict(candles[-1])
    gap["timestamp"] = (pd.Timestamp(candles[-1]["timestamp"]) + timedelta(minutes=10)).isoformat()
    assert builder.append(gap) is None
    assert builder.history_length == 1


def test_rolling_builder_restores_context_and_rejects_insufficient_history():
    candles = _candles()
    first = RollingFeatureBuilder()
    for candle in candles[:202]:
        first.append(candle)
    restored = RollingFeatureBuilder()
    restored.restore_history(first.snapshot())
    assert restored.history_length == first.history_length
    assert restored.append(candles[202]) is not None
    assert RollingFeatureBuilder().append(candles[0]) is None

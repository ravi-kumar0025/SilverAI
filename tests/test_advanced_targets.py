import numpy as np
import pandas as pd

from src.evaluation.selectivity import selective_metrics
from src.spot_targets import make_triple_barrier_labels, sample_uniqueness_weights


def _frame(rows=30):
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    close = np.linspace(50.0, 51.0, rows)
    return pd.DataFrame({
        "timestamp": timestamps, "silver_open": close - .01,
        "silver_high": close + .02, "silver_low": close - .02,
        "silver_close": close,
    })


def test_triple_barrier_and_uniqueness_weights_are_defined():
    labeled, report = make_triple_barrier_labels(_frame(), horizon_bars=3, profit_multiple=.2)
    assert report["rows_kept"] == len(labeled)
    assert {0.0, 1.0}.issuperset(set(labeled.triple_barrier_label.dropna().unique()))
    weights = sample_uniqueness_weights(pd.DataFrame({"label": [0, 1, 0, 1, 0]}), 2)
    assert np.isclose(weights.iloc[0], 0.5)


def test_selective_metrics_include_coverage_and_confidence_interval():
    result = selective_metrics([1, 0, 1, 0], [.8, .2, .55, .45], .52)
    assert result["signals"] == 4
    assert result["accuracy_ci95_lower"] <= result["accuracy"] <= result["accuracy_ci95_upper"]


def test_triple_barrier_uses_real_spot_schema_and_exact_horizon_barrier():
    frame = _frame(18)
    frame[["silver_open", "silver_high", "silver_low", "silver_close"]] = 100.0
    frame["silver_high"] = 100.01
    frame["silver_low"] = 99.99
    frame.loc[15, "silver_high"] = 100.0040001
    frame.loc[15, "silver_low"] = 100.0
    labeled, report = make_triple_barrier_labels(frame, horizon_bars=3, profit_multiple=.2)
    event = labeled[labeled.timestamp == frame.loc[14, "timestamp"]].iloc[0]
    assert event.triple_barrier_label == 1.0
    assert event.triple_barrier_exit_bars == 1.0
    assert report["missing_future_rows"] >= 0


def test_triple_barrier_drops_events_with_missing_future_candles():
    frame = _frame(20).drop(index=16).reset_index(drop=True)
    labeled, report = make_triple_barrier_labels(frame, horizon_bars=3, profit_multiple=.2)
    assert frame.loc[14, "timestamp"] not in set(labeled.timestamp)
    assert report["missing_future_rows"] > 0


def test_triple_barrier_does_not_cross_weekend_gap():
    frame = _frame(14)
    frame["timestamp"] = pd.date_range("2026-01-02 22:45", periods=14, freq="5min", tz="UTC")
    event = _frame(4)
    event["timestamp"] = pd.to_datetime(["2026-01-02 23:55", "2026-01-05 00:00", "2026-01-05 00:05", "2026-01-05 00:10"], utc=True)
    frame = pd.concat([frame, event], ignore_index=True)
    labeled, report = make_triple_barrier_labels(frame, horizon_bars=3, profit_multiple=.2)
    assert pd.Timestamp("2026-01-02 23:55", tz="UTC") not in set(labeled.timestamp)
    assert report["missing_future_rows"] > 0


def test_triple_barrier_timeout_zero_return_is_not_up():
    frame = _frame(20)
    frame[["silver_open", "silver_high", "silver_low", "silver_close"]] = 100.0
    frame.loc[:13, "silver_high"] = 101.0
    frame.loc[:13, "silver_low"] = 99.0
    signed, report = make_triple_barrier_labels(frame, horizon_bars=3, profit_multiple=.2, timeout_policy="sign")
    neutral, _ = make_triple_barrier_labels(frame, horizon_bars=3, profit_multiple=.2, timeout_policy="neutral")
    event_time = frame.loc[14, "timestamp"]
    assert report["timeout_zero_rows"] > 0
    assert event_time not in set(signed.timestamp)
    assert neutral.loc[neutral.timestamp == event_time, "triple_barrier_label"].iloc[0] == 0.5

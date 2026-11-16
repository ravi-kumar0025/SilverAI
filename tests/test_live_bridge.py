from datetime import timedelta

import pandas as pd

from src.data.live_provider import ReplaySource
from src.data.live_log_source import LiveLogSource
from src.prediction.store import PredictionStore
from src.spot_config import SPOT_CONFIG
from backend.services.prediction_service import PredictionService


def _event(timestamp, close=50.0, **extra):
    return {
        "symbol": "xagusd",
        "timestamp": timestamp,
        "open": close - 0.01,
        "high": close + 0.02,
        "low": close - 0.02,
        "close": close,
        **extra,
    }


def test_replay_uses_live_guards_for_forming_duplicates_and_gaps():
    events = [
        _event("2026-01-01T10:05:00Z", forming=True),
        _event("2026-01-01T10:00:00Z"),
        _event("2026-01-01T10:00:00Z", close=51.0),
        _event("2026-01-01T10:03:00Z", close=52.0),
    ]
    replay = ReplaySource(events, now="2026-01-01T10:04:00Z")
    accepted = replay.run()
    status = replay.status()
    assert [item["timestamp"] for item in accepted] == ["2026-01-01T10:00:00+00:00", "2026-01-01T10:03:00+00:00"]
    assert status["forming_discarded"] == 1
    assert status["duplicate_discarded"] == 1
    assert status["gap_events"] == 1
    assert status["fresh"] is True


def test_replay_freshness_stale_and_exact_boundary_use_injected_clock():
    event = [_event("2026-01-01T10:00:00Z")]
    base = pd.Timestamp("2026-01-01T10:00:00Z")
    threshold = SPOT_CONFIG.live_stale_after_seconds
    fresh = ReplaySource(event, now=(base + pd.Timedelta(seconds=threshold - 1)).isoformat())
    boundary = ReplaySource(event, now=(base + pd.Timedelta(seconds=threshold)).isoformat())
    stale = ReplaySource(event, now=(base + pd.Timedelta(seconds=threshold + 1)).isoformat())
    assert fresh.run()
    assert boundary.run()
    assert stale.run()
    assert fresh.status()["fresh"] is True
    assert boundary.status()["fresh"] is True
    assert stale.status()["fresh"] is False


def test_live_cycle_evaluates_pending_silver_forecasts_without_companions():
    class EmptyCompanionSource:
        def candles(self, symbol, minutes):
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

    class RecordingStore:
        def __init__(self):
            self.frames = []

        def evaluate(self, frame):
            self.frames.append(frame.copy())
            return 1

    service = PredictionService.__new__(PredictionService)
    service._historical_frames = {
        "xagusd": pd.DataFrame([{"timestamp": pd.Timestamp("2026-01-01T10:00:00Z"), "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.5}]),
        "xauusd": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"]),
        "eurusd": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"]),
    }
    service._live_logs = EmptyCompanionSource()
    service.store = RecordingStore()
    service._on_live_candle({"symbol": "xagusd"})
    assert len(service.store.frames) == 1
    assert list(service.store.frames[0].columns) == ["datetime_et", "close"]


def _prediction(timestamp, signal="BULLISH"):
    return {
        "prediction_timestamp": timestamp,
        "close_price": 50.0,
        "predicted_direction_name": "UP",
        "class_probabilities": {"down": 0.2, "up": 0.8},
        "calibrated_probabilities": {"down": 0.2, "up": 0.8},
        "confidence": 0.8,
        "signal": {"signal": signal},
        "model_metadata": {"classifier_name": "test-model"},
        "candle_finalized": True,
        "forecast_horizon_minutes": 60,
        "forecast_horizon_bars": 12,
        "instrument": "XAGUSD_SPOT",
        "contract": "XAG/USD",
    }


def test_prediction_is_immutable_and_restart_does_not_duplicate(tmp_path):
    path = tmp_path / "predictions.sqlite3"
    first = PredictionStore(path)
    saved = first.record(_prediction("2026-01-01T10:00:00Z"))
    second = PredictionStore(path)
    duplicate = second.record(_prediction("2026-01-01T10:00:00Z", signal="BEARISH"))
    rows = second.list(limit=10)
    assert len(rows) == 1
    assert duplicate["final_signal"] == saved["final_signal"] == "BULLISH"


def test_replay_callback_only_sees_closed_unique_candles():
    emitted = []
    replay = ReplaySource(
        [
            _event("2026-01-01T10:03:00Z", forming=True),
            _event("2026-01-01T10:00:00Z", close=50.0),
            _event("2026-01-01T10:00:00Z", close=51.0),
            _event("2026-01-01T10:02:00Z", close=52.0),
        ],
        now="2026-01-01T10:04:00Z",
        on_candle=lambda candle: emitted.append({"timestamp": candle["timestamp"], "direction": "Up" if candle["close"] >= candle["open"] else "Down"}),
    )
    replay.run()
    assert emitted == [
        {"timestamp": "2026-01-01T10:00:00+00:00", "direction": "Up"},
        {"timestamp": "2026-01-01T10:02:00+00:00", "direction": "Up"},
    ]


def test_live_log_source_discards_forming_deduplicates_and_builds_complete_5m(tmp_path):
    import json
    rows = []
    for minute in range(5):
        rows.append({"poll_id": "p1", "recv_ms": 1_800_000_600_000, "candles": [[1_800_000_000_000 + minute * 60_000, 50, 50.1, 49.9, 50 + minute * .01]]})
    rows.append({"poll_id": "p2", "recv_ms": 1_800_000_600_000, "candles": [[1_800_000_000_000, 50, 50.2, 49.8, 51]]})
    rows.append({"poll_id": "p3", "recv_ms": 1_800_000_000_000, "candles": [[1_800_000_300_000, 50, 50, 50, 50]]})
    (tmp_path / "xagusd_m1_raw.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    source = LiveLogSource(tmp_path, clock="2027-01-15T08:10:00Z")
    candles = source.candles()
    assert len(candles) == 1
    assert candles.iloc[0].close == 50.04
    assert source.status()["fresh"] is False


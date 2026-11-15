from datetime import timedelta

from src.data.live_provider import ReplaySource
from src.prediction.store import PredictionStore


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
    fresh = ReplaySource(event, now="2026-01-01T10:01:29Z")
    boundary = ReplaySource(event, now="2026-01-01T10:01:30Z")
    stale = ReplaySource(event, now="2026-01-01T10:01:31Z")
    assert fresh.run()
    assert boundary.run()
    assert stale.run()
    assert fresh.status()["fresh"] is True
    assert boundary.status()["fresh"] is True
    assert stale.status()["fresh"] is False


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


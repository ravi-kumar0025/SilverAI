from src.prediction.store import PredictionStore


def _prediction(prediction_id, timestamp, price, signal="UP"):
    return {
        "prediction_id": prediction_id,
        "prediction_timestamp": timestamp,
        "close_price": price,
        "signal": {"signal": signal},
        "model_metadata": {"model_version": "paper-test-v1"},
        "candle_finalized": True,
        "forecast_horizon_minutes": 60,
    }


def test_paper_account_requires_live_and_manages_one_hour_position(tmp_path):
    store = PredictionStore(tmp_path / "paper.sqlite3")
    assert store.paper_process(_prediction("p0", "2026-01-01T10:00:00Z", 50), live=False, fresh=False)["accepted"] is False
    store.paper_start(1_000)

    opened = store.paper_process(_prediction("p1", "2026-01-01T10:00:00Z", 50), live=True, fresh=True)
    assert opened["action"] == "opened"
    duplicate = store.paper_process(_prediction("p1", "2026-01-01T10:05:00Z", 51), live=True, fresh=True)
    assert duplicate["action"] == "held_or_ignored"
    status = store.paper_status()
    assert len(status["open_positions"]) == 1
    assert status["available_balance"] == 0

    store.paper_stop()
    closed = store.paper_process(_prediction("p2", "2026-01-01T11:00:00Z", 51), live=True, fresh=True)
    assert closed["action"] == "closed"
    status = store.paper_status()
    assert status["status"] == "stopped"
    assert len(status["open_positions"]) == 0
    assert status["trade_count"] == 1
    assert status["realized_pnl"] > 0


def test_paper_reset_requires_stopped_and_clears_only_paper_tables(tmp_path):
    store = PredictionStore(tmp_path / "paper.sqlite3")
    store.paper_start()
    store.paper_process(_prediction("p1", "2026-01-01T10:00:00Z", 50), live=True, fresh=True)
    try:
        store.paper_reset()
    except ValueError as exc:
        assert "Stop" in str(exc)
    else:
        raise AssertionError("running account must not reset")
    store.paper_stop()
    store.paper_process(_prediction("p2", "2026-01-01T11:00:00Z", 50), live=True, fresh=True)
    store.paper_reset(2_000)
    status = store.paper_status()
    assert status["initial_balance"] == 2_000
    assert status["trade_count"] == 0
    assert status["open_positions"] == []


def test_clear_testing_data_preserves_database_file_and_requires_stopped(tmp_path):
    path = tmp_path / "paper.sqlite3"
    store = PredictionStore(path)
    store.paper_start()
    store.paper_process(_prediction("p1", "2026-01-01T10:00:00Z", 50), live=True, fresh=True)
    store.paper_stop()
    store.paper_process(_prediction("p2", "2026-01-01T11:00:00Z", 51), live=True, fresh=True)
    result = store.clear_testing_data()
    assert result["cleared"] is True
    assert result["predictions_deleted"] == 0
    assert result["paper_trades_deleted"] == 1
    assert path.is_file()
    assert store.paper_status()["trade_count"] == 0


def test_paper_mark_price_updates_account_without_prediction(tmp_path):
    store = PredictionStore(tmp_path / "paper.sqlite3")
    store.paper_start()
    marked = store.paper_mark_price(52.25, "2026-01-01T10:05:00Z")
    assert marked["current_market_price"] == 52.25
    assert marked["current_market_timestamp"].startswith("2026-01-01T10:05:00")
    assert marked["current_equity"] == 10_000

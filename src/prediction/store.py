"""Durable prediction log and outcome evaluation for live/replay testing."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from src.config import CONFIG
from src.spot_config import SPOT_CONFIG


FINAL_SIGNAL_NAMES = frozenset({"BULLISH", "BEARISH", "NEUTRAL", "UP", "DOWN", "NO_CLEAR_SIGNAL"})


def is_finalized_signal_record(record: dict[str, Any]) -> bool:
    """Return whether a persisted row is safe to expose as a final signal."""

    if not bool(record.get("candle_finalized")):
        return False
    if str(record.get("final_signal") or "").upper() not in FINAL_SIGNAL_NAMES:
        return False
    try:
        pd.to_datetime(record.get("candle_timestamp"), utc=True, errors="raise")
    except (TypeError, ValueError, OverflowError):
        return False
    return True


class PredictionStore:
    """SQLite-backed append-only prediction journal.

    Prediction fields are never overwritten by outcome fields. A unique key
    prevents duplicate signals after a process restart or reconnect.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init(self) -> None:
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    instrument TEXT NOT NULL,
                    contract TEXT NOT NULL,
                    candle_timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    close_price REAL,
                    predicted_direction TEXT NOT NULL,
                    probability_down REAL,
                    probability_neutral REAL,
                    probability_up REAL,
                    predicted_return REAL,
                    predicted_price REAL,
                    price_low REAL,
                    price_high REAL,
                    expected_direction TEXT,
                    raw_probability_down REAL,
                    raw_probability_neutral REAL,
                    raw_probability_up REAL,
                    calibrated_probability_down REAL,
                    calibrated_probability_neutral REAL,
                    calibrated_probability_up REAL,
                    confidence REAL,
                    confidence_kind TEXT,
                    model_version TEXT,
                    configuration_version TEXT,
                    final_signal TEXT,
                    candle_finalized INTEGER NOT NULL DEFAULT 0,
                    scenario_payload TEXT,
                    candle_data_ref TEXT,
                    horizon_minutes INTEGER NOT NULL,
                    horizon_bars INTEGER NOT NULL,
                    feature_payload TEXT,
                    actual_price REAL,
                    actual_return REAL,
                    actual_direction TEXT,
                    outcome_status TEXT NOT NULL DEFAULT 'pending',
                    outcome_timestamp TEXT,
                    outcome_evaluated_at TEXT,
                    UNIQUE(instrument, contract, candle_timestamp, model_version)
                )"""
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(predictions)")}
            migrations = {
                "raw_probability_down": "REAL", "raw_probability_neutral": "REAL", "raw_probability_up": "REAL",
                "calibrated_probability_down": "REAL", "calibrated_probability_neutral": "REAL", "calibrated_probability_up": "REAL",
                "configuration_version": "TEXT", "final_signal": "TEXT", "candle_finalized": "INTEGER NOT NULL DEFAULT 0", "scenario_payload": "TEXT", "candle_data_ref": "TEXT",
                "outcome_timestamp": "TEXT",
                "predicted_price": "REAL", "price_low": "REAL", "price_high": "REAL", "expected_direction": "TEXT",
            }
            for column, kind in migrations.items():
                if column not in columns:
                    db.execute(f"ALTER TABLE predictions ADD COLUMN {column} {kind}")
            db.execute(
                """CREATE TRIGGER IF NOT EXISTS prevent_prediction_forecast_update
                BEFORE UPDATE OF instrument, contract, candle_timestamp, created_at,
                    close_price, predicted_direction, probability_down, probability_neutral,
                    probability_up, predicted_return, raw_probability_down,
                    raw_probability_neutral, raw_probability_up, calibrated_probability_down,
                    calibrated_probability_neutral, calibrated_probability_up, confidence,
                    confidence_kind, model_version, configuration_version, final_signal,
                    candle_finalized, scenario_payload, candle_data_ref, horizon_minutes, horizon_bars,
                    feature_payload ON predictions
                WHEN OLD.instrument IS NOT NEW.instrument OR OLD.contract IS NOT NEW.contract
                    OR OLD.candle_timestamp IS NOT NEW.candle_timestamp OR OLD.created_at IS NOT NEW.created_at
                    OR OLD.close_price IS NOT NEW.close_price OR OLD.predicted_direction IS NOT NEW.predicted_direction
                    OR OLD.probability_down IS NOT NEW.probability_down OR OLD.probability_neutral IS NOT NEW.probability_neutral
                    OR OLD.probability_up IS NOT NEW.probability_up OR OLD.predicted_return IS NOT NEW.predicted_return
                    OR OLD.raw_probability_down IS NOT NEW.raw_probability_down OR OLD.raw_probability_neutral IS NOT NEW.raw_probability_neutral
                    OR OLD.raw_probability_up IS NOT NEW.raw_probability_up OR OLD.calibrated_probability_down IS NOT NEW.calibrated_probability_down
                    OR OLD.calibrated_probability_neutral IS NOT NEW.calibrated_probability_neutral OR OLD.calibrated_probability_up IS NOT NEW.calibrated_probability_up
                    OR OLD.confidence IS NOT NEW.confidence OR OLD.confidence_kind IS NOT NEW.confidence_kind
                    OR OLD.model_version IS NOT NEW.model_version OR OLD.configuration_version IS NOT NEW.configuration_version
                    OR OLD.final_signal IS NOT NEW.final_signal OR OLD.scenario_payload IS NOT NEW.scenario_payload
                    OR OLD.candle_finalized IS NOT NEW.candle_finalized
                    OR OLD.candle_data_ref IS NOT NEW.candle_data_ref OR OLD.horizon_minutes IS NOT NEW.horizon_minutes
                    OR OLD.horizon_bars IS NOT NEW.horizon_bars OR OLD.feature_payload IS NOT NEW.feature_payload
                BEGIN SELECT RAISE(ABORT, 'prediction forecast fields are immutable'); END"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS paper_account (
                    account_id INTEGER PRIMARY KEY CHECK (account_id = 1),
                    status TEXT NOT NULL DEFAULT 'stopped',
                    initial_balance REAL NOT NULL,
                    available_balance REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    total_costs REAL NOT NULL DEFAULT 0,
                    max_equity REAL NOT NULL,
                    max_drawdown REAL NOT NULL DEFAULT 0,
                    current_price REAL,
                    current_timestamp TEXT,
                    started_at TEXT,
                    stopped_at TEXT,
                    updated_at TEXT NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    model_version TEXT,
                    entry_timestamp TEXT NOT NULL,
                    exit_timestamp TEXT,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    position_size REAL NOT NULL,
                    notional REAL NOT NULL,
                    gross_pnl REAL NOT NULL DEFAULT 0,
                    costs REAL NOT NULL DEFAULT 0,
                    net_pnl REAL NOT NULL DEFAULT 0,
                    exit_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            account = db.execute("SELECT account_id FROM paper_account WHERE account_id=1").fetchone()
            if account is None:
                now = datetime.now(timezone.utc).isoformat()
                db.execute(
                    "INSERT INTO paper_account(account_id,status,initial_balance,available_balance,max_equity,updated_at) VALUES(1,'stopped',10000,10000,10000,?)",
                    (now,),
                )

    def record(self, prediction: dict[str, Any]) -> dict[str, Any]:
        prediction_id = str(prediction.get("prediction_id") or uuid.uuid4())
        created_at = str(prediction.get("created_at") or datetime.now(timezone.utc).isoformat())
        model = prediction.get("model_metadata") or {}
        probabilities = prediction.get("class_probabilities") or {}
        horizon_minutes = int(prediction.get("forecast_horizon_minutes", CONFIG.forecast_horizon_minutes))
        horizon_bars = int(prediction.get("forecast_horizon_bars", CONFIG.forecast_horizon_bars))
        row = {
            "prediction_id": prediction_id,
            "instrument": prediction.get("instrument", "XAG/USD Spot"),
            "contract": prediction.get("contract", "XAG/USD"),
            "candle_timestamp": str(prediction.get("prediction_timestamp") or prediction.get("candle_timestamp") or ""),
            "created_at": created_at,
            "close_price": prediction.get("close_price"),
            "predicted_direction": prediction.get("predicted_direction_name", "NEUTRAL"),
            "probability_down": probabilities.get("down"),
            "probability_neutral": probabilities.get("neutral"),
            "probability_up": probabilities.get("up"),
            "predicted_return": prediction.get("predicted_return", prediction.get("predicted_return_15m")),
            "predicted_price": prediction.get("predicted_price"),
            "price_low": prediction.get("price_low"),
            "price_high": prediction.get("price_high"),
            "expected_direction": prediction.get("expected_direction"),
            "raw_probability_down": probabilities.get("down"),
            "raw_probability_neutral": probabilities.get("neutral"),
            "raw_probability_up": probabilities.get("up"),
            "calibrated_probability_down": (prediction.get("calibrated_probabilities") or {}).get("down"),
            "calibrated_probability_neutral": (prediction.get("calibrated_probabilities") or {}).get("neutral"),
            "calibrated_probability_up": (prediction.get("calibrated_probabilities") or {}).get("up"),
            "confidence": prediction.get("confidence"),
            "confidence_kind": prediction.get("confidence_kind"),
            "model_version": prediction.get("model_version") or model.get("classifier_name", "unknown"),
            "configuration_version": prediction.get("configuration_version") or model.get("configuration_version"),
            "final_signal": (prediction.get("signal") or {}).get("signal"),
            "candle_finalized": 1 if prediction.get("candle_finalized") is True else 0,
            "scenario_payload": json.dumps(prediction.get("scenarios"), default=str),
            "candle_data_ref": prediction.get("candle_data_ref"),
            "horizon_minutes": horizon_minutes,
            "horizon_bars": horizon_bars,
            "feature_payload": json.dumps(prediction.get("features"), default=str),
        }
        if not row["candle_timestamp"]:
            raise ValueError("A confirmed candle timestamp is required for persistence.")
        with self._connect() as db:
            db.execute(
                """INSERT OR IGNORE INTO predictions
                (prediction_id, instrument, contract, candle_timestamp, created_at,
                 close_price, predicted_direction, probability_down, probability_neutral,
                 probability_up, predicted_return, predicted_price, price_low, price_high, expected_direction, confidence, confidence_kind, model_version,
                 horizon_minutes, horizon_bars, feature_payload, raw_probability_down,
                 raw_probability_neutral, raw_probability_up, calibrated_probability_down,
                 calibrated_probability_neutral, calibrated_probability_up, configuration_version,
                 final_signal, candle_finalized, scenario_payload, candle_data_ref)
                VALUES (:prediction_id, :instrument, :contract, :candle_timestamp, :created_at,
                 :close_price, :predicted_direction, :probability_down, :probability_neutral,
                 :probability_up, :predicted_return, :predicted_price, :price_low, :price_high, :expected_direction, :confidence, :confidence_kind, :model_version,
                 :horizon_minutes, :horizon_bars, :feature_payload, :raw_probability_down,
                 :raw_probability_neutral, :raw_probability_up, :calibrated_probability_down,
                 :calibrated_probability_neutral, :calibrated_probability_up, :configuration_version,
                 :final_signal, :candle_finalized, :scenario_payload, :candle_data_ref)""",
                row,
            )
            saved = db.execute(
                "SELECT * FROM predictions WHERE instrument=? AND contract=? AND candle_timestamp=? AND model_version=?",
                (row["instrument"], row["contract"], row["candle_timestamp"], row["model_version"]),
            ).fetchone()
        return dict(saved)

    def list(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM predictions"
        args: list[Any] = []
        if status:
            query += " WHERE outcome_status=?"
            args.append(status)
        query += " ORDER BY candle_timestamp DESC LIMIT ?"
        args.append(limit)
        with self._connect() as db:
            return [dict(row) for row in db.execute(query, args).fetchall()]

    def list_finalized_signals(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return only valid final analytical signals from closed candles."""

        placeholders = ",".join("?" for _ in FINAL_SIGNAL_NAMES)
        query = f"SELECT * FROM predictions WHERE candle_finalized=1 AND final_signal IN ({placeholders}) ORDER BY candle_timestamp DESC"
        with self._connect() as db:
            rows = [dict(row) for row in db.execute(query, tuple(FINAL_SIGNAL_NAMES)).fetchall()]
        return [row for row in rows if is_finalized_signal_record(row)][:limit]

    def evaluate(self, candles: pd.DataFrame) -> int:
        """Evaluate only complete, contiguous future windows.

        The exit is selected by the stored horizon and timestamp, never by
        evaluator runtime.  Missing bars and session gaps remain pending.
        """
        required = {"datetime_et", "close"}
        if not required.issubset(candles.columns):
            raise ValueError("Outcome candles require datetime_et and close columns.")
        ordered = candles.sort_values("datetime_et").drop_duplicates("datetime_et").reset_index(drop=True)
        ordered["datetime_et"] = pd.to_datetime(ordered["datetime_et"], utc=True)
        updated = 0
        with self._connect() as db:
            pending = db.execute(
                "SELECT * FROM predictions WHERE outcome_status='pending' AND close_price IS NOT NULL"
            ).fetchall()
            for row in pending:
                prediction_timestamp = pd.to_datetime(row["candle_timestamp"], utc=True)
                candle_timestamps = ordered["datetime_et"]
                matches = ordered.index[candle_timestamps == prediction_timestamp].tolist()
                if not matches:
                    continue
                future_index = matches[0] + int(row["horizon_bars"])
                if future_index >= len(ordered):
                    continue
                anchor_index = matches[0]
                horizon_minutes = int(row["horizon_minutes"])
                future_timestamp = candle_timestamps.iloc[future_index]
                if (future_timestamp - prediction_timestamp).total_seconds() != horizon_minutes * 60:
                    continue
                if anchor_index < future_index:
                    deltas = candle_timestamps.iloc[anchor_index + 1 : future_index + 1].diff().dropna()
                    if not deltas.eq(deltas.iloc[0]).all() or deltas.iloc[0].total_seconds() != (horizon_minutes / int(row["horizon_bars"])) * 60:
                        continue
                future_price = float(ordered.iloc[future_index].close)
                actual_return = (future_price - float(row["close_price"])) / float(row["close_price"])
                actual_direction = "UP" if actual_return > 0 else "DOWN" if actual_return < 0 else "NEUTRAL"
                db.execute(
                    "UPDATE predictions SET actual_price=?, actual_return=?, actual_direction=?, outcome_status='evaluated', outcome_timestamp=?, outcome_evaluated_at=? WHERE prediction_id=? AND outcome_status='pending'",
                    (future_price, actual_return, actual_direction, future_timestamp.isoformat(), datetime.now(timezone.utc).isoformat(), row["prediction_id"]),
                )
                updated += 1
        return updated

    def _paper_account_row(self, db: sqlite3.Connection) -> dict[str, Any]:
        row = db.execute("SELECT * FROM paper_account WHERE account_id=1").fetchone()
        if row is None:
            raise RuntimeError("Paper account is not initialized.")
        return dict(row)

    def paper_start(self, initial_balance: float | None = None) -> dict[str, Any]:
        with self._connect() as db:
            account = self._paper_account_row(db)
            if initial_balance is not None:
                if initial_balance <= 0:
                    raise ValueError("Initial paper balance must be positive.")
                if db.execute("SELECT 1 FROM paper_trades LIMIT 1").fetchone() is not None:
                    raise ValueError("Initial balance can only be changed before the first paper trade.")
                account["initial_balance"] = float(initial_balance)
                account["available_balance"] = float(initial_balance)
                account["max_equity"] = float(initial_balance)
            now = datetime.now(timezone.utc).isoformat()
            db.execute(
                "UPDATE paper_account SET status='running', initial_balance=?, available_balance=?, started_at=COALESCE(started_at, ?), stopped_at=NULL, updated_at=? WHERE account_id=1",
                (account["initial_balance"], account["available_balance"], now, now),
            )
            return self._paper_account_row(db)

    def paper_stop(self) -> dict[str, Any]:
        with self._connect() as db:
            now = datetime.now(timezone.utc).isoformat()
            db.execute("UPDATE paper_account SET status='stopped', stopped_at=?, updated_at=? WHERE account_id=1", (now, now))
            return self._paper_account_row(db)

    def paper_reset(self, initial_balance: float = 10_000.0) -> dict[str, Any]:
        if initial_balance <= 0:
            raise ValueError("Initial paper balance must be positive.")
        with self._connect() as db:
            account = self._paper_account_row(db)
            if account["status"] == "running":
                raise ValueError("Stop paper trading before resetting the virtual account.")
            if db.execute("SELECT 1 FROM paper_trades WHERE status='open' LIMIT 1").fetchone() is not None:
                raise ValueError("Cannot reset while an open paper position exists.")
            db.execute("DELETE FROM paper_trades")
            now = datetime.now(timezone.utc).isoformat()
            db.execute(
                """UPDATE paper_account SET status='stopped', initial_balance=?, available_balance=?, realized_pnl=0,
                   total_costs=0, max_equity=?, max_drawdown=0, current_price=NULL, current_timestamp=NULL,
                   started_at=NULL, stopped_at=?, updated_at=? WHERE account_id=1""",
                (float(initial_balance), float(initial_balance), float(initial_balance), now, now),
            )
            return self._paper_account_row(db)

    def clear_testing_data(self) -> dict[str, Any]:
        """Clear live-testing and paper-session rows without deleting the DB file."""
        with self._connect() as db:
            account = self._paper_account_row(db)
            if account["status"] == "running":
                raise ValueError("Stop paper trading before clearing testing data.")
            prediction_count = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            trade_count = db.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
            open_count = db.execute("SELECT COUNT(*) FROM paper_trades WHERE status='open'").fetchone()[0]
            if open_count:
                raise ValueError("Close the open paper position before clearing testing data.")
            db.execute("BEGIN")
            db.execute("DELETE FROM paper_trades")
            db.execute("DELETE FROM predictions")
            now = datetime.now(timezone.utc).isoformat()
            db.execute(
                """UPDATE paper_account SET status='stopped', initial_balance=10000,
                   available_balance=10000, realized_pnl=0, total_costs=0,
                   max_equity=10000, max_drawdown=0, current_price=NULL,
                   current_timestamp=NULL, started_at=NULL, stopped_at=?, updated_at=?
                   WHERE account_id=1""",
                (now, now),
            )
            return {
                "cleared": True,
                "predictions_deleted": int(prediction_count),
                "paper_trades_deleted": int(trade_count),
                "database_file_preserved": str(self.path),
            }

    def paper_process(self, prediction: dict[str, Any], *, live: bool, fresh: bool) -> dict[str, Any]:
        """Manage one closed live prediction using only the supplied live price."""
        if not live or not fresh:
            return {"accepted": False, "reason": "live_data_unavailable_or_stale"}
        if not prediction.get("candle_finalized"):
            return {"accepted": False, "reason": "candle_not_finalized"}
        try:
            price = float(prediction["close_price"])
            timestamp = pd.to_datetime(prediction["prediction_timestamp"], utc=True)
        except (KeyError, TypeError, ValueError):
            return {"accepted": False, "reason": "invalid_live_price_or_timestamp"}
        if not price > 0:
            return {"accepted": False, "reason": "invalid_live_price"}
        signal = str((prediction.get("signal") or {}).get("signal") or "").upper()
        direction = {"UP": "LONG", "DOWN": "SHORT", "BULLISH": "LONG", "BEARISH": "SHORT"}.get(signal)
        cost_rate = (float(SPOT_CONFIG.spread_bps) + float(SPOT_CONFIG.slippage_bps)) / 10_000.0
        with self._connect() as db:
            account = self._paper_account_row(db)
            open_trade = db.execute("SELECT * FROM paper_trades WHERE status='open' ORDER BY entry_timestamp LIMIT 1").fetchone()
            closed_trade = None
            if open_trade is not None:
                entry_time = pd.to_datetime(open_trade["entry_timestamp"], utc=True)
                horizon = int(prediction.get("forecast_horizon_minutes") or CONFIG.forecast_horizon_minutes)
                if timestamp >= entry_time + pd.Timedelta(minutes=horizon):
                    entry_price = float(open_trade["entry_price"])
                    raw_return = (price - entry_price) / entry_price
                    if open_trade["direction"] == "SHORT":
                        raw_return = -raw_return
                    gross = float(open_trade["notional"]) * raw_return
                    costs = float(open_trade["notional"]) * cost_rate * 2
                    net = gross - costs
                    now = datetime.now(timezone.utc).isoformat()
                    db.execute(
                        """UPDATE paper_trades SET status='closed', exit_timestamp=?, exit_price=?, gross_pnl=?, costs=?, net_pnl=?, exit_reason='12_completed_candles', updated_at=? WHERE trade_id=?""",
                        (timestamp.isoformat(), price, gross, costs, net, now, open_trade["trade_id"]),
                    )
                    db.execute(
                        """UPDATE paper_account SET available_balance=available_balance+?+?, realized_pnl=realized_pnl+?, total_costs=total_costs+?, updated_at=? WHERE account_id=1""",
                        (float(open_trade["notional"]), net, net, costs, now),
                    )
                    closed_trade = {**dict(open_trade), "status": "closed", "exit_timestamp": timestamp.isoformat(), "exit_price": price, "gross_pnl": gross, "costs": costs, "net_pnl": net, "exit_reason": "12_completed_candles"}
                    open_trade = None
            if open_trade is None and account["status"] == "running" and direction is not None:
                prediction_id = str(prediction.get("prediction_id") or "")
                if not prediction_id:
                    return {"accepted": False, "reason": "prediction_id_missing", "closed_trade": closed_trade}
                exists = db.execute("SELECT 1 FROM paper_trades WHERE prediction_id=?", (prediction_id,)).fetchone()
                if exists is None:
                    notional = min(1000.0, float(account["available_balance"]))
                    if notional > 0:
                        trade_id = str(uuid.uuid4())
                        now = datetime.now(timezone.utc).isoformat()
                        db.execute(
                            """INSERT INTO paper_trades(trade_id,prediction_id,status,direction,signal,model_version,entry_timestamp,entry_price,position_size,notional,created_at,updated_at)
                               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (trade_id, prediction_id, "open", direction, signal, (prediction.get("model_metadata") or {}).get("model_version"), timestamp.isoformat(), price, notional / price, notional, now, now),
                        )
                        db.execute("UPDATE paper_account SET available_balance=available_balance-?, current_price=?, current_timestamp=?, updated_at=? WHERE account_id=1", (notional, price, timestamp.isoformat(), now))
                        return {"accepted": True, "action": "opened", "trade_id": trade_id, "closed_trade": closed_trade}
            now = datetime.now(timezone.utc).isoformat()
            db.execute("UPDATE paper_account SET current_price=?, current_timestamp=?, updated_at=? WHERE account_id=1", (price, timestamp.isoformat(), now))
            return {"accepted": True, "action": "closed" if closed_trade else "held_or_ignored", "closed_trade": closed_trade}

    def paper_mark_price(self, price: float, timestamp: Any) -> dict[str, Any]:
        """Mark the virtual account to the newest accepted live market candle.

        Mark-to-market must not depend on a model prediction being generated.
        This keeps the displayed price and unrealized equity current when a
        companion feed is delayed or model scoring is temporarily unavailable.
        """
        try:
            value = float(price)
            observed_at = pd.to_datetime(timestamp, utc=True)
        except (TypeError, ValueError):
            raise ValueError("Invalid paper market price or timestamp.") from None
        if not value > 0 or pd.isna(observed_at):
            raise ValueError("Invalid paper market price or timestamp.")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                """UPDATE paper_account
                   SET current_price=?, current_timestamp=?, updated_at=?
                   WHERE account_id=1""",
                (value, observed_at.isoformat(), now),
            )
        return self.paper_status(current_price=value)

    def paper_status(self, current_price: float | None = None) -> dict[str, Any]:
        with self._connect() as db:
            account = self._paper_account_row(db)
            trades = [dict(row) for row in db.execute("SELECT * FROM paper_trades ORDER BY entry_timestamp DESC").fetchall()]
        price = float(current_price) if current_price is not None else account.get("current_price")
        open_positions = [row for row in trades if row["status"] == "open"]
        unrealized = 0.0
        for row in open_positions:
            if price is not None:
                raw_return = (float(price) - row["entry_price"]) / row["entry_price"]
                unrealized += row["notional"] * (raw_return if row["direction"] == "LONG" else -raw_return)
        realized = float(account["realized_pnl"])
        equity = float(account["available_balance"]) + sum(row["notional"] for row in open_positions) + unrealized
        max_equity = max(float(account["max_equity"]), equity)
        drawdown = max_equity - equity
        with self._connect() as db:
            db.execute("UPDATE paper_account SET max_equity=?, max_drawdown=MAX(max_drawdown,?), current_price=COALESCE(?,current_price), updated_at=? WHERE account_id=1", (max_equity, drawdown, price, datetime.now(timezone.utc).isoformat()))
        closed = [row for row in trades if row["status"] == "closed"]
        wins = [row for row in closed if row["net_pnl"] > 0]
        losses = [row for row in closed if row["net_pnl"] < 0]
        return {
            "paper_only": True,
            "status": account["status"],
            "initial_balance": float(account["initial_balance"]),
            "available_balance": float(account["available_balance"]),
            "current_equity": equity,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "net_pnl": realized + unrealized,
            "total_return_pct": (realized + unrealized) / float(account["initial_balance"]) if account["initial_balance"] else 0.0,
            "open_positions": open_positions,
            "closed_trades": closed,
            "trade_count": len(closed),
            "win_rate": len(wins) / len(closed) if closed else None,
            "average_winning_trade": sum(row["net_pnl"] for row in wins) / len(wins) if wins else None,
            "average_losing_trade": sum(row["net_pnl"] for row in losses) / len(losses) if losses else None,
            "max_drawdown": max(float(account["max_drawdown"]), drawdown),
            "total_costs": float(account["total_costs"]),
            "latest_execution_timestamp": account.get("updated_at"),
            "current_market_price": price,
            "current_market_timestamp": account.get("current_timestamp"),
        }

__all__ = ["FINAL_SIGNAL_NAMES", "PredictionStore", "is_finalized_signal_record"]

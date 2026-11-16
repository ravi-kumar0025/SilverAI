"""Durable prediction log and outcome evaluation for live/replay testing."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from src.project_config import CONFIG


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

    def list(
        self,
        status: str | None = None,
        limit: int = 100,
        model_version: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM predictions"
        args: list[Any] = []
        filters: list[str] = []
        if status:
            filters.append("outcome_status=?")
            args.append(status)
        if model_version:
            filters.append("model_version=?")
            args.append(model_version)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY candle_timestamp DESC LIMIT ?"
        args.append(limit)
        with self._connect() as db:
            return [dict(row) for row in db.execute(query, args).fetchall()]

    def list_finalized_signals(
        self,
        limit: int = 100,
        model_version: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return only valid final analytical signals from closed candles."""

        placeholders = ",".join("?" for _ in FINAL_SIGNAL_NAMES)
        query = f"SELECT * FROM predictions WHERE candle_finalized=1 AND final_signal IN ({placeholders})"
        args: list[Any] = list(FINAL_SIGNAL_NAMES)
        if model_version:
            query += " AND model_version=?"
            args.append(model_version)
        query += " ORDER BY candle_timestamp DESC"
        with self._connect() as db:
            rows = [dict(row) for row in db.execute(query, args).fetchall()]
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


__all__ = ["FINAL_SIGNAL_NAMES", "PredictionStore", "is_finalized_signal_record"]

"""Durable prediction log and outcome evaluation for live/replay testing."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


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
                    confidence REAL,
                    confidence_kind TEXT,
                    model_version TEXT,
                    horizon_minutes INTEGER NOT NULL,
                    horizon_bars INTEGER NOT NULL,
                    feature_payload TEXT,
                    actual_price REAL,
                    actual_return REAL,
                    actual_direction TEXT,
                    outcome_status TEXT NOT NULL DEFAULT 'pending',
                    outcome_evaluated_at TEXT,
                    UNIQUE(instrument, contract, candle_timestamp, model_version)
                )"""
            )

    def record(self, prediction: dict[str, Any]) -> dict[str, Any]:
        prediction_id = str(prediction.get("prediction_id") or uuid.uuid4())
        created_at = str(prediction.get("created_at") or datetime.now(timezone.utc).isoformat())
        model = prediction.get("model_metadata") or {}
        probabilities = prediction.get("class_probabilities") or {}
        horizon_minutes = int(prediction.get("forecast_horizon_minutes", 15))
        horizon_bars = int(prediction.get("forecast_horizon_bars", max(1, horizon_minutes // 5)))
        row = {
            "prediction_id": prediction_id,
            "instrument": prediction.get("instrument", "COMEX Silver Futures"),
            "contract": prediction.get("contract", "SI"),
            "candle_timestamp": str(prediction.get("prediction_timestamp") or prediction.get("candle_timestamp") or ""),
            "created_at": created_at,
            "close_price": prediction.get("close_price"),
            "predicted_direction": prediction.get("predicted_direction_name", "NEUTRAL"),
            "probability_down": probabilities.get("down"),
            "probability_neutral": probabilities.get("neutral"),
            "probability_up": probabilities.get("up"),
            "predicted_return": prediction.get("predicted_return_15m"),
            "confidence": prediction.get("confidence"),
            "confidence_kind": prediction.get("confidence_kind"),
            "model_version": prediction.get("model_version") or model.get("classifier_name", "unknown"),
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
                 probability_up, predicted_return, confidence, confidence_kind, model_version,
                 horizon_minutes, horizon_bars, feature_payload)
                VALUES (:prediction_id, :instrument, :contract, :candle_timestamp, :created_at,
                 :close_price, :predicted_direction, :probability_down, :probability_neutral,
                 :probability_up, :predicted_return, :confidence, :confidence_kind, :model_version,
                 :horizon_minutes, :horizon_bars, :feature_payload)""",
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

    def evaluate(self, candles: pd.DataFrame) -> int:
        """Evaluate only predictions with enough later candles; never mutate forecasts."""
        ordered = candles.sort_values("datetime_et").reset_index(drop=True)
        updated = 0
        with self._connect() as db:
            pending = db.execute(
                "SELECT * FROM predictions WHERE outcome_status='pending' AND close_price IS NOT NULL"
            ).fetchall()
            for row in pending:
                prediction_timestamp = pd.to_datetime(row["candle_timestamp"], utc=True)
                candle_timestamps = pd.to_datetime(ordered["datetime_et"], utc=True)
                matches = ordered.index[candle_timestamps == prediction_timestamp].tolist()
                if not matches:
                    continue
                future_index = matches[0] + int(row["horizon_bars"])
                if future_index >= len(ordered):
                    continue
                future_price = float(ordered.iloc[future_index].close)
                actual_return = (future_price - float(row["close_price"])) / float(row["close_price"])
                actual_direction = "UP" if actual_return > 0 else "DOWN" if actual_return < 0 else "NEUTRAL"
                db.execute(
                    "UPDATE predictions SET actual_price=?, actual_return=?, actual_direction=?, outcome_status='evaluated', outcome_evaluated_at=? WHERE prediction_id=? AND outcome_status='pending'",
                    (future_price, actual_return, actual_direction, datetime.now(timezone.utc).isoformat(), row["prediction_id"]),
                )
                updated += 1
        return updated


__all__ = ["PredictionStore"]

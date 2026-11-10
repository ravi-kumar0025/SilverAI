"""Cached access to the SilverAI prediction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.config import MODEL_ROOT, PREDICTION_DB_PATH, SELECTION_PATH
from src.data.market_data import HistoricalCsvProvider, provider_status
from src.prediction.store import PredictionStore
from src.evaluation.paper_trading import evaluate_paper_trades
from src.models.predict import load_model_artifacts
from src.prediction.pipeline import generate_prediction


logger = logging.getLogger(__name__)


class PredictionService:
    """Lazy-load selected artifacts and retain the latest in-memory result."""

    def __init__(
        self,
        model_root: str | Path = MODEL_ROOT,
        selection_path: str | Path | None = SELECTION_PATH,
    ) -> None:
        self.model_root = Path(model_root)
        self.selection_path = selection_path
        self._artifacts: dict[str, Any] | None = None
        self._latest: dict[str, Any] | None = None
        self.store = PredictionStore(PREDICTION_DB_PATH)
        self.market_provider = HistoricalCsvProvider()

    def _get_artifacts(self) -> dict[str, Any]:
        if self._artifacts is None:
            logger.info("Loading selected SilverAI model artifacts")
            self._artifacts = load_model_artifacts(self.model_root, self.selection_path)
        return self._artifacts

    def health(self) -> dict[str, Any]:
        try:
            artifacts = self._get_artifacts()
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("Model health check failed: %s", type(exc).__name__)
            return {"status": "degraded", "models_available": False}
        return {
            "status": "ok",
            "models_available": True,
            "classifier_name": artifacts["classifier_name"],
            "regressor_name": artifacts["regressor_name"],
        }

    def models(self) -> dict[str, Any]:
        artifacts = self._get_artifacts()
        metadata = artifacts.get("metadata", {})
        return {
            "classifier_name": artifacts["classifier_name"],
            "regressor_name": artifacts["regressor_name"],
            "feature_names": metadata.get("feature_names"),
            "prediction_horizon_minutes": metadata.get("prediction_horizon_minutes", 15),
            "prediction_horizon_bars": metadata.get("prediction_horizon_bars", 3),
            "classification_target": metadata.get("classification_target"),
            "regression_target": metadata.get("regression_target"),
        }

    def predict(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        frame = pd.DataFrame(observations)
        result = generate_prediction(
            frame,
            model_root=self.model_root,
            selection_path=self.selection_path,
            artifacts=self._get_artifacts(),
        )
        self._latest = result["predictions"][-1]
        for observation, prediction in zip(observations, result["predictions"]):
            if prediction.get("prediction_timestamp"):
                record = {**prediction, "features": observation}
                if "close" in observation:
                    record["close_price"] = observation["close"]
                try:
                    self.store.record(record)
                except (TypeError, ValueError):
                    logger.warning("Prediction was returned but could not be persisted", exc_info=True)
        return result

    def latest(self) -> dict[str, Any] | None:
        return self._latest

    def prediction_history(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list(status=status, limit=limit)

    def evaluate_predictions(self, candles: pd.DataFrame) -> int:
        return self.store.evaluate(candles)

    def market_status(self) -> dict[str, Any]:
        return provider_status(self.market_provider)

    def paper_trading(self) -> dict[str, Any]:
        return evaluate_paper_trades(self.store.list(status="evaluated"))

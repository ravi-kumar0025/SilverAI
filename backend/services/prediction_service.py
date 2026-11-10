"""Cached access to the SilverAI prediction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.config import FROZEN_EVALUATION_PATH, MODEL_ROOT, PREDICTION_DB_PATH, SELECTION_PATH
from src.data.market_data import HistoricalCsvProvider, provider_status
from src.prediction.store import PredictionStore
from src.evaluation.paper_trading import evaluate_paper_trades
from src.models.predict import load_model_artifacts
from src.prediction.pipeline import generate_prediction
from src.config import CONFIG
from src.evaluation.frozen_mode import build_manifest, load_manifest, write_manifest


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
        self._diagnostics: list[dict[str, Any]] = []

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
            "provider": self.market_status(),
            "prediction_db": str(self.store.path),
        }

    def models(self) -> dict[str, Any]:
        artifacts = self._get_artifacts()
        metadata = artifacts.get("metadata", {})
        return {
            "classifier_name": artifacts["classifier_name"],
            "regressor_name": artifacts["regressor_name"],
            "feature_names": metadata.get("feature_names"),
            "prediction_horizon_minutes": metadata.get("prediction_horizon_minutes", CONFIG.forecast_horizon_minutes),
            "prediction_horizon_bars": metadata.get("prediction_horizon_bars", CONFIG.forecast_horizon_bars),
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
                except (TypeError, ValueError, OSError) as exc:
                    self._diagnostics.append({"kind": "persistence_failure", "error": str(exc)})
                    logger.warning("Prediction was returned but could not be persisted", exc_info=True)
        return result

    def latest(self) -> dict[str, Any] | None:
        if self._latest is not None:
            return self._latest
        history = self.store.list(limit=1)
        return history[0] if history else None

    def prediction_history(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list(status=status, limit=limit)

    def evaluate_predictions(self, candles: pd.DataFrame) -> int:
        return self.store.evaluate(candles)

    def market_status(self) -> dict[str, Any]:
        return provider_status(self.market_provider)

    def paper_trading(self) -> dict[str, Any]:
        return evaluate_paper_trades(self.store.list(status="evaluated"))

    def evaluation_metrics(self) -> dict[str, Any]:
        rows = self.store.list(status="evaluated", limit=100000)
        completed = len(rows)
        correct = sum(row.get("predicted_direction") == row.get("actual_direction") for row in rows)
        return {
            "predictions_total": len(self.store.list(limit=100000)),
            "completed_outcomes": completed,
            "pending_outcomes": len(self.store.list(status="pending", limit=100000)),
            "directional_accuracy": (correct / completed) if completed else None,
            "coverage": "historical replay only",
        }

    def diagnostics(self) -> dict[str, Any]:
        return {"items": list(self._diagnostics), "count": len(self._diagnostics)}

    def frozen_evaluation(self) -> dict[str, Any]:
        if not FROZEN_EVALUATION_PATH.exists():
            return {
                "status": "not_started",
                "evaluation_trading_days": CONFIG.evaluation_days,
                "live_provider_required": True,
            }
        return {"status": "frozen", **load_manifest(FROZEN_EVALUATION_PATH).__dict__}

    def freeze_evaluation(self) -> dict[str, Any]:
        provider = self.market_status()
        if not provider.get("is_live"):
            raise RuntimeError("A genuine live provider is required before freezing evaluation.")
        artifacts = self._get_artifacts()
        manifest = build_manifest(artifacts["metadata"], artifacts["classifier_name"])
        write_manifest(manifest, FROZEN_EVALUATION_PATH)
        return {"status": "frozen", **manifest.__dict__}

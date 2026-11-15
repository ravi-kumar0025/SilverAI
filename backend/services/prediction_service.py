"""Cached access to the SilverAI prediction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.config import DATA_MODE, FROZEN_EVALUATION_PATH, MODEL_ROOT, PREDICTION_DB_PATH, SELECTION_PATH
from src.data.live_provider import ConfiguredLiveProvider
from src.data.market_data import HistoricalCsvProvider, provider_status
from src.data.spot_data import align_market_data, load_candles
from src.prediction.store import PredictionStore, is_finalized_signal_record
from src.evaluation.paper_trading import evaluate_paper_trades, paper_cost_sensitivity
from src.models.spot_predict import load_spot_artifacts, predict_spot
from src.config import CONFIG
from src.spot_config import SPOT_CONFIG
from src.evaluation.frozen_mode import build_manifest, load_manifest, verify_manifest, write_manifest
from src.evaluation.frozen_mode import configuration_version
from src.market_insights import summarize_market


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
        self.market_provider = ConfiguredLiveProvider() if DATA_MODE == "live" else HistoricalCsvProvider()
        self._diagnostics: list[dict[str, Any]] = []

    def _get_artifacts(self) -> dict[str, Any]:
        if self._artifacts is None:
            logger.info("Loading selected SilverAI model artifacts")
            self._artifacts = {"kind": "spot", **load_spot_artifacts(self.model_root)}
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
            "classifier_name": "spot_lightgbm" if artifacts.get("kind") == "spot" else artifacts["classifier_name"],
            "regressor_name": None if artifacts.get("kind") == "spot" else artifacts["regressor_name"],
            "provider": self.market_status(),
            "prediction_db": str(self.store.path),
        }

    def ingest_live_candle(self, candle: dict[str, Any]) -> dict[str, Any]:
        """Accept one normalized closed candle from the Node bridge."""
        provider = self.market_provider
        ingest = getattr(provider, "ingest", None)
        if not callable(ingest):
            raise RuntimeError("The configured provider does not accept live candles.")
        return ingest(candle)

    def live_heartbeat(self) -> dict[str, Any]:
        heartbeat = getattr(self.market_provider, "heartbeat", None)
        if not callable(heartbeat):
            raise RuntimeError("The configured provider does not accept live heartbeats.")
        return heartbeat()

    def models(self) -> dict[str, Any]:
        artifacts = self._get_artifacts()
        metadata = artifacts.get("metadata", {})
        if artifacts.get("kind") == "spot":
            return {
                "classifier_name": "spot_lightgbm",
                "regressor_name": None,
                "feature_names": list(artifacts["features"]),
                "prediction_horizon_minutes": CONFIG.forecast_horizon_minutes,
                "prediction_horizon_bars": CONFIG.forecast_horizon_bars,
                "candidate_horizons_minutes": [SPOT_CONFIG.candle_minutes * bars for bars in SPOT_CONFIG.candidate_horizons_bars],
                "confidence_threshold": SPOT_CONFIG.confidence_threshold,
                "confidence_thresholds": list(SPOT_CONFIG.confidence_thresholds),
                "costs": {"spread_bps": SPOT_CONFIG.spread_bps, "slippage_bps": SPOT_CONFIG.slippage_bps},
                "holdout": {"start": SPOT_CONFIG.final_holdout_start, "end": SPOT_CONFIG.final_holdout_end, "status": SPOT_CONFIG.holdout_status},
                "classification_target": "binary_atr_scaled_direction",
                "regression_target": None,
                "configuration": CONFIG.as_dict(),
                "configuration_version": CONFIG.version,
                "calibration_status": "available" if artifacts.get("calibrator") is not None else "unavailable",
                "feature_count": len(artifacts["features"]),
                "model_status": "research_artifact_not_promoted",
                "dataset_id": artifacts.get("dataset_id"),
                "model_version": artifacts.get("model_version"),
            }
        raise RuntimeError("Only the final XAG/USD spot artifact is supported by this service.")

    def predict(self, observations: list[dict[str, Any]], *, finalized: bool = False) -> dict[str, Any]:
        frame = pd.DataFrame(observations)
        artifacts = self._get_artifacts()
        if artifacts.get("kind") == "spot":
            prediction_frame = predict_spot(frame, artifacts)
            predictions = []
            for _, row in prediction_frame.iterrows():
                p_up = float(row["probability_up"])
                calibrated_up = float(row["calibrated_probability_up"])
                confidence = max(calibrated_up, 1 - calibrated_up)
                signal = "UP" if calibrated_up >= 0.5 else "DOWN"
                if confidence < float(row["confidence_threshold"]):
                    signal = "NO_CLEAR_SIGNAL"
                prediction = {
                    "prediction_timestamp": row.get("timestamp"),
                    "forecast_horizon_minutes": int(row["prediction_horizon_minutes"]),
                    "predicted_direction": int(row["predicted_direction"]),
                    "predicted_direction_name": row["predicted_direction_name"],
                    "class_probabilities": {"down": 1 - p_up, "up": p_up},
                    "calibrated_probabilities": {"down": 1 - calibrated_up, "up": calibrated_up},
                    "confidence": confidence,
                    "signal": {"signal": signal},
                    "trade_decision": "ACTIONABLE_LONG" if signal == "UP" else "ACTIONABLE_SHORT" if signal == "DOWN" else "WAIT",
                    "decision_reason": "Confidence threshold and closed-candle gates are applied to the binary spot model.",
                    "model_metadata": {"classifier_name": "spot_lightgbm", "regressor_name": None, "feature_count": len(artifacts["features"]), "feature_names": list(artifacts["features"]), "calibration_status": "available" if artifacts.get("calibrator") is not None else "unavailable", "dataset_id": artifacts.get("dataset_id"), "model_version": artifacts.get("model_version")},
                    "warnings": ["Historical Dukascopy data; not a live signal."] if not finalized else [],
                    "candle_finalized": finalized,
                }
                predictions.append(prediction)
            if finalized:
                for observation, prediction in zip(observations, predictions):
                    try:
                        self.store.record({
                            **prediction,
                            "instrument": CONFIG.instrument,
                            "contract": CONFIG.symbol,
                            "features": observation,
                            "candle_finalized": True,
                            "close_price": observation.get("close"),
                        })
                    except (TypeError, ValueError, OSError) as exc:
                        self._diagnostics.append({"kind": "persistence_failure", "error": str(exc)})
            result = {"batch": len(predictions) > 1, "predictions": predictions}
            if len(predictions) == 1:
                result.update(predictions[0])
            self._latest = predictions[-1]
            return result
        raise RuntimeError("The final spot prediction path did not return a result.")

    def latest(self) -> dict[str, Any] | None:
        history = self.store.list_finalized_signals(limit=1)
        return history[0] if history else None

    def prediction_history(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list(status=status, limit=limit)

    def evaluate_predictions(self, candles: pd.DataFrame) -> int:
        return self.store.evaluate(candles)

    def market_status(self) -> dict[str, Any]:
        return provider_status(self.market_provider)

    def paper_trading(self) -> dict[str, Any]:
        rows = self.store.list(status="evaluated")
        result = evaluate_paper_trades(rows)
        result["historical_simulation"] = True
        result["cost_sensitivity"] = [
            {key: value for key, value in scenario.items() if key != "trades"}
            for scenario in paper_cost_sensitivity(rows)
        ]
        return result

    def evaluation_metrics(self) -> dict[str, Any]:
        rows = self.store.list(status="evaluated", limit=100000)
        completed = len(rows)
        correct = sum(row.get("predicted_direction") == row.get("actual_direction") for row in rows)
        result = {
            "predictions_total": len(self.store.list(limit=100000)),
            "completed_outcomes": completed,
            "pending_outcomes": len(self.store.list(status="pending", limit=100000)),
            "directional_accuracy": (correct / completed) if completed else None,
            "coverage": "historical replay only",
        }
        report_root = Path(__file__).resolve().parents[2] / "reports" / "spot_dukascopy_20261115"
        spot_metrics = report_root / "metrics.json"
        if spot_metrics.is_file():
            result["spot_report"] = {"path": "reports/spot_dukascopy_20261115/metrics.json", "status": "available"}
        if "classification_metrics" not in result or "regression_metrics" not in result:
            try:
                metadata = self._get_artifacts().get("metadata", {})
                if "classification_metrics" not in result:
                    result["classification_metrics"] = [
                        {"dataset": "validation", **row} for row in metadata.get("classification_comparison", [])
                    ]
                if "regression_metrics" not in result:
                    result["regression_metrics"] = [
                        {"dataset": "validation", **row} for row in metadata.get("regression_comparison", [])
                    ]
                result["report_source"] = "saved training metadata fallback"
            except (FileNotFoundError, OSError, ValueError):
                result["report_source"] = "unavailable"
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {"items": list(self._diagnostics), "count": len(self._diagnostics)}

    def market_insights(self) -> dict[str, Any]:
        # The checked-in source is the only verified historical frame. A live
        # adapter can replace this method when its candle contract is verified.
        frame, quality = load_candles("xagusd", CONFIG.candle_minutes)
        frame = frame.rename(columns={"timestamp": "datetime_et"})
        latest = self.latest()
        result = summarize_market(frame, latest)
        result["data_quality"] = quality
        result["instrument"] = CONFIG.instrument
        result["symbol"] = CONFIG.symbol
        result["timezone"] = CONFIG.timezone
        result["source"] = "Dukascopy"
        try:
            _, alignment = align_market_data(CONFIG.candle_minutes)
            result["cross_market"] = {
                "instruments": ["XAG/USD", "XAU/USD", "EUR/USD"],
                "complete_rows": alignment["aligned"]["cross_market_complete_rows"],
                "coverage_ratio": alignment["aligned"]["coverage_ratio"],
                "forward_fill_limit_candles": 2,
            }
        except (FileNotFoundError, OSError, ValueError) as exc:
            result["cross_market"] = {"status": "unavailable", "reason": type(exc).__name__}
        return result

    def frozen_evaluation(self) -> dict[str, Any]:
        if not FROZEN_EVALUATION_PATH.exists():
            return {
                "status": "not_started",
                "evaluation_trading_days": CONFIG.evaluation_days,
                "live_provider_required": True,
            }
        manifest = load_manifest(FROZEN_EVALUATION_PATH)
        integrity = verify_manifest(manifest)
        return {"status": "frozen" if integrity["valid"] else "invalid", **manifest.__dict__, "integrity": integrity}

    def freeze_evaluation(self) -> dict[str, Any]:
        provider = self.market_status()
        if not provider.get("is_live"):
            raise RuntimeError("A genuine live provider is required before freezing evaluation.")
        if FROZEN_EVALUATION_PATH.exists():
            manifest = load_manifest(FROZEN_EVALUATION_PATH)
            integrity = verify_manifest(manifest)
            if not integrity["valid"]:
                raise RuntimeError("Existing frozen evaluation manifest failed integrity verification.")
            return {"status": "frozen", **manifest.__dict__, "integrity": integrity, "idempotent": True}
        artifacts = self._get_artifacts()
        manifest = build_manifest(
            artifacts["metadata"],
            artifacts["classifier_name"],
            artifact_paths={
                "classifier": artifacts.get("classifier_path"),
                "regressor": artifacts.get("regressor_path"),
                "calibration": artifacts.get("calibration_path"),
            },
            data_metadata={"status": "repository_artifact", "source": "silver_modeling_data.csv"},
        )
        write_manifest(manifest, FROZEN_EVALUATION_PATH)
        return {"status": "frozen", **manifest.__dict__, "integrity": verify_manifest(manifest), "idempotent": False}

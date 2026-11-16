"""Cached access to the SilverAI prediction pipeline."""

from __future__ import annotations

import logging
import csv
import ast
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.config import DATA_MODE, FROZEN_EVALUATION_PATH, MODEL_ROOT, PREDICTION_DB_PATH, SELECTION_PATH
from src.data.live_market_provider import ConfiguredLiveProvider
from src.data.market_data_contracts import HistoricalCsvProvider, provider_status
from src.data.market_candle_builder import align_market_data, load_candles
from src.data.live_log_market_source import LiveLogSource
from src.features.market_features import generate_spot_features
from src.prediction.store import PredictionStore, is_finalized_signal_record
from src.evaluation.paper_trading import evaluate_paper_trades, paper_cost_sensitivity
from src.models.multiclass_prediction import load_spot_artifacts, predict_spot
from src.project_config import CONFIG
from src.market_model_config import SPOT_CONFIG
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
        self._live_scored: set[str] = set()
        self._historical_frames: dict[str, pd.DataFrame] | None = None
        self._live_logs = LiveLogSource()
        self.market_provider = ConfiguredLiveProvider(on_candle=self._on_live_candle) if DATA_MODE == "live" else HistoricalCsvProvider()
        self._diagnostics: list[dict[str, Any]] = []
        self._market_insights_cache: dict[str, Any] | None = None
        self._market_insights_cache_at = 0.0

    def _get_artifacts(self) -> dict[str, Any]:
        if self._artifacts is None:
            logger.info("Loading selected SilverAI model artifacts")
            self._artifacts = {"kind": "spot", **load_spot_artifacts(self.model_root)}
        return self._artifacts

    def _active_model_version(self) -> str | None:
        # A few lightweight store-restart tests construct the service without
        # invoking __init__; preserve their ability to inspect a standalone DB.
        if not hasattr(self, "_artifacts"):
            return None
        artifacts = self._get_artifacts()
        metadata = artifacts.get("metadata", {})
        return str(artifacts.get("model_version") or metadata.get("model_version") or "unknown")

    def _on_live_candle(self, candle: dict[str, Any]) -> None:
        """Score each newly completed 5m live bar once, using all companions."""
        if candle.get("symbol") not in {"xagusd", "xauusd", "eurusd"}:
            return
        source = self._live_logs
        frames = {}
        if self._historical_frames is None:
            self._historical_frames = {symbol: load_candles(symbol, 5)[0] for symbol in ("xagusd", "xauusd", "eurusd")}
        for symbol in ("xagusd", "xauusd", "eurusd"):
            historical = self._historical_frames[symbol]
            live = source.candles(symbol, 5)
            parts = [historical] if live.empty else [historical, live]
            frames[symbol] = pd.concat(parts, ignore_index=True).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        if frames["xagusd"].empty:
            return
        # Settle eligible forecasts from the silver path even if a companion
        # is temporarily missing; companion data is required for new scoring,
        # not for evaluating an already-persisted forecast.
        outcome_candles = frames["xagusd"].rename(columns={"timestamp": "datetime_et"})[["datetime_et", "close"]]
        self.store.evaluate(outcome_candles)
        if any(frame.empty for frame in frames.values()):
            return
        target = frames["xagusd"].set_index("timestamp").rename(columns=lambda name: f"silver_{name}")
        for symbol in ("xauusd", "eurusd"):
            companion = frames[symbol].set_index("timestamp").rename(columns=lambda name: f"{symbol}_{name}")
            target = target.join(companion, how="left")
        target = target.reset_index()
        features = generate_spot_features(target, SPOT_CONFIG.feature_groups)
        latest = features.tail(1)
        timestamp = str(latest.iloc[0]["timestamp"])
        if timestamp in self._live_scored:
            return
        try:
            self.predict(latest.to_dict(orient="records"), finalized=True)
            self._live_scored.add(timestamp)
            self._market_insights_cache = None
            self._market_insights_cache_at = 0.0
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            item = {"kind": "live_prediction_failure", "candle_timestamp": timestamp, "error": str(exc)}
            if item not in self._diagnostics:
                self._diagnostics.append(item)

    def poll_live_logs(self) -> None:
        """Tail stored live logs so forecasts do not depend on POST delivery."""
        if DATA_MODE != "live":
            return
        # A process restart must not turn the last historical row in the
        # append-only live log into a new forecast.  The log tail remains
        # useful for status and chart backfill, but scoring is allowed only
        # while the source is inside the configured freshness window.
        if not self._live_logs.status().get("fresh", False):
            return
        latest = self._live_logs.candles("xagusd", 5)
        if latest.empty:
            return
        self._on_live_candle({"symbol": "xagusd", "timestamp": str(latest.iloc[-1]["timestamp"])})

    def health(self) -> dict[str, Any]:
        try:
            artifacts = self._get_artifacts()
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("Model health check failed: %s", type(exc).__name__)
            return {"status": "degraded", "models_available": False}
        return {
            "status": "ok",
            "models_available": True,
            "classifier_name": artifacts.get("metadata", {}).get("model_version", "spot_lightgbm"),
            "regressor_name": "spot_return_lgbm" if artifacts.get("regression") else None,
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

    def live_heartbeat(self, *, connected: bool = True) -> dict[str, Any]:
        heartbeat = getattr(self.market_provider, "heartbeat", None)
        if not callable(heartbeat):
            raise RuntimeError("The configured provider does not accept live heartbeats.")
        return heartbeat(connected=connected)

    def models(self) -> dict[str, Any]:
        artifacts = self._get_artifacts()
        metadata = artifacts.get("metadata", {})
        if artifacts.get("kind") == "spot":
            return {
                "classifier_name": "spot_lightgbm",
                "regressor_name": "spot_return_lgbm" if artifacts.get("regression") else None,
                "feature_names": list(artifacts["features"]),
                "prediction_horizon_minutes": CONFIG.forecast_horizon_minutes,
                "prediction_horizon_bars": CONFIG.forecast_horizon_bars,
                "candidate_horizons_minutes": [SPOT_CONFIG.candle_minutes * bars for bars in SPOT_CONFIG.candidate_horizons_bars],
                "confidence_threshold": SPOT_CONFIG.confidence_threshold,
                "confidence_thresholds": list(SPOT_CONFIG.confidence_thresholds),
                "costs": {"spread_bps": SPOT_CONFIG.spread_bps, "slippage_bps": SPOT_CONFIG.slippage_bps},
                "holdout": {"start": SPOT_CONFIG.final_holdout_start, "end": SPOT_CONFIG.final_holdout_end, "status": SPOT_CONFIG.holdout_status},
                "classification_target": "binary_atr_scaled_direction",
                "regression_target": "forward_log_return_12_bars" if artifacts.get("regression") else None,
                "configuration": CONFIG.as_dict(),
                "configuration_version": CONFIG.version,
                "calibration_status": "available" if artifacts.get("calibrator") is not None else "unavailable",
                "feature_count": len(artifacts["features"]),
                "model_status": "research_artifact_not_promoted",
                "dataset_id": artifacts.get("dataset_id"),
                "model_version": artifacts.get("model_version"),
            }
        if artifacts.get("kind") == "spot_15m_multiclass":
            horizon_minutes = int(metadata.get("prediction_horizon_minutes", 15))
            horizon_bars = int(metadata.get("horizon_bars", 3))
            threshold = float(metadata.get("selected_neutral_confidence_threshold", 0.4))
            regression_target = (artifacts.get("regression") or {}).get("mean", {}).get("metadata", {}).get("target")
            configuration = {
                **CONFIG.as_dict(),
                "forecast_horizon_bars": horizon_bars,
                "forecast_horizon_minutes": horizon_minutes,
                "classification_target": "15-minute_down_neutral_up",
                "regression_target": regression_target,
                "threshold_definition": "Calibrated top-class confidence with validation-selected neutral emission threshold.",
            }
            configuration_version = metadata.get(
                "configuration_version",
                f"{metadata.get('model_version', 'spot_15m_multiclass')}:{horizon_minutes}m:{len(artifacts['features'])}features",
            )
            return {
                "classifier_name": metadata.get("model_version", "xagusd_5m_15m_multiclass_v1"),
                "regressor_name": "xagusd_5m_15m_return_regressor_v1" if artifacts.get("regression") else None,
                "feature_names": list(artifacts["features"]),
                "prediction_horizon_minutes": horizon_minutes,
                "prediction_horizon_bars": horizon_bars,
                "input_candle_minutes": SPOT_CONFIG.candle_minutes,
                "candidate_horizons_minutes": [15],
                "confidence_threshold": threshold,
                "neutral_confidence_threshold": threshold,
                "selected_atr_k": metadata.get("selected_atr_k"),
                "costs": {"spread_bps": SPOT_CONFIG.spread_bps, "slippage_bps": SPOT_CONFIG.slippage_bps},
                "holdout": {"start": metadata.get("test_start"), "end": metadata.get("test_end"), "status": "untouched_test_block"},
                "classification_target": "15-minute_down_neutral_up",
                "regression_target": regression_target,
                "configuration": configuration,
                "configuration_version": configuration_version,
                "calibration_status": "available" if artifacts.get("calibrator") is not None else "unavailable",
                "feature_count": len(artifacts["features"]),
                "model_status": metadata.get("status", "research_artifact_not_promoted"),
                "dataset_id": artifacts.get("dataset_id"),
                "model_version": artifacts.get("model_version"),
                "label_mapping": metadata.get("label_mapping", {"0": "down", "1": "neutral", "2": "up"}),
            }
        raise RuntimeError("Only the final XAG/USD spot artifact is supported by this service.")

    def predict(self, observations: list[dict[str, Any]], *, finalized: bool = False) -> dict[str, Any]:
        frame = pd.DataFrame(observations)
        artifacts = self._get_artifacts()
        if artifacts.get("kind") in {"spot", "spot_15m_multiclass"}:
            prediction_frame = predict_spot(frame, artifacts)
            predictions = []
            for _, row in prediction_frame.iterrows():
                if artifacts.get("kind") == "spot_15m_multiclass":
                    probabilities = {name: float(row[f"probability_{name}"]) for name in ("down", "neutral", "up")}
                    calibrated = {name: float(row[f"calibrated_probability_{name}"]) for name in ("down", "neutral", "up")}
                    signal = str(row["predicted_direction_name"])
                    regression_target = (artifacts.get("regression") or {}).get("mean", {}).get("metadata", {}).get("target")
                    configuration_version = (artifacts.get("metadata") or {}).get(
                        "configuration_version",
                        f"{artifacts.get('model_version', 'spot_15m_multiclass')}:{int(row['prediction_horizon_minutes'])}m:{len(artifacts['features'])}features",
                    )
                    prediction = {
                        "prediction_timestamp": row.get("timestamp"),
                        "forecast_horizon_minutes": int(row["prediction_horizon_minutes"]),
                        "forecast_horizon_bars": int(row["prediction_horizon_bars"]),
                        "predicted_direction": int(row["predicted_direction"]),
                        "predicted_direction_name": signal,
                        "class_probabilities": probabilities,
                        "calibrated_probabilities": calibrated,
                        "confidence": float(row["confidence"]),
                        "confidence_kind": "calibrated_top_class_probability",
                        "signal": {"signal": signal},
                        "trade_decision": "ACTIONABLE_LONG" if signal == "UP" else "ACTIONABLE_SHORT" if signal == "DOWN" else "WAIT",
                        "decision_reason": "Calibrated 15-minute three-class model with validation-selected neutral threshold.",
                        "model_version": artifacts.get("model_version"),
                        "configuration_version": configuration_version,
                        "model_metadata": {"classifier_name": artifacts.get("model_version"), "regressor_name": "xagusd_5m_15m_return_regressor_v1" if artifacts.get("regression") else None, "feature_count": len(artifacts["features"]), "feature_names": list(artifacts["features"]), "calibration_status": "available" if artifacts.get("calibrator") is not None else "unavailable", "dataset_id": artifacts.get("dataset_id"), "model_version": artifacts.get("model_version"), "selected_atr_k": (artifacts.get("metadata") or {}).get("selected_atr_k"), "neutral_confidence_threshold": float(row["confidence_threshold"]), "regression_target": regression_target, "configuration_version": configuration_version},
                        "warnings": ["Historical Dukascopy data; not a live signal."] if not finalized else [],
                        "candle_finalized": finalized,
                    }
                    if "predicted_return" in row:
                        prediction.update({
                            "predicted_return": float(row["predicted_return"]),
                            "predicted_price": float(row["predicted_price"]),
                            "price_low": float(row["price_low"]),
                            "price_high": float(row["price_high"]),
                            "expected_direction": row["regression_direction"],
                        })
                    predictions.append(prediction)
                    continue
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
                    "model_metadata": {"classifier_name": "spot_lightgbm", "regressor_name": "spot_return_lgbm" if artifacts.get("regression") else None, "feature_count": len(artifacts["features"]), "feature_names": list(artifacts["features"]), "calibration_status": "available" if artifacts.get("calibrator") is not None else "unavailable", "dataset_id": artifacts.get("dataset_id"), "model_version": artifacts.get("model_version")},
                    "warnings": ["Historical Dukascopy data; not a live signal."] if not finalized else [],
                    "candle_finalized": finalized,
                }
                if "predicted_return" in row:
                    prediction.update({"predicted_return": float(row["predicted_return"]), "predicted_price": float(row["predicted_price"]), "price_low": float(row["price_low"]), "price_high": float(row["price_high"]), "expected_direction": row["regression_direction"], "model_metadata": {**prediction["model_metadata"], "regressor_name": "spot_return_lgbm", "regression_target": "forward_log_return_12_bars"}})
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
                            "close_price": observation.get("close", observation.get("silver_close")),
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
        history = self.store.list_finalized_signals(
            limit=1,
            model_version=self._active_model_version(),
        )
        if not history:
            return None
        row = history[0]
        row["class_probabilities"] = {"down": row.get("probability_down"), "neutral": row.get("probability_neutral"), "up": row.get("probability_up")}
        row["calibrated_probabilities"] = {"down": row.get("calibrated_probability_down"), "neutral": row.get("calibrated_probability_neutral"), "up": row.get("calibrated_probability_up")}
        row["signal"] = {"signal": row.get("final_signal")}
        row["prediction_timestamp"] = row.get("candle_timestamp")
        row["forecast_horizon_minutes"] = row.get("horizon_minutes")
        row["forecast_horizon_bars"] = row.get("horizon_bars")
        row["confidence"] = row.get("confidence")
        return row

    def prediction_history(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list(
            status=status,
            limit=limit,
            model_version=self._active_model_version(),
        )

    def evaluate_predictions(self, candles: pd.DataFrame) -> int:
        return self.store.evaluate(candles)

    def market_status(self) -> dict[str, Any]:
        status = provider_status(self.market_provider)
        if DATA_MODE == "live":
            log_status = self._live_logs.status()
            provider_update = pd.to_datetime(status.get("last_update"), utc=True, errors="coerce")
            log_update = pd.to_datetime(log_status.get("last_update"), utc=True, errors="coerce")
            if pd.notna(log_update) and (pd.isna(provider_update) or log_update >= provider_update):
                status = {**status, **log_status, "provider": "dukascopy-live-log", "source_status": {"receiver": status, "log_tail": log_status}}
        return status

    def paper_trading(self) -> dict[str, Any]:
        rows = self.store.list(
            status="evaluated",
            model_version=self._active_model_version(),
        )
        result = evaluate_paper_trades(rows)
        result["historical_simulation"] = True
        result["cost_sensitivity"] = [
            {key: value for key, value in scenario.items() if key != "trades"}
            for scenario in paper_cost_sensitivity(rows)
        ]
        return result

    def evaluation_metrics(self) -> dict[str, Any]:
        active_model_version = self._active_model_version()
        rows = self.store.list(
            status="evaluated",
            limit=100000,
            model_version=active_model_version,
        )
        completed = len(rows)
        correct = sum(row.get("predicted_direction") == row.get("actual_direction") for row in rows)
        result = {
            "predictions_total": len(self.store.list(limit=100000, model_version=active_model_version)),
            "completed_outcomes": completed,
            "pending_outcomes": len(self.store.list(status="pending", limit=100000, model_version=active_model_version)),
            "directional_accuracy": (correct / completed) if completed else None,
            "coverage": "active model records only",
            "model_version": active_model_version,
        }
        artifacts = self._get_artifacts()
        if artifacts.get("kind") == "spot_15m_multiclass":
            selection_path = self.model_root / "selection.csv"
            try:
                selected_k = float((artifacts.get("metadata") or {}).get("selected_atr_k", 0.1))
                with selection_path.open(newline="", encoding="utf-8") as handle:
                    selected = next(
                        row for row in csv.DictReader(handle)
                        if abs(float(row["k"]) - selected_k) < 1e-9
                    )
                validation = ast.literal_eval(selected["validation"])
                result["classification_metrics"] = [{
                    "dataset": "validation_selected_configuration",
                    "model": active_model_version,
                    "accuracy": validation.get("accuracy"),
                    "balanced_accuracy": validation.get("balanced_accuracy"),
                    "macro_f1": validation.get("macro_f1"),
                    "signal_frequency": None,
                    "neutral_coverage": None,
                }]
                result["active_report"] = {
                    "path": "reports/spot_15m_20261116_multiclass/selection.csv",
                    "status": "available",
                }
                regression_report = (artifacts.get("metadata") or {}).get("regression") or {}
                regression_rows = []
                for dataset_name, metrics in (("validation", regression_report.get("validation")), ("untouched_test", regression_report.get("test"))):
                    if metrics:
                        regression_rows.append({
                            "dataset": dataset_name,
                            "model": "xagusd_5m_15m_return_regressor_v1",
                            "mae": metrics.get("mae"),
                            "rmse": metrics.get("rmse"),
                            "r2": metrics.get("r2_vs_zero"),
                        })
                result["regression_metrics"] = regression_rows
            except (OSError, ValueError, StopIteration, SyntaxError):
                result["classification_metrics"] = []
        result["report_source"] = "active model validation report"
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {"items": list(self._diagnostics), "count": len(self._diagnostics)}

    def market_insights(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._market_insights_cache is not None and now - self._market_insights_cache_at < 30:
            return self._market_insights_cache
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
        self._market_insights_cache = result
        self._market_insights_cache_at = now
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

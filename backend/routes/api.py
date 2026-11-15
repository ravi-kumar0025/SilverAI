"""SilverAI prediction API routes."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, create_model
from src.data.spot_data import build_candles, load_candles, load_raw_instrument
from src.config import CONFIG
from src.features.spot_features import SPOT_FEATURE_COLUMNS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@lru_cache(maxsize=1)
def _market_frame():
    """Load complete UTC XAG/USD Dukascopy candles once per process."""
    frame, _ = load_candles("xagusd", CONFIG.candle_minutes)
    frame = frame.rename(columns={"timestamp": "datetime_et"})
    return frame.sort_values("datetime_et").reset_index(drop=True)


FeatureObservation = create_model(
    "FeatureObservation",
    __config__=ConfigDict(extra="forbid"),
    timestamp=(str | None, None),
    known_at=(str | None, None),
    **{feature: (float, ...) for feature in SPOT_FEATURE_COLUMNS},
)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[FeatureObservation] = Field(min_length=1)


class ConfirmedCandlePrediction(BaseModel):
    """One closed candle plus its causally generated feature observation."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    close: float
    features: FeatureObservation


def _service(request: Request):
    return request.app.state.prediction_service


def _prediction_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (FileNotFoundError, OSError)):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction models are unavailable.",
        )
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prediction input or model metadata is invalid.",
        )
    logger.exception("Unexpected prediction failure")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Prediction failed.",
    )


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    return _service(request).health()


@router.get("/models")
def models(request: Request) -> dict[str, Any]:
    try:
        return _service(request).models()
    except Exception as exc:
        raise _prediction_error(exc) from exc


@router.post("/predict")
def predict(payload: FeatureObservation, request: Request) -> dict[str, Any]:
    try:
        result = _service(request).predict([payload.model_dump(exclude_none=True)])
        return result
    except Exception as exc:
        raise _prediction_error(exc) from exc


@router.post("/predict/batch")
def predict_batch(payload: BatchPredictionRequest, request: Request) -> dict[str, Any]:
    try:
        observations = [item.model_dump(exclude_none=True) for item in payload.observations]
        return _service(request).predict(observations)
    except Exception as exc:
        raise _prediction_error(exc) from exc


@router.post("/predict/confirmed")
def predict_confirmed(payload: ConfirmedCandlePrediction, request: Request) -> dict[str, Any]:
    """Infer once for a closed candle and persist its close for later evaluation."""
    try:
        observation = payload.features.model_dump(exclude_none=True)
        observation["timestamp"] = payload.timestamp
        observation["close"] = payload.close
        return _service(request).predict([observation], finalized=True)
    except Exception as exc:
        raise _prediction_error(exc) from exc


@router.get("/predictions/latest")
def latest(request: Request) -> dict[str, Any]:
    result = _service(request).latest()
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No prediction available.")
    return result


@router.get("/market/candles")
def market_candles(limit: int = 288, timeframe_minutes: int = CONFIG.candle_minutes) -> dict[str, Any]:
    """Return real historical OHLCV observations, optionally resampled."""
    if limit < 24 or limit > 2500:
        raise HTTPException(status_code=422, detail="limit must be between 24 and 2500.")
    if timeframe_minutes not in {5, 10, 15, 30}:
        raise HTTPException(status_code=422, detail="timeframe_minutes must be one of 5, 10, 15, or 30.")
    try:
        source_frame = _market_frame()
        if timeframe_minutes == 5:
            frame = source_frame
        else:
            raw, _ = load_raw_instrument("xagusd")
            frame = build_candles(raw, timeframe_minutes).rename(columns={"timestamp": "datetime_et"})
        frame = frame.tail(limit)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Historical market data is unavailable.") from exc
    candles = [
        {
            "time": row.datetime_et.isoformat(),
            "timestamp": row.datetime_et.isoformat(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for row in frame.itertuples(index=False)
    ]
    last = candles[-1]
    previous = candles[-2] if len(candles) > 1 else last
    session = frame[frame.datetime_et.dt.date == frame.iloc[-1].datetime_et.date()]
    return {
        "instrument": CONFIG.instrument,
        "symbol": CONFIG.symbol,
        "source": "Dukascopy xagusd_1m.csv" if timeframe_minutes == 5 else f"Dukascopy xagusd_1m.csv resampled to {timeframe_minutes}m",
        "timeframe_minutes": timeframe_minutes,
        "data_status": "historical",
        "timezone": "UTC",
        "timestamp": last["time"],
        "candles": candles,
        "stats": {
            "open": candles[0]["open"],
            "high": float(session.high.max()),
            "low": float(session.low.min()),
            "close": last["close"],
            "previous_close": previous["close"],
            "change": last["close"] - previous["close"],
            "change_pct": (last["close"] / previous["close"] - 1) if previous["close"] else 0,
        },
    }


@router.get("/market/status")
def market_status(request: Request) -> dict[str, Any]:
    """Report the configured data source without implying that historical data is live."""
    return _service(request).market_status()


@router.get("/market/insights")
def market_insights(request: Request) -> dict[str, Any]:
    try:
        return _service(request).market_insights()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Market insights are unavailable.") from exc


@router.get("/provider/health")
def provider_health(request: Request) -> dict[str, Any]:
    return _service(request).market_status()


@router.get("/predictions")
def prediction_history(request: Request, outcome_status: str | None = None, limit: int = 100) -> dict[str, Any]:
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000.")
    return {"predictions": _service(request).prediction_history(status=outcome_status, limit=limit)}


@router.get("/paper-trading")
def paper_trading(request: Request) -> dict[str, Any]:
    return _service(request).paper_trading()


@router.post("/predictions/evaluate")
def evaluate_predictions(request: Request) -> dict[str, Any]:
    """Replay the configured historical candles to settle eligible outcomes."""
    try:
        frame = _market_frame()
        return {"evaluated": _service(request).evaluate_predictions(frame)}
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Market data is unavailable for evaluation.") from exc


@router.get("/evaluation/metrics")
def evaluation_metrics(request: Request) -> dict[str, Any]:
    return _service(request).evaluation_metrics()


@router.get("/diagnostics")
def diagnostics(request: Request) -> dict[str, Any]:
    return _service(request).diagnostics()


@router.get("/evaluation/frozen")
def frozen_evaluation(request: Request) -> dict[str, Any]:
    return _service(request).frozen_evaluation()


@router.post("/evaluation/frozen")
def freeze_evaluation(request: Request) -> dict[str, Any]:
    try:
        return _service(request).freeze_evaluation()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

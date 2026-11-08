"""SilverAI prediction API routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class FeatureObservation(BaseModel):
    """One engineered observation accepted by the prediction pipeline."""

    model_config = ConfigDict(extra="forbid")

    datetime_et: str | None = None
    return_1: float
    return_3: float
    return_6: float
    volatility_5: float
    volatility_20: float
    volatility_50: float
    sma_distance_20: float
    sma_distance_50: float
    candle_return: float
    candle_range_pct: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    relative_volume: float
    volume_change: float
    hour_sin: float
    hour_cos: float
    day_of_week: float


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[FeatureObservation] = Field(min_length=1)


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


@router.get("/predictions/latest")
def latest(request: Request) -> dict[str, Any]:
    result = _service(request).latest()
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No prediction available.")
    return result

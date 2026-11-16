"""Endpoint tests for the SilverAI backend contract."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.server import app
from backend.routes.prediction_api_routes import FeatureObservation
from src.features.market_features import SPOT_FEATURE_COLUMNS


FEATURES = {
    "timestamp": "2026-01-01T09:30:00+00:00",
    **{feature: 0.01 for feature in SPOT_FEATURE_COLUMNS},
}


class FakePredictionService:
    def __init__(self) -> None:
        self.last = None

    def health(self):
        return {"status": "ok", "models_available": True}

    def models(self):
        return {
            "classifier_name": "fake_classifier",
            "regressor_name": "fake_regressor",
            "feature_names": list(FEATURES),
            "prediction_horizon_minutes": 15,
        }

    def predict(self, observations, *, finalized=False):
        self.last = {"batch": len(observations) > 1, "predictions": observations, "finalized": finalized}
        return self.last

    def latest(self):
        return self.last

    def market_status(self):
        return {"is_live": False, "fresh": False, "last_update": None}

    def prediction_history(self, status=None, limit=100):
        return [{"prediction_id": "final-1", "final_signal": "BULLISH", "candle_timestamp": "2026-01-01T09:30:00-05:00", "candle_finalized": 1}]


def test_backend_endpoints_and_latest(monkeypatch):
    fake = FakePredictionService()
    monkeypatch.setattr(app.state, "prediction_pipeline_service", fake)
    client = TestClient(app)

    assert client.get("/api/health").json()["models_available"] is True
    assert client.get("/api/models").status_code == 200
    empty_latest = client.get("/api/predictions/latest")
    assert empty_latest.status_code == 200
    assert empty_latest.json()["direction"] == "No clear signal"
    assert empty_latest.json()["reason"] == "no_prediction_yet"

    single = client.post("/api/predict", json=FEATURES)
    assert single.status_code == 200
    assert single.json()["batch"] is False

    batch = client.post(
        "/api/predict/batch",
        json={"observations": [FEATURES, FEATURES]},
    )
    assert batch.status_code == 200
    assert batch.json()["batch"] is True
    assert client.get("/api/predictions/latest").status_code == 200


def test_backend_rejects_missing_and_extra_features(monkeypatch):
    monkeypatch.setattr(app.state, "prediction_pipeline_service", FakePredictionService())
    client = TestClient(app)

    missing = dict(FEATURES)
    missing.pop(SPOT_FEATURE_COLUMNS[0])
    assert client.post("/api/predict", json=missing).status_code == 422

    extra = dict(FEATURES, unexpected=1.0)
    assert client.post("/api/predict", json=extra).status_code == 422


def test_latest_reports_stale_reason_without_changing_status_code(monkeypatch):
    class StaleService(FakePredictionService):
        def market_status(self):
            return {"is_live": True, "fresh": False, "last_update": "2026-01-01T10:00:00+00:00"}

    monkeypatch.setattr(app.state, "prediction_pipeline_service", StaleService())
    response = TestClient(app).get("/api/predictions/latest")
    assert response.status_code == 200
    assert response.json()["direction"] == "No clear signal"
    assert response.json()["reason"] == "stale_data"


def test_prediction_history_returns_persisted_final_signal_fields(monkeypatch):
    monkeypatch.setattr(app.state, "prediction_pipeline_service", FakePredictionService())
    response = TestClient(app).get("/api/predictions?limit=10")
    assert response.status_code == 200
    assert response.json()["predictions"][0]["final_signal"] == "BULLISH"
    assert response.json()["predictions"][0]["candle_finalized"] == 1


def test_backend_cors_and_schema_contract(monkeypatch):
    monkeypatch.setattr(app.state, "prediction_pipeline_service", FakePredictionService())
    client = TestClient(app)
    response = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert set(FeatureObservation.model_fields) == {"timestamp", "known_at", *SPOT_FEATURE_COLUMNS}


def test_backend_hides_model_loading_failure(monkeypatch):
    class FailingService(FakePredictionService):
        def predict(self, observations):
            raise FileNotFoundError("private model path")

    monkeypatch.setattr(app.state, "prediction_pipeline_service", FailingService())
    response = TestClient(app).post("/api/predict", json=FEATURES)
    assert response.status_code == 503
    assert response.json()["detail"] == "Prediction models are unavailable."
    assert "private model path" not in response.text


def test_confirmed_candle_endpoint_preserves_closed_candle_boundary(monkeypatch):
    fake = FakePredictionService()
    monkeypatch.setattr(app.state, "prediction_pipeline_service", fake)
    payload = {
        "timestamp": "2026-01-01T09:30:00+00:00",
        "close": 50.0,
        "features": FEATURES,
    }
    response = TestClient(app).post("/api/predict/confirmed", json=payload)
    assert response.status_code == 200
    assert fake.last["batch"] is False
    assert fake.last["predictions"][0]["close"] == 50.0
    assert fake.last["finalized"] is True

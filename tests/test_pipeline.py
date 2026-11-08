import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.prediction.pipeline as pipeline
from src.features.technical_indicators import FEATURE_COLUMNS


def _features(rows=1):
    frame = pd.DataFrame(
        {name: np.arange(rows, dtype=float) + 1 for name in FEATURE_COLUMNS}
    )
    frame["datetime_et"] = pd.date_range(
        "2026-01-01", periods=rows, freq="5min", tz="America/New_York"
    )
    return frame


def _artifacts():
    return {
        "classifier_name": "classifier",
        "regressor_name": "regressor",
        "metadata": {
            "feature_names": list(FEATURE_COLUMNS),
            "prediction_horizon_minutes": 15,
            "prediction_horizon_bars": 3,
            "threshold_value_pct": 0.1,
        },
    }


def _predictions(features, artifacts):
    return pd.DataFrame(
        {
            "datetime_et": features["datetime_et"].to_numpy(),
            "predicted_direction": [1] * len(features),
            "predicted_direction_name": ["UP"] * len(features),
            "predicted_return_15m": [0.25] * len(features),
            "probability_down": [0.1] * len(features),
            "probability_neutral": [0.2] * len(features),
            "probability_up": [0.7] * len(features),
            "classifier_name": [artifacts["classifier_name"]] * len(features),
            "regressor_name": [artifacts["regressor_name"]] * len(features),
            "prediction_horizon_minutes": [15] * len(features),
        }
    )


def test_single_pipeline_response_is_serializable(monkeypatch):
    artifacts = _artifacts()
    monkeypatch.setattr(pipeline, "load_model_artifacts", lambda *args: artifacts)
    monkeypatch.setattr(pipeline, "predict", _predictions)

    result = pipeline.generate_prediction(_features())

    assert result["batch"] is False
    assert result["signal"]["signal"] == "BULLISH"
    assert result["scenarios"]["Bullish"]["probability"] == 0.7
    json.dumps(result)


def test_batch_pipeline_response(monkeypatch):
    artifacts = _artifacts()
    monkeypatch.setattr(pipeline, "load_model_artifacts", lambda *args: artifacts)
    monkeypatch.setattr(pipeline, "predict", _predictions)

    result = pipeline.generate_prediction(_features(2))

    assert result["batch"] is True
    assert len(result["predictions"]) == 2


def test_invalid_and_empty_inputs_are_rejected():
    with pytest.raises(ValueError):
        pipeline.generate_prediction(pd.DataFrame())
    with pytest.raises(ValueError):
        pipeline.generate_prediction(_features().drop(columns=[FEATURE_COLUMNS[0]]))


def test_model_loading_failure_is_propagated(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("missing model")

    monkeypatch.setattr(pipeline, "load_model_artifacts", fail)
    with pytest.raises(FileNotFoundError, match="missing model"):
        pipeline.generate_prediction(_features())

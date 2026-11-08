import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.technical_indicators import FEATURE_COLUMNS
from src.models.predict import (
    load_model_artifacts,
    predict,
    validate_prediction_features,
)


def _input(rows=3):
    frame = pd.DataFrame(
        {name: np.arange(rows, dtype=float) + 1 for name in FEATURE_COLUMNS}
    )
    frame["datetime_et"] = pd.date_range(
        "2025-01-01", periods=rows, tz="America/New_York"
    )
    return frame


class DummyClassifier:
    classes_ = np.array([1, -1, 0])

    def predict(self, features):
        return np.array([1, -1, 0])[: len(features)]

    def predict_proba(self, features):
        return np.tile([[0.6, 0.1, 0.3]], (len(features), 1))


class DummyRegressor:
    def predict(self, features):
        return np.full(len(features), 0.25)


def _artifacts():
    return {
        "classifier": DummyClassifier(),
        "regressor": DummyRegressor(),
        "classifier_name": "dummy_classifier",
        "regressor_name": "dummy_regressor",
        "metadata": {
            "feature_names": list(FEATURE_COLUMNS),
            "prediction_horizon_minutes": 15,
        },
    }


def test_prediction_maps_classes_and_preserves_timestamp():
    output = predict(_input(), _artifacts())

    assert output["predicted_direction_name"].tolist() == ["UP", "DOWN", "NEUTRAL"]
    assert output["probability_up"].iloc[0] == 0.6
    assert output["probability_down"].iloc[0] == 0.1
    assert output["probability_neutral"].iloc[0] == 0.3
    assert output["predicted_return_15m"].tolist() == [0.25, 0.25, 0.25]
    assert output["datetime_et"].equals(_input()["datetime_et"])


def test_invalid_features_are_rejected():
    for invalid in (
        _input().drop(columns=[FEATURE_COLUMNS[0]]),
        _input().assign(**{FEATURE_COLUMNS[1]: np.inf}),
        _input().assign(**{FEATURE_COLUMNS[2]: "invalid"}),
    ):
        with pytest.raises(ValueError):
            validate_prediction_features(invalid)


def test_missing_metadata_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="model metadata"):
        load_model_artifacts(tmp_path, selection_path=None)

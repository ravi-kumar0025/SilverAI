"""Prediction helpers for the selected SilverAI classifier and regressor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.technical_indicators import FEATURE_COLUMNS


PREDICTION_HORIZON_MINUTES = 15
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2] / "saved_models"
DEFAULT_SELECTION_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "model_comparison"
    / "evaluation_selection.json"
)
CLASS_LABEL_NAMES = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _resolve_model_path(value: str, model_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = model_root.parent
    project_relative = project_root / path
    return project_relative if project_relative.exists() else model_root / path


def _selected_name(
    metadata: dict[str, Any], selection: dict[str, Any], kind: str
) -> str:
    key = f"selected_{kind}"
    name = selection.get(key)
    if name is None:
        name = metadata.get(f"best_validation_{kind}_model")
    if not isinstance(name, str) or not name:
        raise ValueError(f"No selected {kind} name is present in model metadata.")
    return name


def load_model_artifacts(
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    selection_path: str | Path | None = DEFAULT_SELECTION_PATH,
) -> dict[str, Any]:
    """Load the selected classifier, regressor, metadata, and their paths."""

    root = Path(model_root)
    metadata_path = root / "metadata" / "training_metadata.json"
    metadata = _load_json(metadata_path, "model metadata")
    selection = (
        _load_json(Path(selection_path), "model selection metadata")
        if selection_path is not None and Path(selection_path).exists()
        else {}
    )
    classifier_name = _selected_name(metadata, selection, "classifier")
    regressor_name = _selected_name(metadata, selection, "regressor")
    feature_names = metadata.get("feature_names")
    if feature_names != list(FEATURE_COLUMNS):
        raise ValueError("Saved model metadata does not match the approved feature order.")

    saved_paths = metadata.get("saved_model_paths", {})
    classifier_value = saved_paths.get(f"classifier:{classifier_name}")
    regressor_value = saved_paths.get(f"regressor:{regressor_name}")
    classifier_path = (
        _resolve_model_path(classifier_value, root)
        if isinstance(classifier_value, str)
        else root / "classifier" / f"{classifier_name}.joblib"
    )
    regressor_path = (
        _resolve_model_path(regressor_value, root)
        if isinstance(regressor_value, str)
        else root / "regressor" / f"{regressor_name}.joblib"
    )
    for path, kind in (
        (classifier_path, "classifier"),
        (regressor_path, "regressor"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Selected {kind} model not found: {path}")
    try:
        classifier = joblib.load(classifier_path)
        regressor = joblib.load(regressor_path)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"Could not load selected model artifacts: {exc}") from exc

    return {
        "classifier": classifier,
        "regressor": regressor,
        "metadata": metadata,
        "classifier_name": classifier_name,
        "regressor_name": regressor_name,
        "classifier_path": classifier_path,
        "regressor_path": regressor_path,
        "metadata_path": metadata_path,
    }


def validate_prediction_features(
    df: pd.DataFrame, feature_names: tuple[str, ...] = FEATURE_COLUMNS
) -> pd.DataFrame:
    """Validate and return features in the exact training order."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Prediction input must be a pandas DataFrame.")
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required features: {', '.join(missing)}")
    features = df[list(feature_names)].copy()
    numeric = features.apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric)
    if invalid.any().any():
        bad_columns = invalid.any(axis=0)
        bad = ", ".join(bad_columns[bad_columns].index)
        raise ValueError(f"Prediction features contain missing, infinite, or invalid values: {bad}")
    return numeric


def _timestamp_result(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if timestamp_col in df.columns:
        return pd.DataFrame({timestamp_col: df[timestamp_col].to_numpy()})
    return pd.DataFrame(index=range(len(df)))


def _artifact_metadata(
    artifacts: dict[str, Any], key: str, fallback: Any
) -> Any:
    return artifacts.get("metadata", {}).get(key, fallback)


def predict_classification(
    df: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    *,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    selection_path: str | Path | None = DEFAULT_SELECTION_PATH,
    timestamp_col: str = "datetime_et",
) -> pd.DataFrame:
    """Predict direction and class probabilities for one or more rows."""

    loaded = artifacts or load_model_artifacts(model_root, selection_path)
    features = validate_prediction_features(df)
    model = loaded["classifier"]
    predictions = np.asarray(model.predict(features))
    try:
        direction = predictions.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("Classifier predictions are not valid direction labels.") from exc
    if not np.isin(direction, list(CLASS_LABEL_NAMES)).all():
        raise ValueError("Classifier returned labels outside {-1, 0, 1}.")

    result = _timestamp_result(df, timestamp_col)
    result["predicted_direction"] = direction
    result["predicted_direction_name"] = [CLASS_LABEL_NAMES[int(value)] for value in direction]
    result["classifier_name"] = loaded["classifier_name"]
    result["prediction_horizon_minutes"] = int(
        _artifact_metadata(loaded, "prediction_horizon_minutes", PREDICTION_HORIZON_MINUTES)
    )

    predict_proba = getattr(model, "predict_proba", None)
    classes = getattr(model, "classes_", None)
    if callable(predict_proba) and classes is not None:
        probabilities = np.asarray(predict_proba(features))
        if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
            raise ValueError("Classifier returned probabilities with an invalid shape.")
        for label, name in ((-1, "down"), (0, "neutral"), (1, "up")):
            result[f"probability_{name}"] = 0.0
        for column, class_value in enumerate(classes):
            try:
                label = int(class_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Classifier classes contain an invalid direction label.") from exc
            if label not in CLASS_LABEL_NAMES:
                raise ValueError("Classifier classes contain labels outside {-1, 0, 1}.")
            result[f"probability_{CLASS_LABEL_NAMES[label].lower()}"] = probabilities[:, column]
    return result


def predict_regression(
    df: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    *,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    selection_path: str | Path | None = DEFAULT_SELECTION_PATH,
    timestamp_col: str = "datetime_et",
) -> pd.DataFrame:
    """Predict the 15-minute future return in percentage units."""

    loaded = artifacts or load_model_artifacts(model_root, selection_path)
    features = validate_prediction_features(df)
    predictions = np.asarray(loaded["regressor"].predict(features), dtype=float)
    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise ValueError("Regressor returned invalid predictions.")
    result = _timestamp_result(df, timestamp_col)
    result["predicted_return_15m"] = predictions
    result["regressor_name"] = loaded["regressor_name"]
    result["prediction_horizon_minutes"] = int(
        _artifact_metadata(loaded, "prediction_horizon_minutes", PREDICTION_HORIZON_MINUTES)
    )
    return result


def predict(
    df: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    *,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    selection_path: str | Path | None = DEFAULT_SELECTION_PATH,
    timestamp_col: str = "datetime_et",
) -> pd.DataFrame:
    """Return independent classifier and regressor outputs in one DataFrame."""

    loaded = artifacts or load_model_artifacts(model_root, selection_path)
    classification = predict_classification(
        df, loaded, timestamp_col=timestamp_col
    )
    regression = predict_regression(df, loaded, timestamp_col=timestamp_col)
    result = classification.copy()
    result["predicted_return_15m"] = regression["predicted_return_15m"].to_numpy()
    result["regressor_name"] = regression["regressor_name"].to_numpy()
    return result


__all__ = [
    "DEFAULT_MODEL_ROOT",
    "DEFAULT_SELECTION_PATH",
    "PREDICTION_HORIZON_MINUTES",
    "load_model_artifacts",
    "predict",
    "predict_classification",
    "predict_regression",
    "validate_prediction_features",
]

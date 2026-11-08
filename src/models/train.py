"""Reusable, chronological training workflow for SilverAI baseline models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_modeling_data
from src.data.preprocessing import prepare_intraday_data
from src.features.technical_indicators import FEATURE_COLUMNS


SEED = 42
HORIZON_MINUTES = 15
HORIZON_BARS = 3
PURGE_ROWS = 3
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
TRAIN_NEUTRAL_QUANTILE = 0.40
REGRESSION_TARGET = "future_return_15m"
SAVED_CLASSIFICATION_TARGET = "target_direction_15m"
CLASSIFICATION_TARGET = "classification_target"
LABELS = (-1, 0, 1)
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2] / "saved_models"


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}.")


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare modeling data and remove unusable feature/target rows.

    The input is copied and sorted chronologically. Rows with missing or
    non-finite approved features or targets are removed, matching Notebook 07;
    no values are imputed here.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    required = (
        "datetime_et",
        *_OHLCV_COLUMNS,
        *FEATURE_COLUMNS,
        REGRESSION_TARGET,
        SAVED_CLASSIFICATION_TARGET,
    )
    _require_columns(df, required)
    prepared = prepare_intraday_data(df, timestamp_col="datetime_et")
    numeric_columns = [*FEATURE_COLUMNS, REGRESSION_TARGET, SAVED_CLASSIFICATION_TARGET]
    numeric_values = prepared[numeric_columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric_values.notna().all(axis=1) & np.isfinite(numeric_values).all(axis=1)
    return prepared.loc[usable].reset_index(drop=True).copy()


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the 17 approved model-input features in fixed order."""

    _require_columns(df, tuple(FEATURE_COLUMNS))
    return df[list(FEATURE_COLUMNS)].copy()


def calibrate_classification_threshold(
    train_df: pd.DataFrame, neutral_quantile: float = TRAIN_NEUTRAL_QUANTILE
) -> float:
    """Calibrate the neutral-band threshold from training returns only."""

    if not 0 < neutral_quantile < 1:
        raise ValueError("neutral_quantile must be between 0 and 1.")
    _require_columns(train_df, (REGRESSION_TARGET,))
    returns = pd.to_numeric(train_df[REGRESSION_TARGET], errors="coerce")
    if returns.isna().any() or not np.isfinite(returns).all():
        raise ValueError("Training returns must be finite before threshold calibration.")
    return float(returns.abs().quantile(neutral_quantile))


def prepare_targets(
    df: pd.DataFrame, threshold_pct: float
) -> tuple[pd.Series, pd.Series]:
    """Return regression returns and thresholded direction labels."""

    _require_columns(df, (REGRESSION_TARGET,))
    returns = pd.to_numeric(df[REGRESSION_TARGET], errors="coerce").astype(float)
    if returns.isna().any() or not np.isfinite(returns).all():
        raise ValueError("Target returns must be finite before model training.")
    labels = pd.Series(0, index=df.index, dtype="int64")
    labels.loc[returns > threshold_pct] = 1
    labels.loc[returns < -threshold_pct] = -1
    labels.name = CLASSIFICATION_TARGET
    returns.name = REGRESSION_TARGET
    return returns, labels


def split_chronologically(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    purge_rows: int = PURGE_ROWS,
) -> dict[str, Any]:
    """Split sorted data into purged train/validation and untouched test sets."""

    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must leave test rows.")
    if purge_rows < 0 or not isinstance(purge_rows, int):
        raise ValueError("purge_rows must be a nonnegative integer.")
    _require_columns(df, ("datetime_et",))
    if len(df) < 3 * (purge_rows + 1):
        raise ValueError("Dataset is too small for the requested split and purge.")

    ordered = df.sort_values("datetime_et", kind="mergesort").reset_index(drop=True)
    train_boundary = int(len(ordered) * train_fraction)
    validation_boundary = int(len(ordered) * (train_fraction + validation_fraction))
    train_raw = ordered.iloc[:train_boundary].copy()
    validation_raw = ordered.iloc[train_boundary:validation_boundary].copy()
    test = ordered.iloc[validation_boundary:].copy()
    if len(train_raw) <= purge_rows or len(validation_raw) <= purge_rows or test.empty:
        raise ValueError("Partitions are too small for the requested purge.")

    train = train_raw.iloc[:-purge_rows].copy() if purge_rows else train_raw.copy()
    validation = (
        validation_raw.iloc[:-purge_rows].copy() if purge_rows else validation_raw.copy()
    )
    if train.empty or validation.empty:
        raise ValueError("Purging removed an entire training or validation partition.")

    def boundary(name: str, partition: pd.DataFrame) -> dict[str, Any]:
        return {
            "partition": name,
            "rows": len(partition),
            "start": str(partition["datetime_et"].iloc[0]),
            "end": str(partition["datetime_et"].iloc[-1]),
        }

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "metadata": {
            "train_boundary_raw_index": train_boundary,
            "validation_boundary_raw_index": validation_boundary,
            "split_fractions_nominal": {
                "train": train_fraction,
                "validation": validation_fraction,
                "test": 1.0 - train_fraction - validation_fraction,
            },
            "purge_rows_at_boundaries": purge_rows,
            "partitions": [
                boundary("train", train),
                boundary("validation", validation),
                boundary("test", test),
            ],
        },
    }


def build_classification_models() -> dict[str, Any]:
    """Build Notebook 07's three classification estimators."""

    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=150, random_state=SEED, n_jobs=-1
        ),
        "hist_gradient_boosting_classifier": HistGradientBoostingClassifier(
            max_iter=100, random_state=SEED
        ),
    }


def build_regression_models() -> dict[str, Any]:
    """Build Notebook 07's three regression estimators."""

    return {
        "linear_regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())]
        ),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=150, random_state=SEED, n_jobs=-1
        ),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
            max_iter=100, random_state=SEED
        ),
    }


def evaluate_classification(
    y_true: pd.Series, predictions: np.ndarray, model_name: str | None = None
) -> dict[str, Any]:
    """Calculate validation classification metrics and confusion matrix."""

    report = classification_report(
        y_true, predictions, labels=list(LABELS), output_dict=True, zero_division=0
    )
    metrics: dict[str, Any] = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(LABELS)).tolist(),
    }
    for label, name in ((-1, "down"), (0, "neutral"), (1, "up")):
        values = report[str(label)]
        metrics[f"{name}_precision"] = float(values["precision"])
        metrics[f"{name}_recall"] = float(values["recall"])
        metrics[f"{name}_f1"] = float(values["f1-score"])
    return metrics


def evaluate_regression(
    y_true: pd.Series, predictions: np.ndarray, model_name: str | None = None
) -> dict[str, Any]:
    """Calculate validation regression metrics."""

    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def train_classification_models(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, threshold_pct: float
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit all classifiers on train data and evaluate them on validation data."""

    _, y_train = prepare_targets(train_df, threshold_pct)
    _, y_validation = prepare_targets(validation_df, threshold_pct)
    X_train = select_features(train_df)
    X_validation = select_features(validation_df)
    models = build_classification_models()
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics.append(
            evaluate_classification(
                y_validation, model.predict(X_validation), model_name=name
            )
        )
    return models, pd.DataFrame(metrics).sort_values("macro_f1", ascending=False).reset_index(
        drop=True
    )


def train_regression_models(
    train_df: pd.DataFrame, validation_df: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit all regressors on train data and evaluate them on validation data."""

    y_train = pd.to_numeric(train_df[REGRESSION_TARGET], errors="raise").astype(float)
    y_validation = pd.to_numeric(validation_df[REGRESSION_TARGET], errors="raise").astype(
        float
    )
    X_train = select_features(train_df)
    X_validation = select_features(validation_df)
    models = build_regression_models()
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics.append(
            evaluate_regression(y_validation, model.predict(X_validation), model_name=name)
        )
    return models, pd.DataFrame(metrics).sort_values("rmse").reset_index(drop=True)


def compare_with_baselines(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, threshold_pct: float,
    classification_metrics: pd.DataFrame, regression_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add majority-class and zero-return validation baselines."""

    _, y_train_cls = prepare_targets(train_df, threshold_pct)
    _, y_validation_cls = prepare_targets(validation_df, threshold_pct)
    majority_class = int(y_train_cls.mode().iloc[0])
    cls_baseline = evaluate_classification(
        y_validation_cls,
        np.full(len(y_validation_cls), majority_class),
        model_name="majority_class_baseline",
    )
    y_validation_reg = pd.to_numeric(
        validation_df[REGRESSION_TARGET], errors="raise"
    ).astype(float)
    reg_baseline = evaluate_regression(
        y_validation_reg,
        np.zeros(len(y_validation_reg)),
        model_name="zero_return_baseline",
    )
    return (
        pd.concat([pd.DataFrame([cls_baseline]), classification_metrics], ignore_index=True),
        pd.concat([pd.DataFrame([reg_baseline]), regression_metrics], ignore_index=True),
    )


def _jsonable_parameters(models: dict[str, Any]) -> dict[str, Any]:
    return {name: model.get_params() for name, model in models.items()}


def save_models_and_metadata(
    classification_models: dict[str, Any],
    regression_models: dict[str, Any],
    metadata: dict[str, Any],
    model_root: str | Path = DEFAULT_MODEL_ROOT,
) -> dict[str, str]:
    """Save fitted models and JSON metadata under the approved directories."""

    root = Path(model_root)
    classifier_path = root / "classifier"
    regressor_path = root / "regressor"
    metadata_path = root / "metadata"
    for path in (classifier_path, regressor_path, metadata_path):
        path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}
    for name, model in classification_models.items():
        path = classifier_path / f"{name}.joblib"
        joblib.dump(model, path)
        saved_paths[f"classifier:{name}"] = str(path)
    for name, model in regression_models.items():
        path = regressor_path / f"{name}.joblib"
        joblib.dump(model, path)
        saved_paths[f"regressor:{name}"] = str(path)

    metadata = dict(metadata)
    metadata_file = metadata_path / "training_metadata.json"
    metadata["saved_model_paths"] = {
        **saved_paths,
        "metadata": str(metadata_file),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    saved_paths["metadata"] = str(metadata_file)
    return saved_paths


def run_training(
    df: pd.DataFrame | None = None,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Run the complete training workflow and optionally save all artifacts."""

    modeling_data = load_modeling_data() if df is None else df.copy()
    clean_df = prepare_training_data(modeling_data)
    splits = split_chronologically(clean_df)
    train_df = splits["train"]
    validation_df = splits["validation"]
    threshold_pct = calibrate_classification_threshold(train_df)
    classification_models, classification_metrics = train_classification_models(
        train_df, validation_df, threshold_pct
    )
    regression_models, regression_metrics = train_regression_models(train_df, validation_df)
    classification_comparison, regression_comparison = compare_with_baselines(
        train_df,
        validation_df,
        threshold_pct,
        classification_metrics,
        regression_metrics,
    )
    best_classification = str(classification_metrics.iloc[0]["model"])
    best_regression = str(regression_metrics.iloc[0]["model"])
    metadata: dict[str, Any] = {
        "feature_names": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "raw_ohlcv_excluded": True,
        "prediction_horizon_minutes": HORIZON_MINUTES,
        "prediction_horizon_bars": HORIZON_BARS,
        "classification_target": CLASSIFICATION_TARGET,
        "source_classification_target": SAVED_CLASSIFICATION_TARGET,
        "regression_target": REGRESSION_TARGET,
        "classification_labels": {"-1": "DOWN", "0": "NEUTRAL", "1": "UP"},
        "threshold_value_pct": threshold_pct,
        "threshold_calibration_method": (
            "40th percentile of absolute returns in purged training partition only"
        ),
        "threshold_is_provisional": True,
        "random_state": SEED,
        "dataset_rows_after_validation": len(clean_df),
        "best_validation_classification_model": best_classification,
        "best_validation_regression_model": best_regression,
        "validation_classification_metrics": classification_metrics.to_dict(orient="records"),
        "validation_regression_metrics": regression_metrics.to_dict(orient="records"),
        "classification_comparison": classification_comparison.to_dict(orient="records"),
        "regression_comparison": regression_comparison.to_dict(orient="records"),
        "model_parameters": {
            "classification": _jsonable_parameters(classification_models),
            "regression": _jsonable_parameters(regression_models),
        },
        "split_metadata": splits["metadata"],
        "test_evaluated": False,
    }
    saved_paths = (
        save_models_and_metadata(
            classification_models, regression_models, metadata, model_root=model_root
        )
        if save_artifacts
        else {}
    )
    return {
        "clean_data": clean_df,
        "splits": splits,
        "threshold_pct": threshold_pct,
        "classification_models": classification_models,
        "regression_models": regression_models,
        "classification_metrics": classification_metrics,
        "regression_metrics": regression_metrics,
        "classification_comparison": classification_comparison,
        "regression_comparison": regression_comparison,
        "metadata": metadata,
        "saved_paths": saved_paths,
    }


_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

__all__ = [
    "CLASSIFICATION_TARGET",
    "DEFAULT_MODEL_ROOT",
    "FEATURE_COLUMNS",
    "REGRESSION_TARGET",
    "SAVED_CLASSIFICATION_TARGET",
    "build_classification_models",
    "build_regression_models",
    "calibrate_classification_threshold",
    "compare_with_baselines",
    "evaluate_classification",
    "evaluate_regression",
    "prepare_targets",
    "prepare_training_data",
    "run_training",
    "save_models_and_metadata",
    "select_features",
    "split_chronologically",
    "train_classification_models",
    "train_regression_models",
]

"""Reusable evaluation of SilverAI models on chronological data partitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_modeling_data
from src.models.predict import (
    DEFAULT_MODEL_ROOT,
    DEFAULT_SELECTION_PATH,
    load_model_artifacts,
)
from src.models.train import (
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    REGRESSION_TARGET,
    SAVED_CLASSIFICATION_TARGET,
    prepare_targets,
    prepare_training_data,
    select_features,
    split_chronologically,
)
from src.config import CONFIG
from src.evaluation.baselines import rsi_direction


LABELS = (-1, 0, 1)
LABEL_NAMES = {"-1": "DOWN", "0": "NEUTRAL", "1": "UP"}
DEFAULT_REPORT_ROOT = Path(__file__).resolve().parents[2] / "reports" / "model_comparison"
DEFAULT_FIGURE_ROOT = Path(__file__).resolve().parents[2] / "reports" / "figures"


def evaluate_classification(
    y_true: pd.Series,
    predictions: np.ndarray,
    model_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Return complete classification metrics with fixed DOWN/NEUTRAL/UP order."""

    if len(y_true) == 0:
        raise ValueError(f"Cannot evaluate classification on empty {dataset_name} data.")
    report = classification_report(
        y_true,
        predictions,
        labels=list(LABELS),
        target_names=[LABEL_NAMES[str(label)] for label in LABELS],
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "model": model_name,
        "rows": len(y_true),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(LABELS)).tolist(),
        "classification_report": report,
        "signal_frequency": float(np.mean(np.asarray(predictions) != 0)),
        "neutral_coverage": float(np.mean(np.asarray(predictions) == 0)),
        "prediction_distribution": {str(label): int(np.sum(np.asarray(predictions) == label)) for label in LABELS},
    }
    for label in LABELS:
        values = report[LABEL_NAMES[str(label)]]
        name = LABEL_NAMES[str(label)].lower()
        result[f"{name}_precision"] = float(values["precision"])
        result[f"{name}_recall"] = float(values["recall"])
        result[f"{name}_f1"] = float(values["f1-score"])
    return result


def evaluate_regression(
    y_true: pd.Series,
    predictions: np.ndarray,
    model_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Return MAE, RMSE, and safe R² for one chronological partition."""

    if len(y_true) == 0:
        raise ValueError(f"Cannot evaluate regression on empty {dataset_name} data.")
    values = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if len(values) != len(predicted):
        raise ValueError("Regression targets and predictions have different lengths.")
    if not np.isfinite(values).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression targets and predictions must be finite.")
    r2 = float(r2_score(values, predicted)) if len(np.unique(values)) > 1 else float("nan")
    return {
        "dataset": dataset_name,
        "model": model_name,
        "rows": len(values),
        "mae": float(mean_absolute_error(values, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(values, predicted))),
        "r2": r2,
    }


def majority_class_baseline(
    y_train: pd.Series, y_eval: pd.Series, dataset_name: str
) -> dict[str, Any]:
    """Evaluate a majority class learned from training labels only."""

    if len(y_train) == 0:
        raise ValueError("Cannot calculate a majority baseline without training labels.")
    majority = int(y_train.mode().iloc[0])
    predictions = np.full(len(y_eval), majority)
    result = evaluate_classification(
        y_eval, predictions, "majority_class_baseline", dataset_name
    )
    result["majority_class"] = majority
    return result


def zero_return_baseline(y_eval: pd.Series, dataset_name: str) -> dict[str, Any]:
    """Evaluate the no-predicted-return regression baseline."""

    return evaluate_regression(
        y_eval,
        np.zeros(len(y_eval)),
        "zero_return_baseline",
        dataset_name,
    )


def _model_names(metadata: dict[str, Any], family: str, root: Path) -> list[str]:
    configured = metadata.get("model_parameters", {}).get(family, {})
    names = list(configured) if configured else [path.stem for path in (root / family).glob("*.joblib")]
    if not names:
        raise FileNotFoundError(f"No saved {family} models found under {root / family}.")
    return names


def _load_named_model(name: str, family: str, metadata: dict[str, Any], root: Path) -> Any:
    saved_paths = metadata.get("saved_model_paths", {})
    value = saved_paths.get(f"{family}:{name}")
    if isinstance(value, str):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root.parent / candidate if (root.parent / candidate).exists() else root / candidate
    else:
        candidate = root / family / f"{name}.joblib"
    if not candidate.is_file():
        raise FileNotFoundError(f"Saved {family} model not found: {candidate}")
    try:
        return joblib.load(candidate)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"Could not load saved {family} model {candidate}: {exc}") from exc


def _metric_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{key: value for key, value in row.items() if key not in {"confusion_matrix", "classification_report"}}
         for row in results]
    )


def _save_outputs(
    classification_results: list[dict[str, Any]],
    regression_results: list[dict[str, Any]],
    metadata: dict[str, Any],
    report_root: Path,
    figure_root: Path,
) -> dict[str, Path]:
    report_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    classification_table = _metric_table(classification_results)
    regression_table = _metric_table(regression_results)
    classification_table.to_csv(report_root / "classification_metrics.csv", index=False)
    regression_table.to_csv(report_root / "regression_metrics.csv", index=False)

    confusion_rows = []
    report_rows = []
    for result in classification_results:
        matrix = result["confusion_matrix"]
        for row_index, actual in enumerate(("DOWN", "NEUTRAL", "UP")):
            for col_index, predicted in enumerate(("DOWN", "NEUTRAL", "UP")):
                confusion_rows.append({
                    "dataset": result["dataset"], "model": result["model"],
                    "actual": actual, "predicted": predicted,
                    "count": matrix[row_index][col_index],
                })
        for label, values in result["classification_report"].items():
            if isinstance(values, dict):
                report_rows.append({"dataset": result["dataset"], "model": result["model"], "class": label, **values})
    pd.DataFrame(confusion_rows).to_csv(report_root / "classification_confusion_matrix.csv", index=False)
    pd.DataFrame(report_rows).to_csv(report_root / "classification_reports.csv", index=False)

    selection = {
        "selected_classifier": metadata["selected_classifier"],
        "selected_regressor": metadata["selected_regressor"],
        "selection_basis": "Training validation macro F1 and RMSE; test results not used for selection",
        "test_evaluated": True,
    }
    metadata["evaluation_selection"] = selection
    metadata_path = report_root / "evaluation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    selected_test = [
        row for row in classification_results
        if row["dataset"] == "test" and row["model"] == metadata["selected_classifier"]
    ]
    if selected_test:
        matrix = np.asarray(selected_test[0]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(3), ["DOWN", "NEUTRAL", "UP"])
        ax.set_yticks(range(3), ["DOWN", "NEUTRAL", "UP"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Selected Classifier Test Confusion Matrix")
        for row_index in range(3):
            for col_index in range(3):
                ax.text(col_index, row_index, int(matrix[row_index, col_index]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(figure_root / "evaluation_confusion_matrix.png", dpi=160)
        plt.close(fig)

    return {
        "classification_metrics": report_root / "classification_metrics.csv",
        "regression_metrics": report_root / "regression_metrics.csv",
        "confusion_matrix": report_root / "classification_confusion_matrix.csv",
        "classification_reports": report_root / "classification_reports.csv",
        "metadata": metadata_path,
        "confusion_figure": figure_root / "evaluation_confusion_matrix.png",
    }


def evaluate_saved_models(
    df: pd.DataFrame | None = None,
    *,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    selection_path: str | Path | None = DEFAULT_SELECTION_PATH,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
    figure_root: str | Path = DEFAULT_FIGURE_ROOT,
) -> dict[str, Any]:
    """Evaluate all available saved models on the fixed train/validation/test split."""

    artifacts = load_model_artifacts(model_root, selection_path)
    root = Path(model_root)
    modeling_data = load_modeling_data() if df is None else df.copy()
    clean = prepare_training_data(modeling_data)
    splits = split_chronologically(clean)
    metadata = artifacts["metadata"]
    threshold = metadata.get("threshold_value_pct")
    if threshold is None:
        raise ValueError("Training metadata is missing threshold_value_pct.")
    threshold = float(threshold)
    expected_partitions = metadata.get("split_metadata", {}).get("partitions")
    if expected_partitions:
        actual_partitions = splits["metadata"]["partitions"]
        if expected_partitions != actual_partitions:
            raise ValueError("Evaluation split boundaries do not match training metadata.")

    classifier_names = _model_names(metadata, "classifier", root)
    regressor_names = _model_names(metadata, "regressor", root)
    classifier_models = {
        name: artifacts["classifier"] if name == artifacts["classifier_name"] else _load_named_model(name, "classifier", metadata, root)
        for name in classifier_names
    }
    regressor_models = {
        name: artifacts["regressor"] if name == artifacts["regressor_name"] else _load_named_model(name, "regressor", metadata, root)
        for name in regressor_names
    }

    classification_results: list[dict[str, Any]] = []
    regression_results: list[dict[str, Any]] = []
    train_labels = prepare_targets(splits["train"], threshold)[1]
    for dataset_name in ("train", "validation", "test"):
        partition = splits[dataset_name]
        _, labels = prepare_targets(partition, threshold)
        features = select_features(partition)
        for name, model in classifier_models.items():
            classification_results.append(
                evaluate_classification(labels, model.predict(features), name, dataset_name)
            )
        classification_results.append(majority_class_baseline(train_labels, labels, dataset_name))
        classification_results.append(
            evaluate_classification(labels, np.ones(len(labels), dtype=int), "always_up_baseline", dataset_name)
        )
        classification_results.append(
            evaluate_classification(labels, np.full(len(labels), -1, dtype=int), "always_down_baseline", dataset_name)
        )
        classification_results.append(
            evaluate_classification(labels, rsi_direction(partition["close"]).fillna(0).astype(int).to_numpy(), "rsi_14_baseline", dataset_name)
        )
        targets = partition[REGRESSION_TARGET].astype(float)
        for name, model in regressor_models.items():
            regression_results.append(
                evaluate_regression(targets, model.predict(features), name, dataset_name)
            )
        regression_results.append(zero_return_baseline(targets, dataset_name))

    validation_classification = [
        row for row in classification_results if row["dataset"] == "validation"
    ]
    validation_regression = [
        row for row in regression_results if row["dataset"] == "validation"
    ]
    ranked_classification = sorted(
        (row for row in validation_classification if not row["model"].endswith("_baseline")),
        key=lambda row: row["macro_f1"], reverse=True,
    )
    ranked_regression = sorted(
        (row for row in validation_regression if row["model"] != "zero_return_baseline"),
        key=lambda row: row["rmse"],
    )
    if ranked_classification[0]["model"] != artifacts["classifier_name"]:
        raise ValueError("Selected classifier does not match validation macro-F1 ranking.")
    if ranked_regression[0]["model"] != artifacts["regressor_name"]:
        raise ValueError("Selected regressor does not match validation RMSE ranking.")

    output_metadata = {
        "feature_names": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "classification_target": CLASSIFICATION_TARGET,
        "source_classification_target": SAVED_CLASSIFICATION_TARGET,
        "regression_target": REGRESSION_TARGET,
        "prediction_horizon_minutes": metadata.get("prediction_horizon_minutes", CONFIG.forecast_horizon_minutes),
        "threshold_value_pct": threshold,
        "selected_classifier": artifacts["classifier_name"],
        "selected_regressor": artifacts["regressor_name"],
        "clean_rows": len(clean),
        "split_metadata": splits["metadata"],
        "test_used_for_selection": False,
    }
    paths = _save_outputs(
        classification_results,
        regression_results,
        output_metadata,
        Path(report_root),
        Path(figure_root),
    )
    return {
        "splits": splits,
        "classification_results": pd.DataFrame(_metric_table(classification_results)),
        "regression_results": pd.DataFrame(_metric_table(regression_results)),
        "validation_classification": pd.DataFrame(_metric_table(validation_classification)),
        "validation_regression": pd.DataFrame(_metric_table(validation_regression)),
        "selected_classifier": artifacts["classifier_name"],
        "selected_regressor": artifacts["regressor_name"],
        "output_paths": paths,
        "metadata": output_metadata,
    }


__all__ = [
    "DEFAULT_FIGURE_ROOT",
    "DEFAULT_REPORT_ROOT",
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_saved_models",
    "majority_class_baseline",
    "zero_return_baseline",
]

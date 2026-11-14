"""Non-training final holdout evaluation for the frozen calibration experiment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from src.data.loader import load_modeling_data
from src.features.technical_indicators import FEATURE_COLUMNS
from src.models.calibration import calibration_metrics, sha256_file, stable_feature_schema_hash
from src.models.train import prepare_targets, prepare_training_data, select_features, split_chronologically


LABELS = (-1, 0, 1)
LABEL_NAMES = {"-1": "DOWN", "0": "NEUTRAL", "1": "UP"}


def _classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    report = classification_report(actual, predicted, labels=list(LABELS), output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "per_class": {
            LABEL_NAMES[str(label)]: {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(LABELS)).tolist(),
    }


def _distribution(values: np.ndarray) -> dict[str, int]:
    return {LABEL_NAMES[str(label)]: int(np.sum(values == label)) for label in LABELS}


def _assert_same(expected: Any, actual: Any, description: str) -> None:
    if expected != actual:
        raise ValueError(f"{description} mismatch: expected {expected!r}, got {actual!r}.")


def run_final_holdout_evaluation(
    experiment_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_path: str | Path = "data/silver_modeling_data.csv",
) -> dict[str, Any]:
    """Evaluate frozen experiment artifacts exactly once on the untouched test split."""

    experiment = Path(experiment_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    experiment_metadata = json.loads((experiment / "metrics.json").read_text(encoding="utf-8"))
    if experiment_metadata.get("selected_method") != "sigmoid":
        raise ValueError("The supplied experiment does not contain the required selected sigmoid calibrator.")
    if experiment_metadata.get("test_used") is not False:
        raise ValueError("The experiment metadata does not prove that the test partition was unused.")

    classifier_path = experiment / "classifier.joblib"
    calibrator_path = experiment / "calibrator.joblib"
    classifier_hash = sha256_file(str(classifier_path))
    calibrator_hash = sha256_file(str(calibrator_path))
    _assert_same(experiment_metadata["artifact_sha256"]["classifier"], classifier_hash, "Classifier artifact hash")
    _assert_same(experiment_metadata["artifact_sha256"]["calibrator"], calibrator_hash, "Calibrator artifact hash")

    data = prepare_training_data(load_modeling_data(dataset_path.parent if isinstance(dataset_path, Path) else Path(dataset_path).parent))
    splits = split_chronologically(data)
    split_metadata = experiment_metadata["split_metadata"]
    _assert_same(split_metadata["partitions"], splits["metadata"]["partitions"], "Chronological split boundaries")
    test = splits["test"]
    timestamps = pd.to_datetime(test["datetime_et"], utc=True)
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Test timestamps are not strictly chronological and unique.")

    classifier = joblib.load(classifier_path)
    calibrator = joblib.load(calibrator_path)
    if list(getattr(classifier, "classes_", [])) != list(LABELS):
        raise ValueError(f"Classifier class ordering mismatch: {getattr(classifier, 'classes_', None)!r}")
    if not calibrator.is_compatible(
        classifier_name=experiment_metadata["compatibility"]["classifier_name"],
        classifier_sha256=classifier_hash,
        feature_schema_hash=stable_feature_schema_hash(FEATURE_COLUMNS),
        configuration_version=experiment_metadata["compatibility"]["configuration_version"],
    ):
        raise ValueError("Calibration artifact compatibility check failed.")

    _, actual = prepare_targets(test, float(experiment_metadata["threshold_value_pct"]))
    features = select_features(test)
    raw = np.asarray(classifier.predict_proba(features), dtype=float)
    calibrated = np.asarray(calibrator.transform(raw), dtype=float)
    if len(raw) != len(test) or len(calibrated) != len(test):
        raise ValueError("Prediction count does not match the test row count.")
    actual_values = actual.to_numpy(dtype=int)
    raw_predicted = np.asarray(classifier.classes_)[raw.argmax(axis=1)].astype(int)
    calibrated_predicted = np.asarray(LABELS)[calibrated.argmax(axis=1)].astype(int)

    output.mkdir(parents=True, exist_ok=False)
    predictions = pd.DataFrame({
        "datetime_et": test["datetime_et"].astype(str).to_numpy(),
        "actual_label": actual_values,
        "raw_down": raw[:, 0], "raw_neutral": raw[:, 1], "raw_up": raw[:, 2],
        "calibrated_down": calibrated[:, 0], "calibrated_neutral": calibrated[:, 1], "calibrated_up": calibrated[:, 2],
        "raw_predicted_label": raw_predicted,
        "calibrated_predicted_label": calibrated_predicted,
    })
    predictions.to_csv(output / "predictions.csv", index=False)

    raw_metrics = {**calibration_metrics(raw, actual_values), **_classification_metrics(actual_values, raw_predicted)}
    calibrated_metrics = {**calibration_metrics(calibrated, actual_values), **_classification_metrics(actual_values, calibrated_predicted)}
    input_hash = sha256_file(str(Path(dataset_path)))
    report = {
        "evaluation": "final_holdout_20261114",
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(dataset_path), "sha256": input_hash, "validated_rows": len(data)},
        "artifacts": {
            "experiment": str(experiment),
            "classifier": {"path": str(classifier_path), "sha256": classifier_hash},
            "sigmoid_calibrator": {"path": str(calibrator_path), "sha256": calibrator_hash},
        },
        "configuration": {
            "feature_names": list(FEATURE_COLUMNS),
            "feature_schema_hash": stable_feature_schema_hash(FEATURE_COLUMNS),
            "class_order": list(LABELS),
            "class_names": LABEL_NAMES,
            "threshold_value_pct": float(experiment_metadata["threshold_value_pct"]),
            "forecast_horizon_minutes": 15,
            "forecast_horizon_bars": 3,
            "purge_rows": split_metadata["purge_rows_at_boundaries"],
        },
        "split_metadata": splits["metadata"],
        "holdout_integrity": {
            "experiment_test_used": experiment_metadata["test_used"],
            "default_metadata_test_evaluated_before_run": False,
            "rows_evaluated": len(test),
            "rows_excluded": 0,
            "missing_predictions": 0,
            "invalid_probability_rows": 0,
            "class_order_mismatch": False,
            "test_used_for_selection_or_calibration": False,
        },
        "class_distribution": _distribution(actual_values),
        "predicted_class_distribution": {
            "raw": _distribution(raw_predicted),
            "calibrated": _distribution(calibrated_predicted),
        },
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        "reliability_interpretation": "Calibration quality is compared with log loss, Brier scores, and ECE. Changes in probability quality do not establish improved directional accuracy or trading profitability.",
        "reproduction_command": "python -c \"from src.experiments.final_holdout_evaluation import run_final_holdout_evaluation; run_final_holdout_evaluation('saved_models/experiments/calibration_20261114_v1', 'reports/final_holdout_20261114')\"",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_report(output / "report.md", report)
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    raw = report["raw"]
    calibrated = report["calibrated"]
    lines = [
        "# SilverAI final holdout evaluation",
        "",
        "This is a historical evaluation of frozen `20261114_v1` artifacts. It is not the official five-day live-market test.",
        "",
        f"- Dataset rows evaluated: **{report['holdout_integrity']['rows_evaluated']}**",
        f"- Dataset SHA-256: `{report['dataset']['sha256']}`",
        f"- Classifier SHA-256: `{report['artifacts']['classifier']['sha256']}`",
        f"- Sigmoid calibrator SHA-256: `{report['artifacts']['sigmoid_calibrator']['sha256']}`",
        "- Test rows excluded: **0**",
        "- Test rows used for fitting, selection, or calibration: **0**",
        f"- Actual class distribution: `{json.dumps(report['class_distribution'], sort_keys=True)}`",
        f"- Raw predicted class distribution: `{json.dumps(report['predicted_class_distribution']['raw'], sort_keys=True)}`",
        f"- Calibrated predicted class distribution: `{json.dumps(report['predicted_class_distribution']['calibrated'], sort_keys=True)}`",
        "",
        "## Metrics",
        "",
        "| Metric | Raw | Calibrated |",
        "|---|---:|---:|",
        f"| Multiclass log loss | {raw['multiclass_log_loss']:.8f} | {calibrated['multiclass_log_loss']:.8f} |",
        f"| Mean Brier | {raw['brier_mean_one_vs_rest']:.8f} | {calibrated['brier_mean_one_vs_rest']:.8f} |",
        f"| ECE | {raw['expected_calibration_error']:.8f} | {calibrated['expected_calibration_error']:.8f} |",
        f"| Accuracy | {raw['accuracy']:.8f} | {calibrated['accuracy']:.8f} |",
        f"| Balanced accuracy | {raw['balanced_accuracy']:.8f} | {calibrated['balanced_accuracy']:.8f} |",
        "",
        "## Per-class results",
        "",
        "| Class | Raw precision | Raw recall | Raw F1 | Calibrated precision | Calibrated recall | Calibrated F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("DOWN", "NEUTRAL", "UP"):
        r = raw["per_class"][label]; c = calibrated["per_class"][label]
        lines.append(f"| {label} | {r['precision']:.8f} | {r['recall']:.8f} | {r['f1']:.8f} | {c['precision']:.8f} | {c['recall']:.8f} | {c['f1']:.8f} |")
    lines.extend(["", "## Confusion matrices", "", "Raw:", "", "```text", json.dumps(raw["confusion_matrix"]), "```", "", "Calibrated:", "", "```text", json.dumps(calibrated["confusion_matrix"]), "```", "", "## Reproduction", "", f"`{report['reproduction_command']}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["run_final_holdout_evaluation"]

"""Reproducible, non-destructive calibration experiment for SilverAI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import CONFIG
from src.data.loader import load_modeling_data
from src.features.technical_indicators import FEATURE_COLUMNS
from src.models.calibration import (
    ChronologicalProbabilityCalibrator,
    calibration_metrics,
    sha256_file,
    stable_feature_schema_hash,
)
from src.models.train import (
    build_classification_models,
    prepare_targets,
    prepare_training_data,
    select_features,
    split_chronologically,
    calibrate_classification_threshold,
)


def run_calibration_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Train a separate classifier/calibrator pair without touching production artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = prepare_training_data(load_modeling_data())
    splits = split_chronologically(data)
    train = splits["train"]
    validation = splits["validation"]
    threshold = calibrate_classification_threshold(train)
    y_train = prepare_targets(train, threshold)[1]
    y_validation = prepare_targets(validation, threshold)[1]

    model = build_classification_models()["hist_gradient_boosting_classifier"]
    calibration_cut = int(len(train) * (1 - CONFIG.calibration_fraction))
    prefix = train.iloc[:calibration_cut]
    suffix = train.iloc[calibration_cut:]
    prefix_model = build_classification_models()["hist_gradient_boosting_classifier"]
    prefix_model.fit(select_features(prefix), prepare_targets(prefix, threshold)[1])
    suffix_probabilities = prefix_model.predict_proba(select_features(suffix))
    suffix_labels = prepare_targets(suffix, threshold)[1].to_numpy()

    model.fit(select_features(train), y_train)
    raw_validation = model.predict_proba(select_features(validation))
    raw_metrics = calibration_metrics(raw_validation, y_validation.to_numpy())
    candidates: dict[str, ChronologicalProbabilityCalibrator] = {}
    candidate_metrics: dict[str, dict[str, float]] = {}
    for method in ("sigmoid", "isotonic"):
        candidate = ChronologicalProbabilityCalibrator().fit(
            suffix_probabilities,
            suffix_labels,
            calibration_fraction=1.0 - 1e-12,
            method=method,
        )
        if candidate.status.startswith("available"):
            candidates[method] = candidate
            candidate_metrics[method] = calibration_metrics(
                candidate.transform(raw_validation), y_validation.to_numpy()
            )

    selected_method = min(
        candidate_metrics,
        key=lambda method: candidate_metrics[method]["multiclass_log_loss"],
        default=None,
    )
    if selected_method is None:
        raise RuntimeError("No calibration method met the class-support requirements.")

    classifier_path = output / "classifier.joblib"
    calibrator_path = output / "calibrator.joblib"
    joblib.dump(model, classifier_path)
    calibrator = candidates[selected_method]
    calibrator.classifier_name = "hist_gradient_boosting_classifier"
    calibrator.classifier_sha256 = sha256_file(str(classifier_path))
    calibrator.feature_schema_hash = stable_feature_schema_hash(FEATURE_COLUMNS)
    calibrator.configuration_version = CONFIG.version
    calibrator.save(str(calibrator_path))

    result = {
        "experiment": "calibration_v1",
        "dataset": "silver_modeling_data.csv",
        "dataset_rows_after_validation": len(data),
        "feature_names": list(FEATURE_COLUMNS),
        "classification_labels": {"-1": "DOWN", "0": "NEUTRAL", "1": "UP"},
        "threshold_value_pct": threshold,
        "split_metadata": splits["metadata"],
        "test_used": False,
        "calibration_suffix_rows": len(suffix),
        "candidate_methods": sorted(candidate_metrics),
        "selected_method": selected_method,
        "raw_validation_metrics": raw_metrics,
        "calibrated_validation_metrics": candidate_metrics[selected_method],
        "candidate_validation_metrics": candidate_metrics,
        "artifact_paths": {"classifier": str(classifier_path), "calibrator": str(calibrator_path)},
        "artifact_sha256": {
            "classifier": sha256_file(str(classifier_path)),
            "calibrator": sha256_file(str(calibrator_path)),
        },
        "compatibility": {
            "classifier_name": calibrator.classifier_name,
            "feature_schema_hash": calibrator.feature_schema_hash,
            "configuration_version": calibrator.configuration_version,
        },
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


__all__ = ["run_calibration_experiment"]

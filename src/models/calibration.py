"""Leakage-aware multiclass probability calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class ChronologicalProbabilityCalibrator:
    """Calibrate a classifier's probabilities on a suffix of training data.

    The caller supplies predictions generated without fitting on the
    calibration suffix.  No validation or test rows are accepted here.
    """

    model: LogisticRegression | None = None
    isotonic_models: list[IsotonicRegression] | None = None
    method: str = "sigmoid"
    status: str = "unavailable"
    classifier_name: str | None = None
    classifier_sha256: str | None = None
    feature_schema_hash: str | None = None
    configuration_version: str | None = None

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        calibration_fraction: float = 0.2,
        method: str = "sigmoid",
    ):
        probabilities = _validate_probabilities(probabilities)
        labels = _normalize_labels(labels)
        if len(labels) != len(probabilities) or len(labels) < 10:
            raise ValueError("Calibration probabilities and labels must have at least 10 aligned rows.")
        if not 0 < calibration_fraction < 1:
            raise ValueError("calibration_fraction must be between 0 and 1.")
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("Calibration method must be 'sigmoid' or 'isotonic'.")
        start = int(len(labels) * (1 - calibration_fraction))
        calibration_probabilities = probabilities[start:]
        calibration_labels = labels[start:]
        if len(np.unique(calibration_labels)) < 2:
            self.status = "unavailable_insufficient_classes"
            return self
        self.method = method
        if method == "sigmoid":
            self.model = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
            self.model.fit(np.log(np.clip(calibration_probabilities, 1e-8, 1.0)), calibration_labels)
        else:
            if len(calibration_labels) < 30:
                self.status = "unavailable_isotonic_sample_size"
                return self
            self.isotonic_models = []
            for label in range(3):
                binary_labels = (calibration_labels == label).astype(float)
                if len(np.unique(binary_labels)) < 2:
                    self.status = "unavailable_isotonic_class_support"
                    self.isotonic_models = None
                    return self
                model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                model.fit(calibration_probabilities[:, label], binary_labels)
                self.isotonic_models.append(model)
        self.status = "available_train_suffix" if method == "sigmoid" else "available_train_suffix_isotonic"
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = _validate_probabilities(probabilities)
        if self.model is None and not self.isotonic_models:
            raise ValueError("A fitted probability calibrator is not available.")
        if self.method == "isotonic":
            calibrated = np.column_stack([
                model.predict(values[:, label]) for label, model in enumerate(self.isotonic_models or [])
            ])
            calibrated = calibrated / np.clip(calibrated.sum(axis=1, keepdims=True), 1e-12, None)
        else:
            calibrated = np.zeros_like(values)
            for column, label in enumerate(self.model.classes_):
                calibrated[:, int(label)] = self.model.predict_proba(
                    np.log(np.clip(values, 1e-8, 1.0))
                )[:, column]
        return _validate_probabilities(calibrated)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    def is_compatible(
        self,
        *,
        classifier_name: str,
        classifier_sha256: str,
        feature_schema_hash: str,
        configuration_version: str,
    ) -> bool:
        return (
            self.status.startswith("available")
            and self.classifier_name == classifier_name
            and self.classifier_sha256 == classifier_sha256
            and self.feature_schema_hash == feature_schema_hash
            and self.configuration_version == configuration_version
        )

    @classmethod
    def load(cls, path: str) -> "ChronologicalProbabilityCalibrator":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise ValueError("Calibration artifact has an unexpected type.")
        return value


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    values = _validate_probabilities(probabilities)
    labels = _normalize_labels(labels)
    metrics = {
        "multiclass_log_loss": float(log_loss(labels, values, labels=[0, 1, 2])),
        "brier_mean_one_vs_rest": float(np.mean([
            brier_score_loss(labels == label, values[:, label]) for label in range(values.shape[1])
        ])),
        "expected_calibration_error": expected_calibration_error(values, labels),
    }
    for label in range(values.shape[1]):
        metrics[f"brier_class_{label}"] = float(brier_score_loss(labels == label, values[:, label]))
    return metrics


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """Return multiclass top-confidence ECE using equal-width probability bins."""

    values = _validate_probabilities(probabilities)
    normalized = _normalize_labels(labels)
    if bins < 2:
        raise ValueError("ECE requires at least two bins.")
    confidence = values.max(axis=1)
    predictions = values.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            (confidence < edges[index + 1]) if index < bins - 1 else (confidence <= edges[index + 1])
        )
        if mask.any():
            accuracy = np.mean(predictions[mask] == normalized[mask])
            error += float(mask.mean()) * abs(float(accuracy) - float(confidence[mask].mean()))
    return float(error)


def stable_feature_schema_hash(feature_names: list[str] | tuple[str, ...]) -> str:
    encoded = json.dumps(list(feature_names), separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a finite N x 3 array.")
    if (values < 0).any() or not np.isclose(values.sum(axis=1), 1.0, atol=1e-6).all():
        raise ValueError("Each probability row must be nonnegative and sum to one.")
    return values


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=int)
    if not np.isin(values, [-1, 0, 1, 2]).all():
        raise ValueError("Calibration labels must use {-1, 0, 1} or {0, 1, 2}.")
    if -1 in values:
        values = values + 1
    return values


__all__ = ["ChronologicalProbabilityCalibrator", "calibration_metrics", "expected_calibration_error", "sha256_file", "stable_feature_schema_hash"]

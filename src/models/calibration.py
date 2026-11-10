"""Leakage-aware multiclass probability calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class ChronologicalProbabilityCalibrator:
    """Calibrate a classifier's probabilities on a suffix of training data.

    The caller supplies predictions generated without fitting on the
    calibration suffix.  No validation or test rows are accepted here.
    """

    model: LogisticRegression | None = None
    status: str = "unavailable"

    def fit(self, probabilities: np.ndarray, labels: np.ndarray, calibration_fraction: float = 0.2):
        probabilities = _validate_probabilities(probabilities)
        labels = _normalize_labels(labels)
        if len(labels) != len(probabilities) or len(labels) < 10:
            raise ValueError("Calibration probabilities and labels must have at least 10 aligned rows.")
        if not 0 < calibration_fraction < 1:
            raise ValueError("calibration_fraction must be between 0 and 1.")
        start = int(len(labels) * (1 - calibration_fraction))
        calibration_probabilities = probabilities[start:]
        calibration_labels = labels[start:]
        if len(np.unique(calibration_labels)) < 2:
            self.status = "unavailable_insufficient_classes"
            return self
        self.model = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
        self.model.fit(np.log(np.clip(calibration_probabilities, 1e-8, 1.0)), calibration_labels)
        self.status = "available_train_suffix"
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = _validate_probabilities(probabilities)
        if self.model is None:
            raise ValueError("A fitted probability calibrator is not available.")
        calibrated = np.zeros_like(values)
        for column, label in enumerate(self.model.classes_):
            calibrated[:, int(label)] = self.model.predict_proba(
                np.log(np.clip(values, 1e-8, 1.0))
            )[:, column]
        return _validate_probabilities(calibrated)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ChronologicalProbabilityCalibrator":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise ValueError("Calibration artifact has an unexpected type.")
        return value


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    values = _validate_probabilities(probabilities)
    labels = _normalize_labels(labels)
    return {
        "multiclass_log_loss": float(log_loss(labels, values, labels=[0, 1, 2])),
        "brier_mean_one_vs_rest": float(np.mean([
            brier_score_loss(labels == label, values[:, label]) for label in range(values.shape[1])
        ])),
    }


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


__all__ = ["ChronologicalProbabilityCalibrator", "calibration_metrics"]

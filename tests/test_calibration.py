import numpy as np

from src.models.calibration import ChronologicalProbabilityCalibrator, calibration_metrics, expected_calibration_error, stable_feature_schema_hash


def test_calibrator_uses_valid_probabilities_and_returns_distribution():
    probabilities = np.tile([0.2, 0.3, 0.5], (30, 1))
    labels = np.array([-1, 0, 1] * 10)
    calibrator = ChronologicalProbabilityCalibrator().fit(probabilities, labels)
    assert calibrator.status == "available_train_suffix"
    calibrated = calibrator.transform(probabilities)
    assert calibrated.shape == (30, 3)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    metrics = calibration_metrics(calibrated, labels)
    assert {"multiclass_log_loss", "brier_mean_one_vs_rest", "expected_calibration_error"} <= set(metrics)
    assert 0 <= expected_calibration_error(calibrated, labels) <= 1


def test_isotonic_calibrator_is_multiclass_and_normalized():
    probabilities = np.tile([[0.2, 0.5, 0.3]], (60, 1))
    labels = np.array([-1, 0, 1] * 20)
    calibrator = ChronologicalProbabilityCalibrator().fit(probabilities, labels, calibration_fraction=0.5, method="isotonic")
    assert calibrator.status.startswith("available")
    calibrated = calibrator.transform(probabilities)
    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_calibrator_compatibility_requires_model_and_schema_identity():
    probabilities = np.tile([[0.2, 0.3, 0.5]], (30, 1))
    labels = np.array([-1, 0, 1] * 10)
    calibrator = ChronologicalProbabilityCalibrator().fit(probabilities, labels)
    calibrator.classifier_name = "model-v1"
    calibrator.classifier_sha256 = "abc"
    calibrator.feature_schema_hash = stable_feature_schema_hash(["a", "b"])
    calibrator.configuration_version = "config-v1"
    assert calibrator.is_compatible(classifier_name="model-v1", classifier_sha256="abc", feature_schema_hash=stable_feature_schema_hash(["a", "b"]), configuration_version="config-v1")
    assert not calibrator.is_compatible(classifier_name="model-v2", classifier_sha256="abc", feature_schema_hash=stable_feature_schema_hash(["a", "b"]), configuration_version="config-v1")

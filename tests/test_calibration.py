import numpy as np

from src.models.calibration import ChronologicalProbabilityCalibrator, calibration_metrics


def test_calibrator_uses_valid_probabilities_and_returns_distribution():
    probabilities = np.tile([0.2, 0.3, 0.5], (30, 1))
    labels = np.array([-1, 0, 1] * 10)
    calibrator = ChronologicalProbabilityCalibrator().fit(probabilities, labels)
    assert calibrator.status == "available_train_suffix"
    calibrated = calibrator.transform(probabilities)
    assert calibrated.shape == (30, 3)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert set(calibration_metrics(calibrated, labels)) == {"multiclass_log_loss", "brier_mean_one_vs_rest"}

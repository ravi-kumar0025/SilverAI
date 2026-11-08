import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evaluate import (
    evaluate_classification,
    evaluate_regression,
    majority_class_baseline,
    zero_return_baseline,
)
from src.models.train import split_chronologically


def test_classification_metrics_keep_fixed_class_order():
    result = evaluate_classification(
        pd.Series([-1, -1, 1, 1]),
        np.array([-1, 1, 1, 1]),
        "model",
        "test",
    )

    assert result["confusion_matrix"] == [[1, 0, 1], [0, 0, 0], [0, 0, 2]]
    assert result["neutral_f1"] == 0.0
    assert set(result["classification_report"]) >= {"DOWN", "NEUTRAL", "UP"}


def test_regression_metrics_handle_single_sample_r2():
    result = evaluate_regression(
        pd.Series([1.0]), np.array([0.0]), "model", "small"
    )

    assert result["mae"] == 1.0
    assert np.isnan(result["r2"])


def test_baselines_use_expected_definitions():
    classification = majority_class_baseline(
        pd.Series([-1, -1, 1]), pd.Series([-1, 0, 1]), "validation"
    )
    regression = zero_return_baseline(pd.Series([1.0, -1.0]), "validation")

    assert classification["majority_class"] == -1
    assert classification["model"] == "majority_class_baseline"
    assert regression["model"] == "zero_return_baseline"
    assert regression["mae"] == 1.0


def test_split_matches_purged_chronological_policy():
    frame = pd.DataFrame({"datetime_et": pd.date_range("2025-01-01", periods=100)})
    splits = split_chronologically(frame)

    assert [len(splits[name]) for name in ("train", "validation", "test")] == [67, 12, 15]
    assert splits["train"]["datetime_et"].max() < splits["validation"]["datetime_et"].min()
    assert splits["validation"]["datetime_et"].max() < splits["test"]["datetime_et"].min()

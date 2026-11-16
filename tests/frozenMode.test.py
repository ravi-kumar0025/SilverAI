import json

import pytest

from src.evaluation.frozen_mode import build_manifest, write_manifest


def _metadata():
    return {
        "feature_names": ["return_1"],
        "threshold_value_pct": 0.1,
        "prediction_horizon_minutes": 15,
        "prediction_horizon_bars": 3,
        "classification_target": "classification_target",
        "regression_target": "future_return_15m",
    }


def test_frozen_manifest_records_option_b_and_five_trading_days(tmp_path):
    manifest = build_manifest(_metadata(), "hist_gradient_boosting_classifier")
    assert manifest.option == "B_live_web_app"
    assert manifest.evaluation_trading_days == 5
    path = write_manifest(manifest, tmp_path / "frozen.json")
    assert json.loads(path.read_text())["horizon_minutes"] == 15


def test_frozen_manifest_cannot_be_replaced(tmp_path):
    path = tmp_path / "frozen.json"
    write_manifest(build_manifest(_metadata(), "model-a"), path)
    with pytest.raises(FileExistsError):
        write_manifest(build_manifest(_metadata(), "model-b"), path)

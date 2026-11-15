import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.spot_data import build_candles
from download_dukascopy import ensure_output_paths
from src.features.spot_features import SPOT_FEATURE_COLUMNS, generate_spot_features
from src.spot_targets import make_spot_labels
from src.models.spot_predict import load_spot_artifacts, predict_spot


def _minute_frame(n=20):
    timestamps = pd.date_range("2026-01-01 08:00", periods=n, freq="min", tz="UTC")
    close = pd.Series(25.0 + np.arange(n) * 0.01)
    return pd.DataFrame({"timestamp": timestamps, "open": close, "high": close + .01, "low": close - .01, "close": close})


def test_resampler_keeps_only_complete_candles_and_sets_known_at():
    frame = _minute_frame()
    result = build_candles(frame, 5)
    assert len(result) == 4
    assert result.loc[0, "known_at"] == result.loc[0, "timestamp"] + pd.Timedelta(minutes=5)

    incomplete = frame.drop(index=6)
    assert len(build_candles(incomplete, 5)) == 3


def test_downloader_refuses_existing_outputs_without_force(tmp_path):
    output = tmp_path / "dukascopy"
    output.mkdir()
    (output / "xagusd_1m.csv").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ensure_output_paths(output, ["xagusd"])
    ensure_output_paths(output, ["xagusd"], force=True)


def test_spot_feature_schema_and_future_change_do_not_rewrite_past():
    timestamps = pd.date_range("2026-01-01", periods=260, freq="5min", tz="UTC")
    close = pd.Series(25.0 + np.sin(np.arange(260) / 9) * .1 + np.arange(260) * .001)
    frame = pd.DataFrame({"timestamp": timestamps, "silver_open": close, "silver_high": close + .02, "silver_low": close - .02, "silver_close": close, "xauusd_close": close * 72, "eurusd_close": 1.1 + close * 0.0001})
    result = generate_spot_features(frame)
    changed = frame.copy(); changed.loc[259, "silver_close"] = 99
    changed_result = generate_spot_features(changed)
    assert len(SPOT_FEATURE_COLUMNS) == 34
    assert result.loc[200, list(SPOT_FEATURE_COLUMNS)].equals(changed_result.loc[200, list(SPOT_FEATURE_COLUMNS)])
    selected = generate_spot_features(frame, groups=("silver_price", "time_session"))
    assert set(SPOT_FEATURE_COLUMNS).difference(selected.columns) == set(SPOT_FEATURE_COLUMNS) - set((*("ret_1", "ret_5", "ret_15", "ret_60", "range_pct", "body_ratio", "upper_wick", "lower_wick"), "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_overlap", "mins_since_open"))


def test_atr_labels_drop_no_move_rows():
    timestamps = pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC")
    close = pd.Series(25.0 + np.arange(100) * .01)
    frame = pd.DataFrame({"timestamp": timestamps, "silver_close": close, "silver_high": close + .01, "silver_low": close - .01})
    labeled, report = make_spot_labels(frame, horizon_bars=3, k=.3)
    assert report["rows_dropped_no_move_or_warmup"] > 0
    assert set(labeled["label"].unique()).issubset({0.0, 1.0})


def test_atr_labels_require_only_close_high_low_not_open():
    timestamps = pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC")
    close = pd.Series(25.0 + np.arange(100) * .01)
    frame = pd.DataFrame({"timestamp": timestamps, "silver_close": close, "silver_high": close + .01, "silver_low": close - .01})
    labeled, report = make_spot_labels(frame, horizon_bars=3, k=.3)
    assert report["rows_kept"] == len(labeled)


def test_target_labels_raise_clear_error_for_missing_required_price_column():
    timestamps = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC")
    close = pd.Series(25.0 + np.arange(20) * .01)
    frame = pd.DataFrame({"timestamp": timestamps, "silver_close": close, "silver_high": close + .01})
    with pytest.raises(ValueError, match="low=silver_low"):
        make_spot_labels(frame, horizon_bars=3, k=.3)


def test_final_spot_artifact_uses_the_ordered_34_feature_contract():
    pytest.importorskip("lightgbm")
    artifacts = load_spot_artifacts()
    frame = pd.DataFrame({"timestamp": ["2026-01-01T09:30:00+00:00"], **{feature: [0.01] for feature in SPOT_FEATURE_COLUMNS}})
    result = predict_spot(frame, artifacts)
    assert tuple(artifacts["features"]) == SPOT_FEATURE_COLUMNS
    assert result.loc[0, "predicted_direction_name"] in {"UP", "DOWN"}
    assert 0 <= result.loc[0, "calibrated_probability_up"] <= 1

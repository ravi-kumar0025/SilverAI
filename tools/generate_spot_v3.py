"""Generate development-only spot_v3 artifacts and Step 0 tables.

This script intentionally stops before the permanently used original holdout.
It trains/calibrates only on rows before ``SPOT_CONFIG.final_holdout_start``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.market_candle_builder import align_market_data
from src.evaluation.selectivity import selective_metrics
from src.experiments.spot_dukascopy_study import _calibrated, _fit_calibrator, _metrics, _model, _folds
from src.features.market_features import SPOT_FEATURE_COLUMNS, generate_spot_features
from src.models.model_artifact_contract import dump_versioned_artifact, runtime_versions
from src.market_model_config import SPOT_CONFIG
from src.directional_labels import make_spot_labels, make_triple_barrier_labels


OUTPUT = ROOT / "reports" / "spot_v3_20261115"
MODEL_VERSION = "xagusd_5m_60m_atr_binary_lgbm_v3_dev"


def _thresholds(y: np.ndarray, probability: np.ndarray) -> list[dict[str, float | int | None]]:
    return [selective_metrics(y, probability, threshold) for threshold in (0.52, 0.55, 0.60, 0.65, 0.70, 0.75)]


def _corrected_folds(development: pd.DataFrame, horizon: int) -> list[dict[str, object]]:
    rows = []
    for train, calibration, validation in _folds(development, horizon):
        model = _model().fit(train[list(SPOT_FEATURE_COLUMNS)], train.label.astype(int))
        calibration_raw = model.predict_proba(calibration[list(SPOT_FEATURE_COLUMNS)])[:, 1]
        calibrator = _fit_calibrator(calibration_raw, calibration.label.astype(int).to_numpy())
        raw = model.predict_proba(validation[list(SPOT_FEATURE_COLUMNS)])[:, 1]
        calibrated = _calibrated(calibrator, raw)
        labels = validation.label.astype(int).to_numpy()
        rows.append({
            "train_rows": len(train), "calibration_rows": len(calibration), "validation_rows": len(validation),
            "raw": _metrics(labels, raw), "calibrated": _metrics(labels, calibrated),
            "raw_thresholds": _thresholds(labels, raw), "calibrated_thresholds": _thresholds(labels, calibrated),
        })
    return rows


def _old_fold_comparison(old: dict, corrected: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for index, (old_fold, new_fold) in enumerate(zip(old["folds"], corrected), start=1):
        result.append({
            "fold": index,
            "old": {"train_rows": old_fold["train_rows"], "calibration_rows": old_fold["calibration_rows"], "validation_rows": old_fold["validation_rows"], "raw": old_fold["raw"], "calibrated": old_fold["calibrated"]},
            "corrected": new_fold,
        })
    return result


def _timeout_table(features: pd.DataFrame) -> dict[str, object]:
    variants = {}
    for policy in ("sign", "neutral", "drop"):
        labeled, report = make_triple_barrier_labels(features, 12, 0.3, timeout_policy=policy)
        variants[policy] = {
            "report": report,
            "label_counts": {str(value): int((labeled.triple_barrier_label == value).sum()) for value in (0.0, 0.5, 1.0)},
        }
    return variants


def main() -> None:
    versions = runtime_versions()
    if versions["python"] != "3.12" or versions["scikit-learn"] != "1.6.1":
        raise RuntimeError(f"Use the pinned Python 3.12 environment; running {versions}")
    aligned, alignment = align_market_data(SPOT_CONFIG.candle_minutes)
    features = generate_spot_features(aligned, SPOT_CONFIG.feature_groups)
    timestamps = pd.to_datetime(features.timestamp, utc=True)
    development_features = features[timestamps < pd.to_datetime(SPOT_CONFIG.final_holdout_start, utc=True)].reset_index(drop=True)
    labeled, label_report = make_spot_labels(development_features, 12, 0.3)
    labeled = labeled[labeled.cross_market_complete].reset_index(drop=True)
    corrected = _corrected_folds(labeled, 12)
    old_metrics = json.loads((ROOT / "reports" / "spot_dukascopy_20261115" / "metrics.json").read_text(encoding="utf-8"))
    old_selected = old_metrics["candidate_results"]["h12_k0.3"]["core_plus_cross"]

    final_model = _model().fit(labeled[list(SPOT_FEATURE_COLUMNS)], labeled.label.astype(int))
    calibration_size = max(100, int(len(labeled) * 0.1))
    calibration_block = labeled.iloc[-calibration_size:]
    calibration_model = _model().fit(labeled.iloc[:-calibration_size][list(SPOT_FEATURE_COLUMNS)], labeled.iloc[:-calibration_size].label.astype(int))
    calibration_raw = calibration_model.predict_proba(calibration_block[list(SPOT_FEATURE_COLUMNS)])[:, 1]
    calibrator = _fit_calibrator(calibration_raw, calibration_block.label.astype(int).to_numpy())

    OUTPUT.mkdir(parents=True, exist_ok=True)
    dump_versioned_artifact({"model": final_model, "features": list(SPOT_FEATURE_COLUMNS), "instrument": SPOT_CONFIG.instrument, "symbol": SPOT_CONFIG.symbol, "horizon_bars": 12, "k": 0.3, "dataset_id": SPOT_CONFIG.dataset_id, "model_version": MODEL_VERSION}, OUTPUT / "spot_lightgbm.joblib", artifact_kind="spot_classifier")
    dump_versioned_artifact(calibrator, OUTPUT / "spot_sigmoid_calibrator.joblib", artifact_kind="spot_sigmoid_calibrator")

    metrics = {
        "metadata": {"runtime_versions": versions, "dataset_id": SPOT_CONFIG.dataset_id, "model_version": MODEL_VERSION, "holdout_status": "not_evaluated_development_only", "holdout_start": SPOT_CONFIG.final_holdout_start},
        "development": {"rows": len(labeled), "start": str(labeled.timestamp.iloc[0]), "end": str(labeled.timestamp.iloc[-1]), "label_report": label_report, "alignment": alignment},
        "selected_candidate": "h12_k0.3",
        "corrected_folds": corrected,
        "baseline_comparison_old_vs_corrected": _old_fold_comparison(old_selected, corrected),
        "triple_barrier_timeout_breakdown": _timeout_table(development_features),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    report = [
        "# SilverAI spot v3 development-only Step 0",
        "",
        "The original holdout was not read or evaluated. These artifacts use only rows before the configured original holdout start.",
        "",
        f"Development rows: {len(labeled)}; selected candidate: H=12, k=0.3; dataset_id: {SPOT_CONFIG.dataset_id}; model_version: {MODEL_VERSION}.",
        "",
        "See `metrics.json` for old-vs-corrected fold metrics, raw/calibrated threshold coverage, and triple-barrier timeout variants.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT / "model_card.md").write_text("# XAG/USD spot v3 model card\n\nDevelopment-only artifact. No holdout was evaluated.\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "versions": versions, "development_rows": len(labeled), "selected": "h12_k0.3"}, indent=2))


if __name__ == "__main__":
    main()

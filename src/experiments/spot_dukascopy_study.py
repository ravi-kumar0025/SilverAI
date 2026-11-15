"""Leakage-controlled XAG/USD spot study on Dukascopy data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hashlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score

from src.data.spot_data import align_market_data, load_raw_instrument
from src.features.spot_features import FEATURE_GROUPS, SPOT_FEATURE_COLUMNS, generate_spot_features
from src.spot_config import SPOT_CONFIG
from src.spot_targets import make_spot_labels
from src.models.artifact_contract import dump_versioned_artifact, runtime_versions


CORE_FEATURES = tuple(FEATURE_GROUPS["silver_price"] + FEATURE_GROUPS["volatility"] + FEATURE_GROUPS["trend_momentum"] + FEATURE_GROUPS["time_session"])
CROSS_FEATURES = tuple(FEATURE_GROUPS["cross_market"])


def _model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=220, learning_rate=0.03, num_leaves=15,
        min_child_samples=100, colsample_bytree=0.7, subsample=0.7, subsample_freq=1,
        reg_lambda=2.0, reg_alpha=0.05, verbosity=-1, random_state=42, n_jobs=1,
    )


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (probability >= threshold).astype(int)
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "log_loss": float(log_loss(y, np.column_stack([1-probability, probability]), labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "up_rate": float(pred.mean()),
    }


def _folds(frame: pd.DataFrame, horizon: int) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    development = frame.reset_index(drop=True)
    folds: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for fit_end_fraction, calibration_fraction, validation_fraction in ((0.50, 0.10, 0.15), (0.65, 0.10, 0.15), (0.80, 0.05, 0.10)):
        fit_end = int(len(development) * fit_end_fraction)
        calibration_end = min(len(development), fit_end + int(len(development) * calibration_fraction))
        validation_end = min(len(development), calibration_end + int(len(development) * validation_fraction))
        train_end = max(1, fit_end - horizon - SPOT_CONFIG.embargo_bars)
        calibration_start = fit_end + SPOT_CONFIG.purge_bars
        validation_start = calibration_end + SPOT_CONFIG.purge_bars
        train = development.iloc[:train_end].copy()
        calibration = development.iloc[calibration_start:calibration_end].copy()
        validation = development.iloc[validation_start:validation_end].copy()
        if len(train) and len(calibration) and len(validation): folds.append((train, calibration, validation))
    return folds


def _fit_calibrator(probability: np.ndarray, y: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    logits = np.log(np.clip(probability, 1e-6, 1-1e-6) / np.clip(1-probability, 1e-6, 1-1e-6)).reshape(-1, 1)
    return calibrator.fit(logits, y)


def _calibrated(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-6, 1-1e-6) / np.clip(1-probability, 1e-6, 1-1e-6)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _run_candidate(frame: pd.DataFrame, features: tuple[str, ...], horizon: int) -> dict[str, Any]:
    fold_results = []
    for train, calibration, validation in _folds(frame, horizon):
        model = _model().fit(train[list(features)], train.label.astype(int))
        calibration_raw = model.predict_proba(calibration[list(features)])[:, 1]
        calibrator = _fit_calibrator(calibration_raw, calibration.label.astype(int).to_numpy())
        raw = model.predict_proba(validation[list(features)])[:, 1]
        calibrated = _calibrated(calibrator, raw)
        y = validation.label.astype(int).to_numpy()
        shuffled_model = _model().fit(train[list(features)], train.label.sample(frac=1.0, random_state=42).to_numpy())
        shuffled = shuffled_model.predict_proba(validation[list(features)])[:, 1]
        fold_results.append({
            "raw": _metrics(y, raw),
            "calibrated": _metrics(y, calibrated),
            "flip_test": {"accuracy": float(accuracy_score(y, (1.0 - raw >= 0.5).astype(int)))},
            "shuffled_label_test": _metrics(y, shuffled),
            "baselines": {
                "always_up": {"accuracy": float(accuracy_score(y, np.ones_like(y)))},
                "always_down": {"accuracy": float(accuracy_score(y, np.zeros_like(y)))},
                "majority": {"accuracy": float(accuracy_score(y, np.full_like(y, int(train.label.mean() >= 0.5))))},
                "rsi_rule": _rsi_rule(validation),
            },
            "train_rows": len(train), "calibration_rows": len(calibration), "validation_rows": len(validation)
        })
    return {"features": list(features), "feature_count": len(features), "folds": fold_results, "mean_accuracy": float(np.mean([x["calibrated"]["accuracy"] for x in fold_results])), "mean_auc": float(np.mean([x["calibrated"]["auc"] for x in fold_results]))}


def _threshold_table(y: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for threshold in (0.55, 0.60, 0.65, 0.70):
        mask = (probability >= threshold) | (probability <= 1-threshold)
        if not mask.any(): rows.append({"threshold": threshold, "signals": 0, "coverage": 0.0, "accuracy": None}); continue
        pred = (probability[mask] >= 0.5).astype(int)
        rows.append({"threshold": threshold, "signals": int(mask.sum()), "coverage": float(mask.mean()), "accuracy": float(accuracy_score(y[mask], pred))})
    return rows


def _rsi_rule(validation: pd.DataFrame) -> dict[str, Any]:
    """Evaluate RSI<30 => Up and RSI>70 => Down on emitted signals only."""
    rsi = pd.to_numeric(validation["rsi_14"], errors="coerce").to_numpy()
    y = validation.label.astype(int).to_numpy()
    mask = (rsi < 30) | (rsi > 70)
    if not mask.any():
        return {"signals": 0, "coverage": 0.0, "accuracy": None}
    pred = (rsi[mask] < 30).astype(int)
    return {"signals": int(mask.sum()), "coverage": float(mask.mean()), "accuracy": float(accuracy_score(y[mask], pred))}


def run_study(output_dir: str | Path = "reports/spot_dukascopy_20261115") -> dict[str, Any]:
    versions = runtime_versions()
    if versions["python"] != "3.12" or versions["scikit-learn"] != "1.6.1":
        raise RuntimeError(f"Spot artifact regeneration requires Python 3.12 and scikit-learn 1.6.1; running {versions}")
    if SPOT_CONFIG.holdout_status == "used_do_not_reuse":
        raise RuntimeError("The original holdout is permanently closed; configure a newly frozen holdout before running the study.")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    raw_reports = {}; 
    for instrument in ("xagusd", "xauusd", "eurusd"): _, raw_reports[instrument] = load_raw_instrument(instrument)
    aligned, alignment_report = align_market_data(SPOT_CONFIG.candle_minutes)
    features = generate_spot_features(aligned, SPOT_CONFIG.feature_groups)
    candidate_results: dict[str, Any] = {}
    for horizon in SPOT_CONFIG.candidate_horizons_bars:
        for k in SPOT_CONFIG.target_k_values:
            labeled, label_report = make_spot_labels(features, horizon, k)
            labeled = labeled[labeled.cross_market_complete].reset_index(drop=True)
            final_start = pd.to_datetime(SPOT_CONFIG.final_holdout_start, utc=True)
            development = labeled[pd.to_datetime(labeled.timestamp, utc=True) < final_start].reset_index(drop=True)
            key = f"h{horizon}_k{k}"
            candidate_results[key] = {"labels": label_report, "rows_after_alignment": len(labeled), "rows_before_final_holdout": len(development), "core": _run_candidate(development, CORE_FEATURES, horizon), "core_plus_cross": _run_candidate(development, SPOT_FEATURE_COLUMNS, horizon)}

    selected_key = max(candidate_results, key=lambda key: candidate_results[key]["core_plus_cross"]["mean_accuracy"])
    selected = candidate_results[selected_key]
    horizon = int(selected_key.split("_k")[0][1:]); k = float(selected_key.split("_k")[1])
    labeled, _ = make_spot_labels(features, horizon, k); labeled = labeled[labeled.cross_market_complete].reset_index(drop=True)
    final_start = pd.to_datetime(SPOT_CONFIG.final_holdout_start, utc=True)
    final_end = pd.to_datetime(SPOT_CONFIG.final_holdout_end, utc=True)
    timestamps = pd.to_datetime(labeled.timestamp, utc=True)
    development = labeled[timestamps < final_start].reset_index(drop=True)
    holdout = labeled[(timestamps >= final_start) & (timestamps <= final_end)].reset_index(drop=True)
    final_model = _model().fit(development[list(SPOT_FEATURE_COLUMNS)], development.label.astype(int))
    holdout_probability = final_model.predict_proba(holdout[list(SPOT_FEATURE_COLUMNS)])[:, 1]
    holdout_metrics = _metrics(holdout.label.astype(int).to_numpy(), holdout_probability)
    calibration_size = max(100, int(len(development) * 0.1))
    calibration_block = development.iloc[-calibration_size:]
    calibration_model = _model().fit(development.iloc[:-calibration_size][list(SPOT_FEATURE_COLUMNS)], development.iloc[:-calibration_size].label.astype(int))
    calibration_probability = calibration_model.predict_proba(calibration_block[list(SPOT_FEATURE_COLUMNS)])[:, 1]
    calibrator = _fit_calibrator(calibration_probability, calibration_block.label.astype(int).to_numpy())
    calibrated_holdout = _calibrated(calibrator, holdout_probability)
    result = {"metadata": {"runtime_versions": versions, "config_sha256": hashlib.sha256((Path(__file__).resolve().parents[2] / "config.yaml").read_bytes()).hexdigest(), "holdout_status": SPOT_CONFIG.holdout_status, "seed": SPOT_CONFIG.random_seed}, "instrument": SPOT_CONFIG.instrument, "symbol": SPOT_CONFIG.symbol, "source": SPOT_CONFIG.data_source, "price_side": SPOT_CONFIG.price_side, "dataset_id": SPOT_CONFIG.dataset_id, "feature_set_version": SPOT_CONFIG.feature_set_version, "model_version": SPOT_CONFIG.model_version, "raw_data_quality": raw_reports, "alignment": alignment_report, "feature_count": len(SPOT_FEATURE_COLUMNS), "features": list(SPOT_FEATURE_COLUMNS), "core_features": list(CORE_FEATURES), "cross_market_features": list(CROSS_FEATURES), "candidate_results": candidate_results, "selected_candidate": selected_key, "final_holdout": {"start": SPOT_CONFIG.final_holdout_start, "end": SPOT_CONFIG.final_holdout_end, "rows": len(holdout), "raw": holdout_metrics, "calibrated": _metrics(holdout.label.astype(int).to_numpy(), calibrated_holdout), "threshold_table": _threshold_table(holdout.label.astype(int).to_numpy(), calibrated_holdout)}, "label_definition": "log(close[t+H]/close[t]); Up if fwd_ret > k*ATR_pct, Down if below -k*ATR_pct; no-move rows dropped", "leakage_checks": {"holdout_used_for_selection": False, "previous_holdout_used_for_selection": True, "cross_market_fill_limit_bars": 2, "purge_bars": SPOT_CONFIG.purge_bars, "embargo_bars": SPOT_CONFIG.embargo_bars, "candle_known_at": "timestamp + candle interval"}, "baseline_note": "Previous SI results were 3-class and are not comparable to this binary spot study."}
    output.joinpath("metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    dump_versioned_artifact({"model": final_model, "features": list(SPOT_FEATURE_COLUMNS), "instrument": SPOT_CONFIG.instrument, "symbol": SPOT_CONFIG.symbol, "horizon_bars": horizon, "k": k, "dataset_id": SPOT_CONFIG.dataset_id, "model_version": SPOT_CONFIG.model_version}, output / "spot_lightgbm.joblib", artifact_kind="spot_classifier")
    dump_versioned_artifact(calibrator, output / "spot_sigmoid_calibrator.joblib", artifact_kind="spot_sigmoid_calibrator")
    return result


if __name__ == "__main__":
    print(json.dumps(run_study(), indent=2, default=str))


__all__ = ["run_study"]

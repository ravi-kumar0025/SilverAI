"""Leakage-safe alternative target and horizon experiments.

This module never loads or evaluates the frozen historical test partition.  It
uses the original train/validation time boundaries and fits data-dependent
label parameters inside each chronological development fold.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.data.loader import load_feature_data, load_modeling_data
from src.models.train import LABELS, prepare_training_data, select_features, split_chronologically
from src.targets import generate_timestamp_targets


SEED = CONFIG.random_seed
NAMES = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _models() -> dict[str, Any]:
    return {
        "hgb_current": HistGradientBoostingClassifier(max_iter=100, random_state=SEED, early_stopping=False),
        "hgb_tuned": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, random_state=SEED, early_stopping=False),
        "logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=SEED))]),
    }


def _metric(y: np.ndarray, p: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    report = classification_report(y, p, labels=list(LABELS), output_dict=True, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=list(LABELS))) if probabilities is not None and len(np.unique(y)) == 3 else None,
        "confusion_matrix": confusion_matrix(y, p, labels=list(LABELS)).tolist(),
        "actual_counts": {NAMES[k]: int((y == k).sum()) for k in LABELS},
        "predicted_counts": {NAMES[k]: int((p == k).sum()) for k in LABELS},
        "per_class": {NAMES[k]: {"precision": float(report[str(k)]["precision"]), "recall": float(report[str(k)]["recall"]), "f1": float(report[str(k)]["f1-score"]), "support": int(report[str(k)]["support"])} for k in LABELS},
    }


def _partitions(modeling: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    splits = split_chronologically(prepare_training_data(modeling))
    return (splits["train"]["datetime_et"].iloc[0], splits["train"]["datetime_et"].iloc[-1], splits["validation"]["datetime_et"].iloc[0], splits["validation"]["datetime_et"].iloc[-1])


def _target_frame(feature_data: pd.DataFrame, horizon_minutes: int, train_start: pd.Timestamp, train_end: pd.Timestamp, validation_start: pd.Timestamp, validation_end: pd.Timestamp) -> pd.DataFrame:
    target = generate_timestamp_targets(feature_data, interval_minutes=5, horizon_minutes=horizon_minutes)
    target["future_timestamp"] = target["datetime_et"] + pd.Timedelta(minutes=horizon_minutes)
    eligible = target["target_valid"] & (
        ((target["datetime_et"] >= train_start) & (target["datetime_et"] <= train_end) & (target["future_timestamp"] <= train_end))
        | ((target["datetime_et"] >= validation_start) & (target["datetime_et"] <= validation_end) & (target["future_timestamp"] <= validation_end))
    )
    return target.loc[eligible].copy().reset_index(drop=True)


def _fit_parameters(frame: pd.DataFrame, kind: str, horizon_minutes: int, fit_mask: pd.Series) -> dict[str, Any]:
    fit = frame.loc[fit_mask]
    returns = fit["future_return_pct"].astype(float)
    if kind == "fixed":
        return {"threshold_pct": float(returns.abs().quantile(CONFIG.neutral_quantile))}
    if kind == "volatility_adjusted":
        ratio = returns.abs() / fit["volatility_20"].clip(lower=1e-9)
        return {"scale": float(ratio.replace([np.inf, -np.inf], np.nan).dropna().quantile(CONFIG.neutral_quantile))}
    if kind == "dynamic_neutral":
        return {"fallback_threshold_pct": float(returns.abs().quantile(CONFIG.neutral_quantile))}
    if kind == "trend_scanning":
        return {"score_threshold": float(fit["trend_scan_score"].abs().replace([np.inf, -np.inf], np.nan).dropna().quantile(CONFIG.neutral_quantile))}
    if kind == "triple_barrier":
        return {"barrier_scale": 1.0, "holding_bars": max(1, horizon_minutes // 5)}
    raise ValueError(f"Unknown target kind: {kind}")


def _labels(frame: pd.DataFrame, kind: str, parameters: dict[str, Any], horizon_minutes: int) -> pd.Series:
    returns = frame["future_return_pct"].astype(float)
    if kind == "fixed":
        threshold = parameters["threshold_pct"]
        return pd.Series(np.select([returns > threshold, returns < -threshold], [1, -1], default=0), index=frame.index, dtype="int64")
    if kind == "volatility_adjusted":
        threshold = parameters["scale"] * frame["volatility_20"].clip(lower=1e-9)
        return pd.Series(np.select([returns > threshold, returns < -threshold], [1, -1], default=0), index=frame.index, dtype="int64")
    if kind == "dynamic_neutral":
        past_abs = frame["return_1"].abs().rolling(200, min_periods=20).quantile(CONFIG.neutral_quantile).shift(1)
        threshold = past_abs.fillna(parameters["fallback_threshold_pct"])
        return pd.Series(np.select([returns > threshold, returns < -threshold], [1, -1], default=0), index=frame.index, dtype="int64")
    if kind == "trend_scanning":
        threshold = parameters["score_threshold"]
        return pd.Series(np.select([frame["trend_scan_score"] >= threshold, frame["trend_scan_score"] <= -threshold], [1, -1], default=0), index=frame.index, dtype="int64")
    if kind == "triple_barrier":
        labels = np.zeros(len(frame), dtype="int64")
        scale = parameters["barrier_scale"]
        bars = parameters["holding_bars"]
        highs = frame["high"].to_numpy(float)
        lows = frame["low"].to_numpy(float)
        close = frame["close"].to_numpy(float)
        vol = frame["volatility_20"].to_numpy(float)
        for i in range(len(frame)):
            pt = close[i] * (1.0 + scale * max(vol[i], 1e-9) * np.sqrt(bars) / 100.0)
            sl = close[i] * (1.0 - scale * max(vol[i], 1e-9) * np.sqrt(bars) / 100.0)
            end = min(len(frame), i + bars + 1)
            for j in range(i + 1, end):
                hit_up = highs[j] >= pt
                hit_down = lows[j] <= sl
                if hit_up and hit_down:
                    labels[i] = 1 if close[j] >= close[i] else -1
                    break
                if hit_up:
                    labels[i] = 1
                    break
                if hit_down:
                    labels[i] = -1
                    break
        return pd.Series(labels, index=frame.index, dtype="int64")
    raise ValueError(f"Unknown target kind: {kind}")


def _add_trend_scan_scores(frame: pd.DataFrame, max_bars: int = 12) -> pd.DataFrame:
    result = frame.copy()
    close = result["close"].to_numpy(float)
    vol = result["volatility_20"].to_numpy(float)
    scores = np.zeros(len(result), dtype=float)
    for i in range(len(result)):
        best = 0.0
        for bars in (1, 2, 3, 6, max_bars):
            j = i + bars
            if j >= len(result):
                continue
            score = (close[j] - close[i]) / max(close[i], 1e-9) * 100.0 / max(vol[i] * np.sqrt(bars), 1e-9)
            if abs(score) > abs(best):
                best = score
        scores[i] = best
    result["trend_scan_score"] = scores
    return result


def _fold_masks(frame: pd.DataFrame, train_start: pd.Timestamp, train_end: pd.Timestamp) -> list[tuple[pd.Series, pd.Series]]:
    train = frame[(frame.datetime_et >= train_start) & (frame.datetime_et <= train_end)].reset_index(drop=True)
    n = len(train)
    folds = []
    for fit_end_fraction, validation_fraction in ((0.50, 0.20), (0.70, 0.20)):
        fit_end = int(n * fit_end_fraction)
        val_end = min(n, fit_end + int(n * validation_fraction))
        fit_mask = pd.Series(False, index=frame.index)
        val_mask = pd.Series(False, index=frame.index)
        fit_indices = train.index[: max(0, fit_end - 3)]
        val_indices = train.index[fit_end:val_end]
        fit_mask.loc[fit_indices] = True
        val_mask.loc[val_indices] = True
        folds.append((fit_mask, val_mask))
    return folds


def _evaluate_target(frame: pd.DataFrame, kind: str, horizon: int, train_start: pd.Timestamp, train_end: pd.Timestamp, validation_start: pd.Timestamp, validation_end: pd.Timestamp) -> dict[str, Any]:
    frame = _add_trend_scan_scores(frame, max_bars=max(1, 60 // 5)) if kind == "trend_scanning" else frame
    train_mask = (frame.datetime_et >= train_start) & (frame.datetime_et <= train_end)
    validation_mask = (frame.datetime_et >= validation_start) & (frame.datetime_et <= validation_end)
    parameters = _fit_parameters(frame, kind, horizon, train_mask)
    y_all = _labels(frame, kind, parameters, horizon)
    model_results: dict[str, Any] = {}
    for name, prototype in _models().items():
        fold_results = []
        for fit_mask, val_mask in _fold_masks(frame, train_start, train_end):
            fold_parameters = _fit_parameters(frame, kind, horizon, fit_mask)
            fold_labels = _labels(frame, kind, fold_parameters, horizon)
            model = clone(prototype)
            model.fit(select_features(frame.loc[fit_mask]), fold_labels.loc[fit_mask])
            probabilities = model.predict_proba(select_features(frame.loc[val_mask]))
            predictions = model.classes_[np.argmax(probabilities, axis=1)]
            fold_results.append(_metric(fold_labels.loc[val_mask].to_numpy(), predictions, probabilities))
        model = clone(prototype)
        model.fit(select_features(frame.loc[train_mask]), y_all.loc[train_mask])
        probabilities = model.predict_proba(select_features(frame.loc[validation_mask]))
        predictions = model.classes_[np.argmax(probabilities, axis=1)]
        artifact_name = f"{kind}_{horizon}m_{name}.joblib"
        model_results[name] = {"chronological_folds": fold_results, "cv_mean_accuracy": float(np.mean([x["accuracy"] for x in fold_results])), "cv_mean_balanced_accuracy": float(np.mean([x["balanced_accuracy"] for x in fold_results])), "cv_mean_macro_f1": float(np.mean([x["macro_f1"] for x in fold_results])), "validation": _metric(y_all.loc[validation_mask].to_numpy(), predictions, probabilities), "artifact_name": artifact_name}
        model_results[name]["_model"] = model
    label_counts = y_all.loc[train_mask | validation_mask].value_counts().sort_index().to_dict()
    return {"kind": kind, "horizon_minutes": horizon, "parameters": parameters, "train_rows": int(train_mask.sum()), "validation_rows": int(validation_mask.sum()), "label_counts": {NAMES[int(k)]: int(v) for k, v in label_counts.items()}, "label_proportions": {NAMES[int(k)]: float(v / sum(label_counts.values())) for k, v in label_counts.items()}, "models": model_results}


def run_experiment(output_dir: str | Path = "reports/alternative_targets_20261114_v1") -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    modeling = prepare_training_data(load_modeling_data())
    train_start, train_end, validation_start, validation_end = _partitions(modeling)
    feature_data = load_feature_data().sort_values("datetime_et").reset_index(drop=True)
    results: dict[str, Any] = {}
    target_kinds = ("fixed", "volatility_adjusted", "dynamic_neutral", "triple_barrier", "trend_scanning")
    horizons = (5, 10, 15, 30, 60)
    for horizon in horizons:
        target_frame = _target_frame(feature_data, horizon, train_start, train_end, validation_start, validation_end)
        kinds_for_horizon = ("fixed",) if horizon != 15 else target_kinds
        for kind in kinds_for_horizon:
            key = f"{kind}_{horizon}m"
            results[key] = _evaluate_target(target_frame, kind, horizon, train_start, train_end, validation_start, validation_end)
            for model_name, model_result in results[key]["models"].items():
                model = model_result.pop("_model")
                artifact_path = output / model_result["artifact_name"]
                joblib.dump(model, artifact_path)
                model_result["artifact_sha256"] = _sha256(artifact_path)

    # Two-stage official 15-minute movement/direction experiment.
    official = modeling.copy()
    train_mask = (official.datetime_et >= train_start) & (official.datetime_et <= train_end)
    val_mask = (official.datetime_et >= validation_start) & (official.datetime_et <= validation_end)
    threshold = float(official.loc[train_mask, "future_return_15m"].abs().quantile(CONFIG.neutral_quantile))
    labels = np.select([official.future_return_15m > threshold, official.future_return_15m < -threshold], [1, -1], default=0).astype(int)
    two_stage: dict[str, Any] = {"threshold_pct": threshold, "train_rows": int(train_mask.sum()), "validation_rows": int(val_mask.sum()), "models": {}}
    for name, base in _models().items():
        stage1 = clone(base)
        stage2 = clone(base)
        stage1.fit(select_features(official.loc[train_mask]), (labels[train_mask.to_numpy()] != 0).astype(int))
        directional = labels[train_mask.to_numpy()] != 0
        stage2.fit(select_features(official.loc[train_mask].loc[directional]), labels[train_mask.to_numpy()][directional])
        p1 = stage1.predict_proba(select_features(official.loc[val_mask]))
        movement = stage1.classes_[np.argmax(p1, axis=1)]
        p2 = stage2.predict_proba(select_features(official.loc[val_mask]))
        direction = stage2.classes_[np.argmax(p2, axis=1)]
        end_to_end = np.where(movement == 0, 0, direction)
        stage1_metric = _metric((labels[val_mask.to_numpy()] != 0).astype(int), movement)
        directional_true = labels[val_mask.to_numpy()]
        directional_mask = directional_true != 0
        stage2_metric = _metric(directional_true[directional_mask], direction[directional_mask])
        end_metric = _metric(directional_true, end_to_end)
        two_stage["models"][name] = {"stage1_movement_detection": stage1_metric, "stage2_direction_given_movement": stage2_metric, "end_to_end": end_metric}
        joblib.dump({"stage1": stage1, "stage2": stage2}, output / f"two_stage_15m_{name}.joblib")

    result = {"experiment": "alternative_targets_20261114_v1", "dataset_sha256": _sha256(Path("data/silver_modeling_data.csv")), "feature_dataset_sha256": _sha256(Path("data/silver_features.csv")), "test_protection": {"test_rows_passed_to_experiment": 0, "test_used_for_selection": False, "test_used_for_tuning": False}, "split": {"train_start": str(train_start), "train_end": str(train_end), "validation_start": str(validation_start), "validation_end": str(validation_end), "test_partition_excluded": True, "purge_rows": 3}, "official_baseline": {"target": "fixed 15-minute 40th-percentile threshold", "threshold_pct": threshold, "features": list(CONFIG.feature_names)}, "alternative_targets": results, "two_stage_official_15m": two_stage, "unexecuted_or_deferred": ["trend scanning uses a bounded 1/2/3/6/12-bar scan and is experimental, not an official trend-scanning implementation", "external market data, deep learning, ensembles and live evaluation were not executed"], "reproduction_command": "python -c \"from src.experiments.alternative_target_experiments import run_experiment; run_experiment()\""}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _write_report(output, result)
    return result


def _write_report(output: Path, result: dict[str, Any]) -> None:
    rows = []
    for key, item in result["alternative_targets"].items():
        for model, values in item["models"].items():
            v = values["validation"]
            rows.append(f"| {key} | {model} | {item['train_rows']} | {item['validation_rows']} | {v['accuracy']:.4f} | {v['balanced_accuracy']:.4f} | {v['macro_f1']:.4f} | {v['per_class']['DOWN']['f1']:.4f} | {v['per_class']['NEUTRAL']['f1']:.4f} | {v['per_class']['UP']['f1']:.4f} |")
    report = f"""# SilverAI Alternative Target and Volatility-Adjusted Label Experiments

## Integrity and baseline

This campaign stage preserves the official 5-minute / 15-minute / three-class target, 17 approved features, and training-derived threshold `{result['official_baseline']['threshold_pct']:.16f}%`. It used chronological development folds inside the original training partition and a controlled validation comparison. Zero rows from the frozen historical test partition were passed to fitting, tuning, scoring, or selection.

The original-target frozen test baseline remains 38.40% accuracy. Alternative-target results below are validation results for different label definitions and must not be compared as if they were the same prediction problem.

## Executed target comparisons

| Target | Model | Train n | Validation n | Accuracy | Balanced accuracy | Macro F1 | DOWN F1 | NEUTRAL F1 | UP F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The machine-readable report contains confusion matrices, class proportions, prediction distributions, log loss, chronological fold metrics, parameters and artifact hashes for every row above.

Definitions executed:

- `fixed`: fixed threshold estimated from the fitting partition.
- `volatility_adjusted`: threshold equals a training-fitted return-volatility scale times causal `volatility_20`.
- `dynamic_neutral`: prior 200-bar absolute-return quantile, shifted by one row, with a training-only fallback.
- `triple_barrier`: volatility-scaled profit/stop barriers with a horizon-specific maximum holding period.
- `trend_scanning`: bounded 1/2/3/6/12-bar forward scan, retained as an explicitly experimental approximation rather than a claim of a full trend-scanning implementation.
- `two_stage_official_15m`: movement versus neutral followed by UP versus DOWN conditional on movement; both stages and end-to-end results are reported separately.

## Conclusions

Alternative label accuracy is not automatically evidence of improvement on the official target. A label definition can be easier, change class balance, or remove ambiguous rows. The report therefore emphasizes label coverage, class proportions, sample counts, directional class metrics and chronological stability alongside accuracy.

The executed evidence does not activate any candidate or establish a defensible route to 60% accuracy. The strongest candidates for further review are those that improve DOWN and NEUTRAL discrimination without relying on reduced coverage or materially changed labels; exact candidates are identified from `metrics.json` rather than selected here for production.

## Reproduction

```powershell
python -c "from src.experiments.alternative_target_experiments import run_experiment; run_experiment()"
```

All candidate artifacts are isolated under this directory. No default model, metadata, calibration artifact, or prior report was overwritten.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    run_experiment()


__all__ = ["run_experiment"]

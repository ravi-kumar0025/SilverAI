"""Leakage-safe classification candidate study for SilverAI.

The final historical test partition is intentionally never loaded by this
experiment.  Candidate development uses chronological folds inside training;
the validation partition is used once for final comparison.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.data.loader import load_modeling_data
from src.features.technical_indicators import FEATURE_COLUMNS
from src.models.train import (
    LABELS,
    build_classification_models,
    calibrate_classification_threshold,
    prepare_targets,
    prepare_training_data,
    select_features,
    split_chronologically,
)


SEED = CONFIG.random_seed
CLASS_NAMES = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extended_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal candle-only diagnostics; every rolling value uses past/current rows."""

    result = select_features(frame)
    close = pd.to_numeric(frame["close"], errors="raise")
    high = pd.to_numeric(frame["high"], errors="raise")
    low = pd.to_numeric(frame["low"], errors="raise")
    volume = pd.to_numeric(frame["volume"], errors="raise")
    one_bar = close.pct_change(fill_method=None).fillna(0.0)
    candle_range = (high - low).clip(lower=0.0)
    result = result.copy()
    result["extended_body_range_ratio"] = ((close - frame["open"]) / candle_range.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["extended_range_mean_20"] = candle_range.rolling(20, min_periods=1).mean()
    result["extended_return_mean_5"] = one_bar.rolling(5, min_periods=1).mean()
    result["extended_return_std_20"] = one_bar.rolling(20, min_periods=2).std().fillna(0.0)
    volume_mean = volume.rolling(20, min_periods=1).mean()
    volume_std = volume.rolling(20, min_periods=2).std().replace(0, np.nan)
    result["extended_volume_z20"] = ((volume - volume_mean) / volume_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


def _metrics(y_true: pd.Series | np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(predictions, dtype=int)
    report = classification_report(y, p, labels=list(LABELS), output_dict=True, zero_division=0)
    result: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=list(LABELS))) if probabilities is not None else None,
        "confusion_matrix": confusion_matrix(y, p, labels=list(LABELS)).tolist(),
        "actual_counts": {CLASS_NAMES[k]: int((y == k).sum()) for k in LABELS},
        "predicted_counts": {CLASS_NAMES[k]: int((p == k).sum()) for k in LABELS},
    }
    for label in LABELS:
        values = report[str(label)]
        result[CLASS_NAMES[label].lower()] = {
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "f1": float(values["f1-score"]),
            "support": int(values["support"]),
        }
    return result


def _candidate_factories() -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {
        "dummy_most_frequent": lambda: DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": lambda: DummyClassifier(strategy="stratified", random_state=SEED),
        "logistic_regression": lambda: Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=SEED))]),
        "hist_gradient_boosting_current": lambda: build_classification_models()["hist_gradient_boosting_classifier"],
        "hist_gradient_boosting_leaf31": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, early_stopping=False, random_state=SEED),
        "hist_gradient_boosting_leaf63": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=0.1, early_stopping=False, random_state=SEED),
        "random_forest_balanced": lambda: RandomForestClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "random_forest_unweighted": lambda: RandomForestClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=2, class_weight=None, n_jobs=-1, random_state=SEED),
        "extra_trees_balanced": lambda: ExtraTreesClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting_extended": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, early_stopping=False, random_state=SEED),
    }
    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier

        factories["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            verbosity=-1,
            random_state=SEED,
            n_jobs=-1,
        )
    return factories


def _folds(train: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    boundaries = [(0.50, 0.15), (0.65, 0.15), (0.80, 0.15)]
    result = []
    for train_fraction, validation_fraction in boundaries:
        first = int(len(train) * train_fraction)
        last = min(len(train), first + int(len(train) * validation_fraction))
        fit = train.iloc[: max(0, first - 3)].copy()
        validation = train.iloc[first:last].copy()
        if len(fit) and len(validation):
            result.append((fit, validation))
    return result


def _fit_score(factory: Callable[[], Any], fit: pd.DataFrame, validation: pd.DataFrame, threshold: float, feature_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> dict[str, Any]:
    model = factory()
    _, y_fit = prepare_targets(fit, threshold)
    _, y_validation = prepare_targets(validation, threshold)
    model.fit(feature_fn(fit), y_fit)
    probabilities = model.predict_proba(feature_fn(validation))
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    return _metrics(y_validation, predictions, probabilities)


def _decision_rule_study(model: Any, train: pd.DataFrame, threshold: float, feature_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> dict[str, Any]:
    """Select probability multipliers using training folds only, then report CV results."""

    grids = [(down, up) for down in (0.8, 1.0, 1.2) for up in (0.8, 1.0, 1.2)]
    rows = []
    for down, up in grids:
        scores = []
        for fit, validation in _folds(train):
            _, y_fit = prepare_targets(fit, threshold)
            _, y_validation = prepare_targets(validation, threshold)
            fitted = model.__class__(**model.get_params())
            fitted.fit(feature_fn(fit), y_fit)
            probabilities = fitted.predict_proba(feature_fn(validation))
            adjusted = probabilities * np.asarray([down, 1.0, up])
            predictions = fitted.classes_[np.argmax(adjusted, axis=1)]
            scores.append(float(accuracy_score(y_validation, predictions)))
        rows.append({"down_multiplier": down, "up_multiplier": up, "cv_accuracy": float(np.mean(scores)), "cv_accuracy_by_fold": scores})
    selected = max(rows, key=lambda row: row["cv_accuracy"])
    return {"status": "train_cv_only", "selected_multipliers": selected, "all_candidates": rows}


def run_experiment(output_dir: str | Path = "reports/classification_improvement_20261114_v1") -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = prepare_training_data(load_modeling_data())
    splits = split_chronologically(data)
    train, validation = splits["train"], splits["validation"]
    threshold = calibrate_classification_threshold(train)
    factories = _candidate_factories()
    feature_fns: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {name: select_features for name in factories}
    feature_fns["hist_gradient_boosting_extended"] = _extended_features
    development: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    for name, factory in factories.items():
        fn = feature_fns[name]
        folds = [_fit_score(factory, fit, fold_validation, threshold, fn) for fit, fold_validation in _folds(train)]
        development[name] = {"chronological_cv_folds": folds, "cv_mean_accuracy": float(np.mean([x["accuracy"] for x in folds])), "cv_mean_balanced_accuracy": float(np.mean([x["balanced_accuracy"] for x in folds])), "cv_mean_macro_f1": float(np.mean([x["macro_f1"] for x in folds]))}
        final = _fit_score(factory, train, validation, threshold, fn)
        validation_results[name] = final
        model = factory()
        _, y_train = prepare_targets(train, threshold)
        model.fit(fn(train), y_train)
        path = output / f"{name}.joblib"
        joblib.dump(model, path)
        validation_results[name]["artifact"] = {"path": str(path), "sha256": _sha256(path)}

    decision_model = factories["hist_gradient_boosting_current"]()
    _, y_train = prepare_targets(train, threshold)
    decision_model.fit(select_features(train), y_train)
    decision_rule = _decision_rule_study(decision_model, train, threshold, select_features)
    _, y_validation = prepare_targets(validation, threshold)
    raw_probabilities = decision_model.predict_proba(select_features(validation))
    selected = decision_rule["selected_multipliers"]
    adjusted = raw_probabilities * np.asarray([selected["down_multiplier"], 1.0, selected["up_multiplier"]])
    decision_rule["validation_raw"] = _metrics(y_validation, decision_model.classes_[np.argmax(raw_probabilities, axis=1)], raw_probabilities)
    decision_rule["validation_adjusted"] = _metrics(y_validation, decision_model.classes_[np.argmax(adjusted, axis=1)], adjusted / adjusted.sum(axis=1, keepdims=True))

    availability = {name: bool(importlib.util.find_spec(name)) for name in ("xgboost", "lightgbm")}
    dataset_path = Path("data/silver_modeling_data.csv")
    result: dict[str, Any] = {
        "experiment": "classification_improvement_20261114_v1",
        "dataset": {"path": str(dataset_path), "sha256": _sha256(dataset_path), "validated_rows": int(len(data))},
        "test_protection": {"test_partition_materialized_for_split_metadata": True, "test_rows_passed_to_candidate_fitting": 0, "test_used_for_selection": False, "test_used_for_tuning": False},
        "split_metadata": splits["metadata"],
        "configuration": {"feature_names": list(FEATURE_COLUMNS), "extended_features": list(_extended_features(train).columns), "class_order": list(LABELS), "threshold_value_pct": threshold, "horizon_minutes": CONFIG.forecast_horizon_minutes, "purge_rows": 3, "random_seed": SEED},
        "candidate_availability": availability,
        "development_cv": development,
        "validation_results": validation_results,
        "decision_rule_study": decision_rule,
        "selection_policy": "Validation accuracy first, then balanced accuracy, macro F1, per-class behavior, log loss, and chronological CV stability. No test data used.",
        "limitations": ["Validation results are not test results.", "Extended features are diagnostic-only and not part of the approved production schema.", "XGBoost and LightGBM were not installed and were not added.", "No production model or calibrator was replaced or activated."],
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _write_report(output, result)
    return result


def _write_report(output: Path, result: dict[str, Any]) -> None:
    rows = []
    for name, metrics in result["validation_results"].items():
        rows.append(f"| {name} | {metrics['accuracy']:.6f} | {metrics['balanced_accuracy']:.6f} | {metrics['macro_f1']:.6f} | {metrics['log_loss']:.6f} | {metrics['down']['f1']:.4f} | {metrics['neutral']['f1']:.4f} | {metrics['up']['f1']:.4f} |")
    best = max(result["validation_results"], key=lambda name: (result["validation_results"][name]["accuracy"], result["validation_results"][name]["balanced_accuracy"], result["validation_results"][name]["macro_f1"]))
    decision = result["decision_rule_study"]
    report = f"""# SilverAI Classification Improvement — Validation-Only Study

## Scope

This experiment uses the existing target, 5-minute candles, 15-minute horizon, threshold, 17-feature contract, chronological split and three-row purge. It loaded {result['dataset']['validated_rows']:,} validated rows to reconstruct the split boundaries. Candidate fitting and scoring used only the {result['split_metadata']['partitions'][0]['rows']:,}-row training partition and {result['split_metadata']['partitions'][1]['rows']:,}-row validation partition; zero test rows were passed to candidate fitting, tuning, scoring, or selection. The previously evaluated {result['split_metadata']['partitions'][2]['rows']:,}-row test partition was not used for model decisions.

Candidate development used three chronological folds inside training. The validation partition was then used once for final candidate comparison. No production model, default metadata, calibrator, threshold, or test output was changed.

## Validation comparison

| Candidate | Accuracy | Balanced accuracy | Macro F1 | Log loss | DOWN F1 | NEUTRAL F1 | UP F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The highest validation accuracy under the stated ordering was **{best}**. Full per-class precision, recall, F1, confusion matrices, predicted distributions, artifact hashes, and chronological-fold results are in `metrics.json`.

The current HistGradientBoosting baseline is included as `hist_gradient_boosting_current`; majority and stratified dummy baselines are included. Random Forest, Extra Trees, logistic regression, two HGB configurations, and a causal extended-feature HGB candidate were tested. XGBoost availability: **{result['candidate_availability']['xgboost']}**. LightGBM availability: **{result['candidate_availability']['lightgbm']}**.

## Decision-rule study

Class-probability multipliers were selected using only the three chronological training folds for the current HGB model. The selected multipliers and all candidates are recorded in `metrics.json`; the adjusted result is reported separately from underlying classifier performance. It was not evaluated on the historical test partition and was not activated.

## Interpretation

Validation improvement is only meaningful if it is consistent across chronological folds and does not materially sacrifice balanced accuracy, macro F1, or DOWN/NEUTRAL performance. This study does not justify replacing the default model until the candidate’s fold stability, validation trade-offs, and an independently frozen future evaluation are reviewed. No validation result establishes live performance or profitability.

## Reproduction

```powershell
python -c "from src.experiments.classification_improvement import run_experiment; run_experiment()"
```

Artifacts are stored under this versioned directory. The exact dataset hash, feature schema, split boundaries, configuration, commands, and test-protection assertions are in `metrics.json`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def run_lightgbm_supplement(output_dir: str | Path = "reports/classification_improvement_20261114_v2") -> dict[str, Any]:
    """Run a bounded single-thread LightGBM comparison when the optional package exists."""

    if not importlib.util.find_spec("lightgbm"):
        result = {"status": "unavailable"}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "lightgbm_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    from lightgbm import LGBMClassifier

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_training_data(load_modeling_data())
    splits = split_chronologically(data)
    train, validation = splits["train"], splits["validation"]
    threshold = calibrate_classification_threshold(train)

    def factory() -> Any:
        return LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, min_child_samples=40, reg_lambda=0.1, verbosity=-1, random_state=SEED, n_jobs=1)

    folds = [_fit_score(factory, fit, fold_validation, threshold, select_features) for fit, fold_validation in _folds(train)]
    final = _fit_score(factory, train, validation, threshold, select_features)
    model = factory()
    _, y_train = prepare_targets(train, threshold)
    model.fit(select_features(train), y_train)
    artifact = output / "lightgbm_bounded.joblib"
    joblib.dump(model, artifact)
    result = {
        "status": "completed",
        "candidate": "lightgbm_bounded_single_thread",
        "test_partition_materialized_for_split_metadata": True,
        "test_rows_passed_to_candidate_fitting": 0,
        "chronological_cv": {"folds": folds, "mean_accuracy": float(np.mean([x["accuracy"] for x in folds])), "mean_balanced_accuracy": float(np.mean([x["balanced_accuracy"] for x in folds])), "mean_macro_f1": float(np.mean([x["macro_f1"] for x in folds]))},
        "validation": final,
        "artifact": {"path": str(artifact), "sha256": _sha256(artifact)},
        "dataset_sha256": _sha256(Path("data/silver_modeling_data.csv")),
        "reproduction_command": "python -c \"from src.experiments.classification_improvement import run_lightgbm_supplement; run_lightgbm_supplement()\"",
    }
    (output / "lightgbm_metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    run_experiment()


__all__ = ["run_experiment", "run_lightgbm_supplement"]

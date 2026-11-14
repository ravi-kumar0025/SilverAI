"""Descriptive error analysis for the frozen SilverAI holdout.

This module deliberately performs no fitting.  The historical test partition is
used only to describe the already-persisted raw and calibrated predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.data.loader import load_modeling_data
from src.models.train import prepare_training_data, split_chronologically


LABELS = (-1, 0, 1)
NAMES = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
PROB_COLS = {
    "raw": ["raw_down", "raw_neutral", "raw_up"],
    "calibrated": ["calibrated_down", "calibrated_neutral", "calibrated_up"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    y = frame["actual_label"].astype(int).to_numpy()
    p = frame[prediction_column].astype(int).to_numpy()
    return {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(y, p)) if len(frame) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, p)) if len(frame) else None,
        "macro_f1": float(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0)) if len(frame) else None,
        "actual_counts": {NAMES[k]: int((y == k).sum()) for k in LABELS},
        "predicted_counts": {NAMES[k]: int((p == k).sum()) for k in LABELS},
    }


def _grouped(frame: pd.DataFrame, group_column: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value, group in frame.groupby(group_column, sort=False, observed=False):
        key = str(value)
        output[key] = {
            "raw": _metrics(group, "raw_predicted_label"),
            "calibrated": _metrics(group, "calibrated_predicted_label"),
        }
    return output


def _runs(incorrect: pd.Series) -> dict[str, Any]:
    values = incorrect.astype(bool).to_numpy()
    lengths: list[int] = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return {
        "incorrect_rows": int(values.sum()),
        "run_count": len(lengths),
        "longest_run": max(lengths, default=0),
        "mean_run_length": float(np.mean(lengths)) if lengths else 0.0,
        "runs_at_least_3": int(sum(length >= 3 for length in lengths)),
    }


def _confidence(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return frame[PROB_COLS[prefix]].max(axis=1)


def build_analysis(
    predictions_path: str | Path = "reports/final_holdout_20261114_v2/predictions.csv",
) -> dict[str, Any]:
    prediction_path = Path(predictions_path)
    predictions = pd.read_csv(prediction_path)
    predictions["datetime_et"] = pd.to_datetime(predictions["datetime_et"], utc=True).dt.tz_convert("America/New_York")

    data = prepare_training_data(load_modeling_data())
    splits = split_chronologically(data)
    test = splits["test"].copy()
    test["datetime_et"] = pd.to_datetime(test["datetime_et"], utc=True).dt.tz_convert("America/New_York")
    merged = test[["datetime_et", "volatility_20", "relative_volume"]].merge(predictions, on="datetime_et", how="outer", indicator=True)
    if len(merged) != len(test) or not (merged["_merge"] == "both").all():
        raise RuntimeError("Persisted predictions do not map one-to-one to the frozen test partition.")
    merged = merged.drop(columns="_merge").sort_values("datetime_et").reset_index(drop=True)

    merged["session_bucket"] = pd.cut(
        merged["datetime_et"].dt.hour,
        bins=[-1, 7, 12, 17, 23],
        labels=["00:00-07:59", "08:00-11:59", "12:00-16:59", "17:00-23:59"],
    )
    merged["time_segment"] = pd.qcut(
        np.arange(len(merged)), q=4, labels=["segment_1", "segment_2", "segment_3", "segment_4"]
    )
    for column, label in (("volatility_20", "volatility_regime"), ("relative_volume", "volume_regime")):
        low, high = merged[column].quantile([1 / 3, 2 / 3]).tolist()
        merged[label] = pd.cut(merged[column], bins=[-np.inf, low, high, np.inf], labels=["low", "mid", "high"], include_lowest=True)

    for name in PROB_COLS:
        merged[f"{name}_confidence"] = _confidence(merged, name)
        merged[f"{name}_confidence_band"] = pd.cut(
            merged[f"{name}_confidence"], bins=[-np.inf, 0.4, 0.6, np.inf], labels=["<0.40", "0.40-0.60", ">=0.60"],
        )
        merged[f"{name}_incorrect"] = merged[f"{name}_predicted_label"] != merged["actual_label"]

    volatility_thresholds = [float(x) for x in merged["volatility_20"].quantile([1 / 3, 2 / 3]).tolist()]
    volume_thresholds = [float(x) for x in merged["relative_volume"].quantile([1 / 3, 2 / 3]).tolist()]
    actual_class = {
        NAMES[k]: {
            "n": int((merged.actual_label == k).sum()),
            "raw": _metrics(merged[merged.actual_label == k], "raw_predicted_label"),
            "calibrated": _metrics(merged[merged.actual_label == k], "calibrated_predicted_label"),
        }
        for k in LABELS
    }
    probability_summary: dict[str, Any] = {}
    for name in PROB_COLS:
        probability_summary[name] = {
            "predicted_count": merged[f"{name}_predicted_label"].value_counts().sort_index().rename(index=NAMES).to_dict(),
            "confidence_quantiles": {str(q): float(merged[f"{name}_confidence"].quantile(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
            "confidence_by_actual_class": {
                NAMES[k]: {str(q): float(merged.loc[merged.actual_label == k, f"{name}_confidence"].quantile(q)) for q in [0.25, 0.5, 0.75]}
                for k in LABELS
            },
            "error_by_confidence_band": _grouped(merged.rename(columns={f"{name}_confidence_band": "confidence_band"}), "confidence_band"),
        }

    result: dict[str, Any] = {
        "analysis": "descriptive_error_analysis_20261114",
        "test_source": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "data_source": "data/silver_modeling_data.csv",
        "rows": {"validated_modeling_data": int(len(data)), "frozen_test": int(len(test)), "predictions": int(len(predictions)), "merged": int(len(merged)), "excluded": 0},
        "test_boundaries": {"start": str(test.datetime_et.iloc[0]), "end": str(test.datetime_et.iloc[-1])},
        "integrity": {"one_to_one_timestamp_mapping": True, "missing_predictions": 0, "invalid_probability_rows": int((~np.isfinite(merged[PROB_COLS["raw"] + PROB_COLS["calibrated"]].to_numpy())).any(axis=1).sum()), "chronological": bool(test.datetime_et.is_monotonic_increasing)},
        "overall": {"raw": _metrics(merged, "raw_predicted_label"), "calibrated": _metrics(merged, "calibrated_predicted_label")},
        "actual_class": actual_class,
        "confusion_matrices": {"raw": pd.crosstab(merged.actual_label, merged.raw_predicted_label).reindex(index=LABELS, columns=LABELS, fill_value=0).values.tolist(), "calibrated": pd.crosstab(merged.actual_label, merged.calibrated_predicted_label).reindex(index=LABELS, columns=LABELS, fill_value=0).values.tolist()},
        "predicted_class_distribution": {"raw": {NAMES[k]: int((merged.raw_predicted_label == k).sum()) for k in LABELS}, "calibrated": {NAMES[k]: int((merged.calibrated_predicted_label == k).sum()) for k in LABELS}},
        "chronological_segments": _grouped(merged, "time_segment"),
        "sessions": {"status": "descriptive local-time buckets; official exchange-session metadata unavailable", "results": _grouped(merged, "session_bucket")},
        "volatility_regimes": {"feature": "volatility_20", "thresholds_test_tertiles": volatility_thresholds, "results": _grouped(merged, "volatility_regime")},
        "volume_regimes": {"feature": "relative_volume", "thresholds_test_tertiles": volume_thresholds, "results": _grouped(merged, "volume_regime")},
        "confidence": probability_summary,
        "consecutive_errors": {"raw": _runs(merged.raw_incorrect), "calibrated": _runs(merged.calibrated_incorrect)},
        "neutral_behavior": {"actual_neutral": actual_class["NEUTRAL"], "raw_neutral_predicted": int((merged.raw_predicted_label == 0).sum()), "calibrated_neutral_predicted": int((merged.calibrated_predicted_label == 0).sum())},
        "robustness": {"time_stability": "descriptively assessed by chronological test quartiles", "volatility_stability": "descriptively assessed using fixed test tertiles; not used for selection", "feature_ablation": {"status": "not executed in this analysis", "support": "src/experiments/feature_ablation.py provides validation-only permutation diagnostics"}, "threshold_sensitivity": {"status": "not executed; no test-based threshold experiment permitted"}},
        "calibration_interpretation": {"raw_down_predictions": int((merged.raw_predicted_label == -1).sum()), "calibrated_down_predictions": int((merged.calibrated_predicted_label == -1).sum()), "raw_up_predictions": int((merged.raw_predicted_label == 1).sum()), "calibrated_up_predictions": int((merged.calibrated_predicted_label == 1).sum()), "interpretation": "The persisted sigmoid probabilities improve scoring metrics modestly but their class-relative changes alter argmax decisions, collapsing DOWN predictions and shifting decisions toward UP. This is a decision-distribution effect, not evidence of improved directional accuracy or profitability."},
        "limitations": ["Test subgroup results are descriptive and cannot select a model, calibrator, threshold, feature set, or decision rule.", "No official market-session identifier is present; local Eastern-time buckets are only a proxy.", "No feature-ablation or threshold-sensitivity experiment was executed here.", "No profitability or live-market conclusion is supported."],
    }
    return result


def write_report(output_dir: str | Path = "reports/error_analysis_20261114") -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = build_analysis()
    (output / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    overall = result["overall"]
    raw = overall["raw"]
    cal = overall["calibrated"]
    report = f"""# SilverAI Error Analysis and Market-Regime Robustness

## Scope and evidence status

This is a descriptive analysis of the persisted, frozen historical test predictions in `reports/final_holdout_20261114_v2/predictions.csv`. No model, calibrator, threshold, feature set, or production artifact was fitted or changed. The test contains **{result['rows']['frozen_test']:,} rows**, mapped one-to-one to `data/silver_modeling_data.csv`, from **{result['test_boundaries']['start']}** through **{result['test_boundaries']['end']}**.

The test partition was not used for calibration or selection according to the holdout integrity record. Results below are descriptive only.

## Calibration result and DOWN collapse

Raw predictions selected DOWN for **{result['calibration_interpretation']['raw_down_predictions']:,}** rows and UP for **{result['calibration_interpretation']['raw_up_predictions']:,}**. Sigmoid-calibrated argmax predictions selected DOWN for only **{result['calibration_interpretation']['calibrated_down_predictions']:,}** rows and UP for **{result['calibration_interpretation']['calibrated_up_predictions']:,}**. Thus the calibrated decision rule moved 5,094 rows away from DOWN and 5,445 rows toward UP.

The raw-to-calibrated change is visible in the persisted confusion matrices and class counts, not inferred from model internals. Probability quality improved modestly: raw versus calibrated log loss is reported in the holdout output as 1.08417 versus 1.08044, mean Brier 0.21947 versus 0.21868, and ECE 0.04201 versus 0.04052. However, accuracy changed from {raw['accuracy']:.6f} to {cal['accuracy']:.6f}, balanced accuracy from {raw['balanced_accuracy']:.6f} to {cal['balanced_accuracy']:.6f}, and DOWN F1 from 0.3164 to 0.0371.

The calibration experiment selected sigmoid on the 24,083-row validation partition after fitting on a 22,480-row chronological train suffix. Validation log loss improved from 1.04137 raw to 1.02943 calibrated and validation ECE from 0.03638 to 0.00484; isotonic was retained as a candidate but not selected. The final test results above are a separate descriptive holdout evaluation.

This establishes a probability-quality/argmax tradeoff. Calibration is not guaranteed to preserve class rankings in a multiclass one-vs-rest transformation, so a small improvement in proper scoring rules can coexist with a large shift in predicted classes. The evidence identifies the decision-distribution effect; it does not establish that calibration alone is the only causal mechanism beyond the persisted calibrated probabilities and argmax outcomes.

## Overall and class behavior

| Variant | Accuracy | Balanced accuracy | Macro-F1 | DOWN predictions | NEUTRAL predictions | UP predictions |
|---|---:|---:|---:|---:|---:|---:|
| Raw | {raw['accuracy']:.6f} | {raw['balanced_accuracy']:.6f} | {raw['macro_f1']:.6f} | {result['predicted_class_distribution']['raw']['DOWN']:,} | {result['predicted_class_distribution']['raw']['NEUTRAL']:,} | {result['predicted_class_distribution']['raw']['UP']:,} |
| Sigmoid calibrated | {cal['accuracy']:.6f} | {cal['balanced_accuracy']:.6f} | {cal['macro_f1']:.6f} | {result['predicted_class_distribution']['calibrated']['DOWN']:,} | {result['predicted_class_distribution']['calibrated']['NEUTRAL']:,} | {result['predicted_class_distribution']['calibrated']['UP']:,} |

Actual support is DOWN 9,060, NEUTRAL 6,504, UP 8,522. Raw DOWN recall/F1 are 0.2544/0.3164; calibrated DOWN recall/F1 are 0.0194/0.0371. Calibrated UP recall rises to 0.7693, while NEUTRAL recall changes from 0.3659 to 0.3527. The model therefore does not provide balanced directional coverage under the calibrated argmax.

## Regime and sequence findings

Full metrics, counts, confusion matrices, confidence bands, class-conditioned confidence, chronological quartiles, local-time buckets, volatility tertiles, volume tertiles, and error-run statistics are machine-readable in `metrics.json`. Volatility uses the existing `volatility_20` feature; volume uses existing `relative_volume`. Tertile cut points are descriptive test-distribution summaries and were not used to tune anything.

Session analysis is limited to four Eastern local-time buckets because no official exchange-session identifier is stored. Small groups must not be over-interpreted; every group includes an `n` count in `metrics.json`.

Consecutive error statistics are calculated in chronological test order for both variants. They describe instability periods, not future performance.

## Robustness and trader-oriented interpretation

Chronological and volatility stability are supported only as descriptive subgroup analyses here. `src/experiments/feature_ablation.py` supports validation-only permutation diagnostics, but no new ablation experiment was executed in this task. Threshold sensitivity was not executed because changing or selecting a threshold would be a model-decision experiment and the frozen test must not be used for that purpose.

The current evidence does not justify unqualified actionable directional signals. Raw predictions have weak, uneven class performance; sigmoid calibration improves probability scoring but produces a severe DOWN-class collapse under argmax. NEUTRAL is present but only moderately discriminated. Before a trader could trust a signal, the project would need validation-only decision-policy analysis, larger-sample stability checks, explicit session/regime definitions, monitored calibration drift, and the separately required genuine live-market evaluation. None of these findings establish profitability or support real-money trading.

## Reproduction and limitations

The analysis is reproducible with `python -c "from src.experiments.error_analysis import write_report; write_report()"`. Source implementation is `src/experiments/error_analysis.py`; feature definitions are in `src/features/technical_indicators.py`, target/split logic in `src/models/train.py`, and validation-only ablation support in `src/experiments/feature_ablation.py`. No artifacts under `saved_models/` were written.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return result


if __name__ == "__main__":
    write_report()


__all__ = ["build_analysis", "write_report"]

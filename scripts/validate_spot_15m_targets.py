"""Validation-only search for the 15-minute spot target and emission policy.

This audit deliberately does not write model artifacts.  It uses three
expanding chronological folds before the final 10% untouched block, fits the
classifier and calibrator inside each fold, and selects k/threshold from
out-of-fold validation predictions only.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_spot_15m_multiclass import (
    SPOT_FEATURE_COLUMNS,
    aligned_latest,
    complete_feature_rows,
)
from src.directional_labels import make_spot_multiclass_labels
from src.features.market_features import generate_spot_features
from src.models.probability_calibration import ChronologicalProbabilityCalibrator

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


OUTPUT = ROOT / "reports" / "spot_15m_20261116_multiclass" / "validation_audit.json"
HORIZON_BARS = 3
CANDIDATE_K = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0]
THRESHOLDS = [0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
LABEL_NAMES = {0: "down", 1: "neutral", 2: "up"}


def audit_model() -> LGBMClassifier:
    """Use production hyperparameters with parallel tree construction."""
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=240,
        learning_rate=.03,
        num_leaves=15,
        min_child_samples=100,
        colsample_bytree=.8,
        reg_lambda=2,
        reg_alpha=.05,
        verbosity=-1,
        random_state=42,
        n_jobs=4,
    )


def policy_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    raw = probabilities.argmax(axis=1)
    confident_direction = (raw != 1) & (probabilities.max(axis=1) >= threshold)
    predicted = np.where(confident_direction, raw, 1)
    directional = predicted != 1
    directional_y = y[directional]
    directional_predicted = predicted[directional]
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "directional_macro_f1": float(
            f1_score(directional_y, directional_predicted, labels=[0, 2], average="macro", zero_division=0)
        ) if len(directional_y) else 0.0,
        "directional_precision": float(np.mean(directional_predicted == directional_y)) if len(directional_y) else None,
        "directional_coverage": float(np.mean(directional)),
        "neutral_coverage": float(np.mean(predicted == 1)),
        "down_recall": float(np.mean(predicted[y == 0] == 0)) if np.any(y == 0) else None,
        "up_recall": float(np.mean(predicted[y == 2] == 2)) if np.any(y == 2) else None,
        "neutral_recall": float(np.mean(predicted[y == 1] == 1)) if np.any(y == 1) else None,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
    }


def fold_ranges(n: int) -> list[dict[str, int]]:
    """Expanding train/calibration/validation ranges; last 10% stays untouched."""
    return [
        {"train_end": int(n * 0.45), "calibration_end": int(n * 0.55), "validation_end": int(n * 0.65)},
        {"train_end": int(n * 0.55), "calibration_end": int(n * 0.65), "validation_end": int(n * 0.75)},
        {"train_end": int(n * 0.65), "calibration_end": int(n * 0.75), "validation_end": int(n * 0.90)},
    ]


def fit_fold(labeled: pd.DataFrame, ranges: dict[str, int]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    train_end = ranges["train_end"]
    calibration_end = ranges["calibration_end"]
    validation_end = ranges["validation_end"]
    train = labeled.iloc[: train_end - 3]
    calibration = labeled.iloc[train_end + 3 : calibration_end - 3]
    validation = labeled.iloc[calibration_end + 3 : validation_end - 3]
    if min(len(train), len(calibration), len(validation)) < 100:
        raise ValueError("Chronological fold is too small after purge/embargo.")
    if train.label.nunique() < 3 or calibration.label.nunique() < 3 or validation.label.nunique() < 3:
        raise ValueError("Chronological fold does not contain all three labels.")
    classifier = audit_model().fit(train[list(SPOT_FEATURE_COLUMNS)], train.label.astype(int))
    calibration_raw = classifier.predict_proba(calibration[list(SPOT_FEATURE_COLUMNS)])
    calibrator = ChronologicalProbabilityCalibrator().fit(
        calibration_raw,
        calibration.label.astype(int).to_numpy(),
        calibration_fraction=0.2,
    )
    probabilities = calibrator.transform(classifier.predict_proba(validation[list(SPOT_FEATURE_COLUMNS)]))
    return validation.label.astype(int).to_numpy(), probabilities, {
        "train": len(train),
        "calibration": len(calibration),
        "validation": len(validation),
        "validation_start": str(validation.timestamp.iloc[0]),
        "validation_end": str(validation.timestamp.iloc[-1]),
    }


def main() -> None:
    aligned, data_report = aligned_latest()
    features = generate_spot_features(
        aligned,
        ("silver_price", "volatility", "trend_momentum", "cross_market", "time_session"),
    )
    results: list[dict[str, object]] = []
    fold_report: list[dict[str, object]] = []

    for k in CANDIDATE_K:
        labeled, label_report = make_spot_multiclass_labels(features, HORIZON_BARS, k, 5)
        labeled = complete_feature_rows(labeled)
        n = len(labeled)
        out_of_fold_y: list[np.ndarray] = []
        out_of_fold_p: list[np.ndarray] = []
        current_folds = []
        for fold_number, ranges in enumerate(fold_ranges(n), start=1):
            y, probabilities, sizes = fit_fold(labeled, ranges)
            out_of_fold_y.append(y)
            out_of_fold_p.append(probabilities)
            current_folds.append({"fold": fold_number, **sizes})
        y_all = np.concatenate(out_of_fold_y)
        p_all = np.vstack(out_of_fold_p)
        fold_report.append({"k": k, "label_report": label_report, "folds": current_folds})
        for threshold in THRESHOLDS:
            metrics = policy_metrics(y_all, p_all, threshold)
            results.append({
                "k": k,
                "threshold": threshold,
                "label_counts": label_report["class_counts"],
                **metrics,
            })

    eligible = [
        row for row in results
        if row["directional_coverage"] >= 0.05
        and (row["down_recall"] or 0) > 0
        and (row["up_recall"] or 0) > 0
    ]
    if not eligible:
        raise RuntimeError("No candidate produced both directional classes at 5% validation coverage.")
    selected = max(
        eligible,
        key=lambda row: (
            row["directional_macro_f1"],
            row["balanced_accuracy"],
            row["directional_precision"] or 0,
            row["directional_coverage"],
        ),
    )
    ranked = sorted(
        eligible,
        key=lambda row: (
            row["directional_macro_f1"],
            row["balanced_accuracy"],
            row["directional_precision"] or 0,
        ),
        reverse=True,
    )
    report = {
        "status": "validation_only",
        "selection_rule": "maximize out_of_fold directional macro F1, then balanced accuracy, with both UP and DOWN recall nonzero and at least 5% directional coverage",
        "data": data_report,
        "untouched_test_fraction": 0.10,
        "candidate_k": CANDIDATE_K,
        "candidate_thresholds": THRESHOLDS,
        "folds": fold_report,
        "selected": selected,
        "top_candidates": ranked[:20],
        "all_results": results,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "selected": selected, "top_candidates": ranked[:10]}, indent=2, default=str))


if __name__ == "__main__":
    main()

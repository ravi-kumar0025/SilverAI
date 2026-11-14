"""Validation-only feature importance and ablation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.train import LABELS, select_features


def permutation_feature_importance(
    model: Any,
    validation_frame: pd.DataFrame,
    validation_labels: pd.Series,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure one-feature validation macro-F1 degradation.

    The fitted model is not refit and the test partition is never touched.
    This is a diagnostic ablation, not a causal importance claim.
    """

    features = select_features(validation_frame)
    labels = np.asarray(validation_labels, dtype=int)
    baseline = f1_score(labels, model.predict(features), labels=list(LABELS), average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    rows = []
    for feature in features.columns:
        ablated = features.copy()
        values = ablated[feature].to_numpy(copy=True)
        rng.shuffle(values)
        ablated[feature] = values
        score = f1_score(labels, model.predict(ablated), labels=list(LABELS), average="macro", zero_division=0)
        rows.append({"feature": feature, "baseline_macro_f1": float(baseline), "ablated_macro_f1": float(score), "importance_macro_f1_drop": float(baseline - score)})
    return pd.DataFrame(rows).sort_values("importance_macro_f1_drop", ascending=False).reset_index(drop=True)


__all__ = ["permutation_feature_importance"]

"""Independent directional baselines used for model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def rsi_direction(close: pd.Series, period: int = 14, overbought: float = 70, oversold: float = 30) -> pd.Series:
    """Return an RSI-only direction rule: oversold UP, overbought DOWN, else NEUTRAL."""
    delta = pd.to_numeric(close, errors="coerce").diff()
    gains = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    losses = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    result = pd.Series(0, index=close.index, dtype="int64")
    result.loc[rsi < oversold] = 1
    result.loc[rsi > overbought] = -1
    return result


def classification_baselines(y_true: pd.Series, close: pd.Series | None = None) -> pd.DataFrame:
    """Compare constant UP/DOWN/majority and an independent RSI rule."""
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    candidates: dict[str, np.ndarray] = {
        "always_up_baseline": np.ones(len(y), dtype=int),
        "always_down_baseline": np.full(len(y), -1, dtype=int),
        "majority_class_baseline": np.full(len(y), int(y.mode().iloc[0]), dtype=int),
    }
    if close is not None:
        candidates["rsi_14_baseline"] = rsi_direction(close).fillna(0).astype(int).to_numpy()[: len(y)]
    rows = []
    for name, pred in candidates.items():
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        })
    return pd.DataFrame(rows)


__all__ = ["classification_baselines", "rsi_direction"]

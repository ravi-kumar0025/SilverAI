"""Trained XAG/USD return and quantile regression artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.market_features import SPOT_FEATURE_COLUMNS
from src.models.model_artifact_contract import load_versioned_artifact


def load_regression_artifacts(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    names = {"mean": "spot_return_regressor.joblib", "q10": "spot_return_q10.joblib", "q50": "spot_return_q50.joblib", "q90": "spot_return_q90.joblib"}
    loaded: dict[str, Any] = {}
    for key, filename in names.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Regression artifact not found: {path}")
        loaded[key] = load_versioned_artifact(path, artifact_kind=f"spot_regressor_{key}")
    features = tuple(loaded["mean"].get("features", ()))
    if features != SPOT_FEATURE_COLUMNS:
        raise ValueError("Regression artifact feature order does not match the 34-feature schema.")
    return loaded


def predict_regression(frame: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    features = frame.loc[:, list(SPOT_FEATURE_COLUMNS)]
    values = {key: np.asarray(artifacts[key]["model"].predict(features), dtype=float) for key in ("mean", "q10", "q50", "q90")}
    result = pd.DataFrame(index=frame.index)
    result["predicted_return"] = values["mean"]
    result["return_q10"] = values["q10"]
    result["return_q50"] = values["q50"]
    result["return_q90"] = values["q90"]
    close = pd.to_numeric(frame["silver_close"] if "silver_close" in frame else frame["close"], errors="raise").to_numpy()
    result["predicted_price"] = close * np.exp(result["return_q50"])
    result["price_low"] = close * np.exp(result["return_q10"])
    result["price_high"] = close * np.exp(result["return_q90"])
    result["regression_direction"] = np.where(result["return_q50"] > 0, "Up", np.where(result["return_q50"] < 0, "Down", "No clear signal"))
    return result.reset_index(drop=True)


__all__ = ["load_regression_artifacts", "predict_regression"]

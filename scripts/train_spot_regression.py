"""Train/evaluate the development-only XAG/USD return regressors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.market_candle_builder import align_market_data
from src.features.market_features import SPOT_FEATURE_COLUMNS, generate_spot_features
from src.models.model_artifact_contract import dump_versioned_artifact, runtime_versions
from src.market_model_config import SPOT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "spot_v3_20261116_regression"


def model(alpha: float | None = None) -> LGBMRegressor:
    return LGBMRegressor(objective="quantile" if alpha is not None else "huber", alpha=alpha, n_estimators=140, learning_rate=0.035, num_leaves=15, min_child_samples=100, reg_lambda=2.0, verbosity=-1, random_state=42, n_jobs=1)


def wilson(correct: int, total: int) -> tuple[float, float]:
    if not total: return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = correct / total; den = 1 + z*z/total
    centre = (p + z*z/(2*total))/den; half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return centre-half, centre+half


def metrics(y: np.ndarray, pred: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> dict[str, float]:
    naive = np.zeros_like(y)
    sign = (np.sign(pred) == np.sign(y)) & (y != 0)
    lo, hi = wilson(int(sign.sum()), int(sign.size))
    return {"rows": int(len(y)), "r2_vs_zero": float(r2_score(y, pred)), "mae_vs_naive": float(mean_absolute_error(y, pred)), "rmse_vs_naive": float(np.sqrt(mean_squared_error(y, pred))), "spearman_ic": float(spearmanr(y, pred).statistic), "sign_accuracy": float(sign.mean()), "sign_ci_low": float(lo), "sign_ci_high": float(hi), "band_coverage": float(((y >= q10) & (y <= q90)).mean()), "band_width": float(np.mean(q90-q10))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aligned, _ = align_market_data(5)
    frame = generate_spot_features(aligned, SPOT_CONFIG.feature_groups)
    frame["target_return"] = np.log(frame.silver_close.shift(-SPOT_CONFIG.forecast_horizon_bars) / frame.silver_close)
    end = pd.Timestamp(SPOT_CONFIG.final_holdout_start, tz="UTC")
    frame = frame[(pd.to_datetime(frame.timestamp, utc=True) < end) & frame[list(SPOT_FEATURE_COLUMNS)].notna().all(axis=1) & frame.target_return.notna()].reset_index(drop=True)
    rows = []
    folds = [(0.50, 0.15), (0.65, 0.15), (0.80, 0.10)]
    for fold, (train_frac, valid_frac) in enumerate(folds, 1):
        train_end = int(len(frame)*train_frac) - SPOT_CONFIG.purge_bars - SPOT_CONFIG.embargo_bars
        valid_start = int(len(frame)*train_frac) + SPOT_CONFIG.purge_bars
        valid_end = min(len(frame), valid_start + int(len(frame)*valid_frac))
        train, valid = frame.iloc[:train_end], frame.iloc[valid_start:valid_end]
        fitted = {"mean": model().fit(train[list(SPOT_FEATURE_COLUMNS)], train.target_return), "q10": model(0.1).fit(train[list(SPOT_FEATURE_COLUMNS)], train.target_return), "q50": model(0.5).fit(train[list(SPOT_FEATURE_COLUMNS)], train.target_return), "q90": model(0.9).fit(train[list(SPOT_FEATURE_COLUMNS)], train.target_return)}
        pred = {key: fitted[key].predict(valid[list(SPOT_FEATURE_COLUMNS)]) for key in fitted}
        row = {"fold": fold, **metrics(valid.target_return.to_numpy(), pred["mean"], pred["q10"], pred["q90"])}
        rows.append(row)
    final_train = frame.iloc[: int(len(frame)*0.85)]
    fitted = {"mean": model().fit(final_train[list(SPOT_FEATURE_COLUMNS)], final_train.target_return), "q10": model(0.1).fit(final_train[list(SPOT_FEATURE_COLUMNS)], final_train.target_return), "q50": model(0.5).fit(final_train[list(SPOT_FEATURE_COLUMNS)], final_train.target_return), "q90": model(0.9).fit(final_train[list(SPOT_FEATURE_COLUMNS)], final_train.target_return)}
    metadata = {"runtime_versions": runtime_versions(), "dataset_id": SPOT_CONFIG.dataset_id, "config_sha256": hashlib.sha256((ROOT/"config.yaml").read_bytes()).hexdigest(), "feature_schema": list(SPOT_FEATURE_COLUMNS), "horizon_bars": SPOT_CONFIG.forecast_horizon_bars, "target": "log(close[t+12]/close[t])", "development_rows": len(frame), "model_version": "xagusd_5m_60m_return_lgbm_v1_research"}
    for key, fitted_model in fitted.items(): dump_versioned_artifact({"model": fitted_model, "features": list(SPOT_FEATURE_COLUMNS), **metadata}, OUT/f"spot_return_{key if key != 'mean' else 'regressor'}.joblib", artifact_kind=f"spot_regressor_{key}")
    table = pd.DataFrame(rows); table.loc[len(table)] = {"fold": "mean", **table.drop(columns="fold").mean(numeric_only=True).to_dict()}; table.to_csv(OUT/"regression_metrics.csv", index=False)
    (OUT/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__": main()

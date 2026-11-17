"""Run an isolated OOS threshold study for the existing 60-minute regressors.

This script deliberately stops at threshold selection. It never reads the
configured final holdout into a fitting or selection operation and writes to a
new versioned report directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.spot_data import align_market_data
from src.features.spot_features import SPOT_FEATURE_COLUMNS, generate_spot_features
from src.spot_config import SPOT_CONFIG


OUT = ROOT / "reports" / "spot_v3_20261117_threshold_study"
THRESHOLDS_BPS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)


def model(alpha: float | None = None) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile" if alpha is not None else "huber",
        alpha=alpha,
        n_estimators=140,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=100,
        reg_lambda=2.0,
        verbosity=-1,
        random_state=42,
        n_jobs=1,
    )


def _fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, np.ndarray]:
    x_train = train.loc[:, SPOT_FEATURE_COLUMNS]
    y_train = train["target_return"].to_numpy()
    x_valid = valid.loc[:, SPOT_FEATURE_COLUMNS]
    fitted = {
        "mean": model().fit(x_train, y_train),
        "q10": model(0.1).fit(x_train, y_train),
        "q50": model(0.5).fit(x_train, y_train),
        "q90": model(0.9).fit(x_train, y_train),
    }
    return {key: np.asarray(value.predict(x_valid), dtype=float) for key, value in fitted.items()}


def _non_overlapping(signal_rows: pd.DataFrame) -> pd.DataFrame:
    if signal_rows.empty:
        return signal_rows.copy()
    chosen: list[int] = []
    last_exit: pd.Timestamp | None = None
    for index, row in signal_rows.sort_values("timestamp").iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        if last_exit is not None and timestamp < last_exit:
            continue
        chosen.append(index)
        last_exit = timestamp + pd.Timedelta(minutes=SPOT_CONFIG.forecast_horizon_minutes)
    return signal_rows.loc[chosen]


def _evaluate(predictions: pd.DataFrame, source: str, threshold_bps: int) -> dict[str, object]:
    threshold = threshold_bps / 10_000.0
    predicted = predictions[source]
    side = np.where(predicted >= threshold, "BUY", np.where(predicted <= -threshold, "SELL", "NO_CLEAR_SIGNAL"))
    selected = predictions.assign(signal=side)
    selected = selected[selected.signal != "NO_CLEAR_SIGNAL"].copy()
    selected["directional_return"] = np.where(selected.signal == "BUY", selected.actual_return, -selected.actual_return)
    executed = _non_overlapping(selected)
    cost_rate = (SPOT_CONFIG.spread_bps + SPOT_CONFIG.slippage_bps) / 10_000.0
    executed["gross_return"] = executed.directional_return
    executed["net_return"] = executed.directional_return - 2 * cost_rate
    buy = selected[selected.signal == "BUY"]
    sell = selected[selected.signal == "SELL"]
    directional = selected.directional_return
    actual = executed.actual_return.to_numpy(dtype=float)
    return {
        "source": source,
        "threshold_bps": threshold_bps,
        "threshold_decimal_log_return": threshold,
        "buy_signal_count": int(len(buy)),
        "sell_signal_count": int(len(sell)),
        "total_signal_count": int(len(selected)),
        "executed_trade_count": int(len(executed)),
        "signal_coverage": float(len(selected) / len(predictions)) if len(predictions) else float("nan"),
        "directional_hit_rate": float((directional > 0).mean()) if len(directional) else float("nan"),
        "buy_directional_accuracy": float((buy.actual_return > 0).mean()) if len(buy) else float("nan"),
        "sell_directional_accuracy": float((sell.actual_return < 0).mean()) if len(sell) else float("nan"),
        "average_predicted_return": float(selected[source].mean()) if len(selected) else float("nan"),
        "average_actual_forward_return": float(executed.actual_return.mean()) if len(executed) else float("nan"),
        "average_gross_return": float(executed.gross_return.mean()) if len(executed) else float("nan"),
        "average_net_return": float(executed.net_return.mean()) if len(executed) else float("nan"),
        "total_net_return": float(executed.net_return.sum()) if len(executed) else 0.0,
        "mae": float(mean_absolute_error(predictions.actual_return, predictions[source])),
        "rmse": float(np.sqrt(mean_squared_error(predictions.actual_return, predictions[source]))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aligned, _ = align_market_data(5)
    frame = generate_spot_features(aligned, SPOT_CONFIG.feature_groups)
    frame["target_return"] = np.log(frame.silver_close.shift(-SPOT_CONFIG.forecast_horizon_bars) / frame.silver_close)
    holdout_start = pd.Timestamp(SPOT_CONFIG.final_holdout_start, tz="UTC")
    frame = frame[
        (pd.to_datetime(frame.timestamp, utc=True) < holdout_start)
        & frame[list(SPOT_FEATURE_COLUMNS)].notna().all(axis=1)
        & frame.target_return.notna()
    ].reset_index(drop=True)

    # These are the existing development-only chronological windows. The
    # purge/embargo gap is applied on both sides of each validation window.
    folds = [(1, 0.50, 0.15), (2, 0.65, 0.15), (3, 0.80, 0.10)]
    prediction_rows: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    for fold, train_frac, valid_frac in folds:
        boundary = int(len(frame) * train_frac)
        train_end = boundary - SPOT_CONFIG.purge_bars - SPOT_CONFIG.embargo_bars
        valid_start = boundary + SPOT_CONFIG.purge_bars
        valid_end = min(len(frame), valid_start + int(len(frame) * valid_frac))
        train, valid = frame.iloc[:train_end], frame.iloc[valid_start:valid_end]
        predictions = _fit_predict(train, valid)
        rows = pd.DataFrame({
            "fold": fold,
            "timestamp": pd.to_datetime(valid.timestamp, utc=True).to_numpy(),
            "actual_return": valid.target_return.to_numpy(dtype=float),
            **predictions,
        })
        prediction_rows.append(rows)
        fold_records.append({
            "fold": fold,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "train_end_timestamp": pd.Timestamp(train.timestamp.iloc[-1]).isoformat(),
            "validation_start_timestamp": pd.Timestamp(valid.timestamp.iloc[0]).isoformat(),
            "validation_end_timestamp": pd.Timestamp(valid.timestamp.iloc[-1]).isoformat(),
            "purge_bars": SPOT_CONFIG.purge_bars,
            "embargo_bars": SPOT_CONFIG.embargo_bars,
        })

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(OUT / "validation_predictions.csv", index=False)
    fold_table = pd.DataFrame(fold_records)
    fold_table.to_csv(OUT / "fold_boundaries.csv", index=False)

    all_metrics: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    for source in ("mean", "q50"):
        for threshold in THRESHOLDS_BPS:
            all_metrics.append(_evaluate(predictions, source, threshold))
            for fold in sorted(predictions.fold.unique()):
                fold_pred = predictions[predictions.fold == fold].copy()
                fold_metrics.append({"fold": int(fold), **_evaluate(fold_pred, source, threshold)})
    metrics_table = pd.DataFrame(all_metrics)
    fold_metrics_table = pd.DataFrame(fold_metrics)
    metrics_table.to_csv(OUT / "threshold_comparison.csv", index=False)
    fold_metrics_table.to_csv(OUT / "threshold_comparison_per_fold.csv", index=False)

    # Selection is validation-only and intentionally conservative: positive
    # net return, at least 30 non-overlapping trades, and no negative fold net
    # return. If none qualifies, freeze no threshold.
    grouped = fold_metrics_table.groupby(["source", "threshold_bps"], as_index=False)
    fold_summary = grouped.agg(
        fold_count=("fold", "count"),
        negative_fold_count=("total_net_return", lambda s: int((s < 0).sum())),
        min_fold_net_return=("total_net_return", "min"),
        mean_fold_net_return=("total_net_return", "mean"),
    )
    candidates = metrics_table.merge(fold_summary, on=["source", "threshold_bps"])
    candidates = candidates[
        (candidates.executed_trade_count >= 30)
        & (candidates.total_net_return > 0)
        & (candidates.negative_fold_count == 0)
    ].sort_values(["total_net_return", "executed_trade_count"], ascending=[False, False])
    selected = None if candidates.empty else candidates.iloc[0].to_dict()
    selection = {
        "selected": selected,
        "status": "selected" if selected else "no_defensible_threshold",
        "selection_policy": "validation-only; positive net return after configured costs; >=30 non-overlapping trades; no negative fold net return; rank by total net return",
        "threshold_units": "decimal log-return; 1 bp = 0.0001",
        "cost_assumptions": {"spread_bps": SPOT_CONFIG.spread_bps, "slippage_bps": SPOT_CONFIG.slippage_bps, "round_trip_cost_bps": 2 * (SPOT_CONFIG.spread_bps + SPOT_CONFIG.slippage_bps)},
        "holdout": {"status": SPOT_CONFIG.holdout_status, "excluded_before": SPOT_CONFIG.final_holdout_start, "used_for_selection": False},
    }
    (OUT / "selected_threshold.json").write_text(json.dumps(selection, indent=2, default=str), encoding="utf-8")
    metadata = {
        "experiment_version": "spot_v3_20261117_threshold_study_v1",
        "dataset_id": SPOT_CONFIG.dataset_id,
        "feature_schema": list(SPOT_FEATURE_COLUMNS),
        "target": "log(close[t+12]/close[t])",
        "horizon_bars": SPOT_CONFIG.forecast_horizon_bars,
        "horizon_minutes": SPOT_CONFIG.forecast_horizon_minutes,
        "thresholds_bps": list(THRESHOLDS_BPS),
        "threshold_decimal_conversion": "bps / 10000",
        "models": ["mean", "q10", "q50", "q90"],
        "selection_sources": ["mean", "q50"],
        "development_rows": len(frame),
        "prediction_rows": len(predictions),
        "folds": fold_records,
        "config_sha256": hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest(),
        "selection": selection,
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report = [
        "# Regression threshold study",
        "",
        f"Status: **{selection['status']}**.",
        "",
        "This isolated study uses chronological OOS predictions from the existing LightGBM mean/Q10/Q50/Q90 architecture, the exact 34-feature schema, and target `log(close[t+12]/close[t])`. The configured final holdout was excluded before fitting and selection.",
        "",
        f"Thresholds tested: {', '.join(f'{x} bp' for x in THRESHOLDS_BPS)}; decimal log-return conversion is `bp / 10000`. Costs: {SPOT_CONFIG.spread_bps} bp spread + {SPOT_CONFIG.slippage_bps} bp slippage per side.",
        "",
        "See `threshold_comparison.csv`, `threshold_comparison_per_fold.csv`, and `selected_threshold.json` for the complete evidence.",
    ]
    (OUT / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "status": selection["status"], "selected": selected}, indent=2, default=str))


if __name__ == "__main__":
    main()

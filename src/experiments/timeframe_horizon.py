"""Coverage and target-validity experiments for timeframe/horizon selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.preprocessing import validate_ohlcv
from src.targets import generate_timestamp_targets


def resample_ohlcv(frame: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive.")
    required = {"datetime_et", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError("OHLCV frame is missing required columns.")
    ordered = frame.copy().sort_values("datetime_et").set_index("datetime_et")
    result = ordered.resample(f"{timeframe_minutes}min", origin="start_day", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    result = result.reset_index()
    result.attrs["timeframe_minutes"] = timeframe_minutes
    return result


def evaluate_candidate(frame: pd.DataFrame, timeframe_minutes: int, horizon_minutes: int, neutral_quantile: float = 0.40) -> dict[str, Any]:
    candles = resample_ohlcv(frame, timeframe_minutes)
    target = generate_timestamp_targets(candles, interval_minutes=timeframe_minutes, horizon_minutes=horizon_minutes)
    valid = target[target["target_valid"]].copy()
    if valid.empty:
        raise ValueError("Candidate has no valid timestamp-aware targets.")
    threshold = float(valid["future_return_pct"].abs().quantile(neutral_quantile))
    target = generate_timestamp_targets(candles, interval_minutes=timeframe_minutes, horizon_minutes=horizon_minutes, threshold_pct=threshold)
    valid = target[target["target_valid"]].copy()
    counts = valid["target_direction"].value_counts(normalize=True).to_dict()
    return {
        "timeframe_minutes": timeframe_minutes,
        "horizon_minutes": horizon_minutes,
        "rows": len(candles),
        "valid_target_rows": len(valid),
        "coverage_ratio": len(valid) / len(candles),
        "first_timestamp": candles["datetime_et"].iloc[0].isoformat(),
        "last_timestamp": candles["datetime_et"].iloc[-1].isoformat(),
        "neutral_threshold_pct_train_proxy": threshold,
        "class_balance": {str(int(key)): float(value) for key, value in counts.items()},
        "session_gap_policy": "require every expected interval and exact future timestamp; unresolved rows excluded from target evaluation",
    }


def run_timeframe_horizon_experiment(frame: pd.DataFrame, candidates: Iterable[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for timeframe_minutes, horizon_minutes in candidates:
        try:
            rows.append(evaluate_candidate(frame, timeframe_minutes, horizon_minutes))
        except ValueError as exc:
            rows.append({"timeframe_minutes": timeframe_minutes, "horizon_minutes": horizon_minutes, "status": "rejected", "reason": str(exc)})
    return pd.DataFrame(rows)


def save_experiment_report(results: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination.with_suffix(".csv"), index=False)
    destination.with_suffix(".json").write_text(results.to_json(orient="records", indent=2), encoding="utf-8")
    return destination


__all__ = ["evaluate_candidate", "resample_ohlcv", "run_timeframe_horizon_experiment", "save_experiment_report"]

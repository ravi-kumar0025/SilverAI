"""Causal target construction for the isolated XAG/USD spot pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_model_config import SPOT_CONFIG


def _price_columns(frame: pd.DataFrame, required_roles: tuple[str, ...]) -> dict[str, str]:
    mapping = dict(SPOT_CONFIG.price_columns)
    missing = [role for role in required_roles if mapping[role] not in frame.columns]
    if missing:
        raise ValueError("Spot target frame is missing required configured price columns: " + ", ".join(f"{role}={mapping[role]}" for role in missing))
    return mapping


def _atr(frame: pd.DataFrame) -> pd.Series:
    columns = _price_columns(frame, ("close", "high", "low"))
    close = frame[columns["close"]].astype(float)
    previous = close.shift(1)
    true_range = pd.concat([(frame[columns["high"]] - frame[columns["low"]]).abs(),
                            (frame[columns["high"]] - previous).abs(),
                            (frame[columns["low"]] - previous).abs()], axis=1).max(axis=1)
    return true_range.rolling(14, min_periods=14).mean() / close


def make_spot_multiclass_labels(frame: pd.DataFrame, horizon_bars: int, k: float,
                                candle_minutes: int | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create 0=Down, 1=Neutral, 2=Up labels using a causal ATR dead-zone."""
    if horizon_bars <= 0 or k <= 0:
        raise ValueError("horizon_bars and k must be positive")
    columns = _price_columns(frame, ("timestamp", "close", "high", "low"))
    result = frame.copy().sort_values(columns["timestamp"], kind="mergesort").reset_index(drop=True)
    close = result[columns["close"]].astype(float)
    atr_pct = _atr(result)
    future = np.log(close.shift(-horizon_bars) / close)
    threshold = k * atr_pct
    valid = future.notna() & threshold.notna() & np.isfinite(future) & np.isfinite(threshold)
    label = pd.Series(np.nan, index=result.index, dtype=float)
    label.loc[valid & (future < -threshold)] = 0
    label.loc[valid & (future.abs() <= threshold)] = 1
    label.loc[valid & (future > threshold)] = 2
    result["fwd_ret"] = future
    result["atr_pct"] = atr_pct
    result["target_threshold"] = threshold
    result["label"] = label
    kept = result.dropna(subset=["label"]).reset_index(drop=True)
    interval = int(candle_minutes or SPOT_CONFIG.candle_minutes)
    counts = kept["label"].astype(int).value_counts().reindex([0, 1, 2], fill_value=0)
    report = {"horizon_bars": int(horizon_bars), "horizon_minutes": horizon_bars * interval,
              "k": float(k), "rows_before": len(result), "rows_kept": len(kept),
              "rows_dropped_no_move_or_warmup": int(len(result) - len(kept)),
              "class_counts": {"down": int(counts[0]), "neutral": int(counts[1]), "up": int(counts[2])}}
    return kept, report


def make_spot_labels(frame: pd.DataFrame, horizon_bars: int, k: float,
                     candle_minutes: int | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Backward-compatible binary target builder; new work should use multiclass labels."""
    columns = _price_columns(frame, ("timestamp", "close", "high", "low"))
    result = frame.copy().sort_values(columns["timestamp"], kind="mergesort").reset_index(drop=True)
    close = result[columns["close"]].astype(float); atr_pct = _atr(result)
    future = np.log(close.shift(-horizon_bars) / close); threshold = k * atr_pct
    valid = future.notna() & threshold.notna() & np.isfinite(future) & np.isfinite(threshold)
    label = pd.Series(np.nan, index=result.index, dtype=float)
    label.loc[valid & (future > threshold)] = 1; label.loc[valid & (future < -threshold)] = 0
    result["fwd_ret"], result["atr_pct"], result["target_threshold"], result["label"] = future, atr_pct, threshold, label
    kept = result.dropna(subset=["label"]).reset_index(drop=True)
    interval = int(candle_minutes or SPOT_CONFIG.candle_minutes)
    return kept, {"horizon_bars": horizon_bars, "horizon_minutes": horizon_bars * interval, "k": float(k), "rows_before": len(result), "rows_kept": len(kept), "rows_dropped_no_move_or_warmup": int(len(result) - len(kept)), "up_fraction": float(kept.label.mean()) if len(kept) else None}


def make_triple_barrier_labels(frame: pd.DataFrame, horizon_bars: int, profit_multiple: float,
                               stop_multiple: float | None = None, timeout_policy: str = "sign",
                               candle_minutes: int | None = None, return_full: bool = False):
    """Create causal triple-barrier labels retained for legacy tests and reports."""
    if horizon_bars <= 0 or profit_multiple <= 0:
        raise ValueError("horizon_bars and profit_multiple must be positive")
    if timeout_policy not in {"sign", "neutral", "drop"}:
        raise ValueError("timeout_policy must be sign, neutral, or drop")
    stop_multiple = profit_multiple if stop_multiple is None else float(stop_multiple)
    columns = _price_columns(frame, ("timestamp", "close", "high", "low"))
    result = frame.copy().sort_values(columns["timestamp"], kind="mergesort").reset_index(drop=True)
    ts = pd.to_datetime(result[columns["timestamp"]], utc=True, errors="raise")
    close = result[columns["close"]].astype(float).to_numpy(); high = result[columns["high"]].astype(float).to_numpy(); low = result[columns["low"]].astype(float).to_numpy()
    atr = _atr(result).to_numpy(); labels = np.full(len(result), np.nan); exits = np.full(len(result), np.nan)
    index = {int(value): i for i, value in enumerate(ts.array.asi8)}; step = pd.Timedelta(minutes=int(candle_minutes or SPOT_CONFIG.candle_minutes)).value
    missing_future_rows = both_barrier_rows = timeout_rows = timeout_up_rows = timeout_down_rows = timeout_zero_rows = 0
    for i in range(len(result)):
        if not np.isfinite(atr[i]): continue
        future = [index.get(int(ts.array.asi8[i] + j * step)) for j in range(1, horizon_bars + 1)]
        if any(j is None for j in future): missing_future_rows += 1; continue
        upper, lower = close[i] * (1 + profit_multiple * atr[i]), close[i] * (1 - stop_multiple * atr[i])
        for offset, j in enumerate(future, 1):
            if high[j] >= upper and low[j] <= lower: both_barrier_rows += 1; exits[i] = offset; break
            if high[j] >= upper: labels[i], exits[i] = 1, offset; break
            if low[j] <= lower: labels[i], exits[i] = 0, offset; break
        if np.isnan(labels[i]) and np.isnan(exits[i]):
            timeout_rows += 1; exits[i] = horizon_bars; ret = close[future[-1]] - close[i]
            labels[i] = 1 if ret > 0 else 0 if ret < 0 else np.nan
            timeout_up_rows += int(ret > 0); timeout_down_rows += int(ret < 0); timeout_zero_rows += int(ret == 0)
            if timeout_policy == "neutral": labels[i] = 0.5
            if timeout_policy == "drop": labels[i] = np.nan
    result["triple_barrier_label"], result["triple_barrier_exit_bars"], result["triple_barrier_atr_pct"] = labels, exits, atr
    kept = result.dropna(subset=["triple_barrier_label"]).reset_index(drop=True)
    report = {"horizon_bars": horizon_bars, "profit_multiple": float(profit_multiple), "stop_multiple": stop_multiple,
              "rows_before": len(result), "rows_kept": len(kept), "rows_dropped": len(result) - len(kept),
              "up_fraction": float(kept.triple_barrier_label.mean()) if len(kept) else None,
              "missing_future_rows": missing_future_rows, "both_barrier_rows": both_barrier_rows,
              "timeout_rows": timeout_rows, "timeout_up_rows": timeout_up_rows,
              "timeout_down_rows": timeout_down_rows, "timeout_zero_rows": timeout_zero_rows}
    return (result if return_full else kept), report


def sample_uniqueness_weights(frame: pd.DataFrame, horizon_bars: int, label_column: str = "label") -> pd.Series:
    valid = frame[label_column].notna().to_numpy(); weights = np.zeros(len(frame)); positions = np.flatnonzero(valid)
    for start in positions: weights[start] = 1.0 / (1 + valid[start + 1:min(len(frame), start + horizon_bars)].sum())
    return pd.Series(weights, index=frame.index, name="sample_uniqueness_weight")


__all__ = ["make_spot_labels", "make_spot_multiclass_labels", "make_triple_barrier_labels", "sample_uniqueness_weights"]

"""ATR-scaled binary labels for the isolated XAG/USD spot pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.spot_config import SPOT_CONFIG


def _price_columns(frame: pd.DataFrame, required_roles: tuple[str, ...]) -> dict[str, str]:
    mapping = dict(SPOT_CONFIG.price_columns)
    missing = [role for role in required_roles if mapping[role] not in frame.columns]
    if missing:
        expected = ", ".join(f"{role}={mapping[role]}" for role in missing)
        raise ValueError(f"Spot target frame is missing required configured price columns: {expected}")
    return mapping


def make_spot_labels(frame: pd.DataFrame, horizon_bars: int, k: float, candle_minutes: int | None = None) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if horizon_bars <= 0 or k <= 0: raise ValueError("horizon_bars and k must be positive")
    columns = _price_columns(frame, ("timestamp", "close", "high", "low"))
    result = frame.copy().sort_values(columns["timestamp"], kind="mergesort").reset_index(drop=True)
    close = result[columns["close"]].astype(float); high, low = result[columns["high"]].astype(float), result[columns["low"]].astype(float)
    previous = close.shift(1); true_range = pd.concat([(high-low).abs(),(high-previous).abs(),(low-previous).abs()],axis=1).max(axis=1); atr_pct = true_range.rolling(14,min_periods=14).mean()/close
    future = np.log(close.shift(-horizon_bars)/close); threshold = k*atr_pct
    valid = future.notna() & threshold.notna() & np.isfinite(future) & np.isfinite(threshold)
    label = pd.Series(np.nan,index=result.index,dtype=float); label.loc[valid & (future > threshold)] = 1; label.loc[valid & (future < -threshold)] = 0
    result["fwd_ret"] = future; result["atr_pct"] = atr_pct; result["target_threshold"] = threshold; result["label"] = label
    kept = result[result.label.notna()].copy().reset_index(drop=True)
    interval = int(candle_minutes or SPOT_CONFIG.candle_minutes)
    report = {"horizon_bars": horizon_bars, "horizon_minutes": horizon_bars*interval, "k": k, "rows_before": len(result), "rows_dropped_no_move_or_warmup": int(len(result)-len(kept)), "rows_kept": len(kept), "up_fraction": float(kept.label.mean()) if len(kept) else None}
    return kept, report


__all__ = ["make_spot_labels"]


def make_triple_barrier_labels(
    frame: pd.DataFrame,
    horizon_bars: int,
    profit_multiple: float,
    stop_multiple: float | None = None,
    timeout_policy: str = "sign",
    candle_minutes: int | None = None,
    return_full: bool = False,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Create causal ATR-scaled triple-barrier labels.

    The upper/lower barriers are fixed from the close and ATR known at the
    event candle. If neither barrier is touched, the label is the sign of the
    horizon close relative to the event close. Rows without a complete future
    window or ATR are dropped. No holdout-specific selection occurs here.
    """
    if horizon_bars <= 0 or profit_multiple <= 0:
        raise ValueError("horizon_bars and profit_multiple must be positive")
    stop_multiple = profit_multiple if stop_multiple is None else float(stop_multiple)
    if stop_multiple <= 0:
        raise ValueError("stop_multiple must be positive")
    if timeout_policy not in {"sign", "neutral", "drop"}:
        raise ValueError("timeout_policy must be sign, neutral, or drop")
    columns = _price_columns(frame, ("timestamp", "close", "high", "low"))
    result = frame.copy().sort_values(columns["timestamp"], kind="mergesort").reset_index(drop=True)
    timestamp = pd.to_datetime(result[columns["timestamp"]], utc=True, errors="raise")
    timestamp_ns = timestamp.array.asi8
    close = result[columns["close"]].astype(float)
    close_values = close.to_numpy()
    high_values = result[columns["high"]].astype(float).to_numpy()
    low_values = result[columns["low"]].astype(float).to_numpy()
    previous = close.shift(1)
    true_range = pd.concat(
        [(result[columns["high"]] - result[columns["low"]]).abs(),
         (result[columns["high"]] - previous).abs(),
         (result[columns["low"]] - previous).abs()], axis=1,
    ).max(axis=1)
    atr_pct = true_range.rolling(14, min_periods=14).mean() / close
    labels = np.full(len(result), np.nan)
    exit_offsets = np.full(len(result), np.nan)
    timestamp_index = {int(value): index for index, value in enumerate(timestamp_ns)}
    interval_ns = pd.Timedelta(minutes=int(candle_minutes or SPOT_CONFIG.candle_minutes)).value
    missing_future_rows = 0
    both_barrier_rows = 0
    timeout_rows = 0
    timeout_up_rows = 0
    timeout_down_rows = 0
    timeout_zero_rows = 0
    for index in range(len(result)):
        if not np.isfinite(atr_pct.iloc[index]):
            continue
        future_indices = [timestamp_index.get(int(timestamp_ns[index] + offset * interval_ns)) for offset in range(1, horizon_bars + 1)]
        if any(value is None for value in future_indices):
            missing_future_rows += 1
            continue
        upper = close_values[index] * (1.0 + profit_multiple * atr_pct.iloc[index])
        lower = close_values[index] * (1.0 - stop_multiple * atr_pct.iloc[index])
        for offset in range(1, horizon_bars + 1):
            future_index = int(future_indices[offset - 1])
            high = float(high_values[future_index])
            low = float(low_values[future_index])
            if high >= upper and low <= lower:
                both_barrier_rows += 1
                labels[index], exit_offsets[index] = np.nan, offset
                break
            if high >= upper:
                labels[index], exit_offsets[index] = 1.0, offset
                break
            if low <= lower:
                labels[index], exit_offsets[index] = 0.0, offset
                break
        if np.isnan(labels[index]) and np.isnan(exit_offsets[index]):
            timeout_rows += 1
            exit_offsets[index] = horizon_bars
            timeout_return = close_values[int(future_indices[-1])] - close_values[index]
            if timeout_return > 0:
                timeout_up = 1.0
                timeout_up_rows += 1
            elif timeout_return < 0:
                timeout_up = 0.0
                timeout_down_rows += 1
            else:
                timeout_up = 0.5
                timeout_zero_rows += 1
            labels[index] = timeout_up if timeout_policy == "sign" and timeout_up != 0.5 else 0.5 if timeout_policy == "neutral" else np.nan
    result["triple_barrier_label"] = labels
    result["triple_barrier_exit_bars"] = exit_offsets
    result["triple_barrier_atr_pct"] = atr_pct
    kept = result.dropna(subset=["triple_barrier_label"]).reset_index(drop=True)
    report = {
        "horizon_bars": int(horizon_bars), "profit_multiple": float(profit_multiple),
        "stop_multiple": float(stop_multiple), "rows_before": len(result),
        "rows_dropped": int(len(result) - len(kept)), "rows_kept": len(kept),
        "up_fraction": float(kept.triple_barrier_label.mean()) if len(kept) else None,
        "missing_future_rows": int(missing_future_rows),
        "both_barrier_rows": int(both_barrier_rows),
        "timeout_rows": int(timeout_rows),
        "timeout_up_rows": int(timeout_up_rows),
        "timeout_down_rows": int(timeout_down_rows),
        "timeout_zero_rows": int(timeout_zero_rows),
        "timeout_policy": timeout_policy,
    }
    return (result if return_full else kept), report


def sample_uniqueness_weights(frame: pd.DataFrame, horizon_bars: int, label_column: str = "label") -> pd.Series:
    """Return forward-overlap uniqueness weights for labeled events.

    An event is assigned the inverse of itself plus the number of later
    labeled events whose start falls inside its forecast interval. This is a
    deterministic causal approximation that can be passed as sample weights
    without inspecting labels after the event's configured horizon.
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    valid = frame[label_column].notna().to_numpy()
    weights = np.zeros(len(frame), dtype=float)
    valid_positions = np.flatnonzero(valid)
    for start in valid_positions:
        later = valid[start + 1:min(len(frame), start + horizon_bars)].sum()
        weights[start] = 1.0 / (1.0 + float(later))
    return pd.Series(weights, index=frame.index, name="sample_uniqueness_weight")


__all__ = ["make_spot_labels", "make_triple_barrier_labels", "sample_uniqueness_weights"]

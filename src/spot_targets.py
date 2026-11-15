"""ATR-scaled binary labels for the isolated XAG/USD spot pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_spot_labels(frame: pd.DataFrame, horizon_bars: int, k: float) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if horizon_bars <= 0 or k <= 0: raise ValueError("horizon_bars and k must be positive")
    result = frame.copy().sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    close = result["silver_close"].astype(float); high, low = result["silver_high"].astype(float), result["silver_low"].astype(float)
    previous = close.shift(1); true_range = pd.concat([(high-low).abs(),(high-previous).abs(),(low-previous).abs()],axis=1).max(axis=1); atr_pct = true_range.rolling(14,min_periods=14).mean()/close
    future = np.log(close.shift(-horizon_bars)/close); threshold = k*atr_pct
    valid = future.notna() & threshold.notna() & np.isfinite(future) & np.isfinite(threshold)
    label = pd.Series(np.nan,index=result.index,dtype=float); label.loc[valid & (future > threshold)] = 1; label.loc[valid & (future < -threshold)] = 0
    result["fwd_ret"] = future; result["atr_pct"] = atr_pct; result["target_threshold"] = threshold; result["label"] = label
    kept = result[result.label.notna()].copy().reset_index(drop=True)
    report = {"horizon_bars": horizon_bars, "horizon_minutes": horizon_bars*5, "k": k, "rows_before": len(result), "rows_dropped_no_move_or_warmup": int(len(result)-len(kept)), "rows_kept": len(kept), "up_fraction": float(kept.label.mean()) if len(kept) else None}
    return kept, report


__all__ = ["make_spot_labels"]

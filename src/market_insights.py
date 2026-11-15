"""Causal, trader-facing summaries derived from the latest closed candles."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def summarize_market(frame: pd.DataFrame, prediction: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return descriptive market state and levels without using future data."""
    required = {"datetime_et", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Market insights require: {', '.join(sorted(missing))}")
    data = frame.sort_values("datetime_et", kind="mergesort").copy()
    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=list(required))
    if len(data) < 51:
        raise ValueError("At least 51 closed candles are required for market insights.")
    close = data["close"]
    returns = close.pct_change(fill_method=None) * 100.0
    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    vol20 = returns.rolling(20, min_periods=20).std()
    vol100 = returns.rolling(100, min_periods=100).std()
    last = data.iloc[-1]
    last_date = pd.Timestamp(last["datetime_et"]).date()
    session = data[pd.to_datetime(data["datetime_et"]).dt.date == last_date]
    prior = data[pd.to_datetime(data["datetime_et"]).dt.date < last_date]
    prior_session = prior[pd.to_datetime(prior["datetime_et"]).dt.date == prior["datetime_et"].map(lambda value: pd.Timestamp(value).date()).max()] if not prior.empty else prior

    close_value = float(last["close"])
    sma20_value, sma50_value = float(sma20.iloc[-1]), float(sma50.iloc[-1])
    vol_ratio = float(vol20.iloc[-1] / vol100.iloc[-1]) if vol100.iloc[-1] and np.isfinite(vol100.iloc[-1]) else None
    trend_up = close_value > sma20_value > sma50_value
    trend_down = close_value < sma20_value < sma50_value
    trend = "UPTREND" if trend_up else "DOWNTREND" if trend_down else "MIXED / RANGE"
    volatility = "EXPANDING" if vol_ratio is not None and vol_ratio >= 1.25 else "CONTRACTING" if vol_ratio is not None and vol_ratio <= 0.80 else "NORMAL"
    bias = "LONG BIAS" if trend_up else "SHORT BIAS" if trend_down else "WAIT FOR BREAKOUT"
    insight = {
        "timestamp": str(last["datetime_et"]),
        "close": close_value,
        "trend": {"state": trend, "bias": bias, "close_vs_sma20_pct": (close_value / sma20_value - 1) * 100, "close_vs_sma50_pct": (close_value / sma50_value - 1) * 100},
        "volatility": {"state": volatility, "ratio_to_100_bar": vol_ratio, "volatility_20_pct": _finite(vol20.iloc[-1])},
        "levels": {
            "session_high": float(session["high"].max()), "session_low": float(session["low"].min()),
            "prior_session_high": float(prior_session["high"].max()) if not prior_session.empty else None,
            "prior_session_low": float(prior_session["low"].min()) if not prior_session.empty else None,
        },
        "decision_context": "Use directional forecasts only when the model says ACTIONABLE; expanding volatility increases risk and mixed trend favors waiting for a level break.",
        "prediction": dict(prediction) if prediction else None,
    }
    return insight


__all__ = ["summarize_market"]

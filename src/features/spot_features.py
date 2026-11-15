"""Switchable 34-feature causal schema for Dukascopy XAG/USD spot."""

from __future__ import annotations

import numpy as np
import pandas as pd

SILVER_PRICE_FEATURES = ("ret_1", "ret_5", "ret_15", "ret_60", "range_pct", "body_ratio", "upper_wick", "lower_wick")
VOLATILITY_FEATURES = ("atr_14_pct", "rv_20", "vol_ratio", "parkinson_20")
TREND_FEATURES = ("rsi_14", "macd_hist_c", "bb_pctb", "ema_dist_21", "ema_dist_50", "ema_dist_200", "zscore_60")
CROSS_MARKET_FEATURES = ("gold_ret_1", "gold_ret_5", "gold_ret_15", "gsr", "gsr_z", "gsr_chg_5", "eur_ret_5", "lead_lag", "corr_gold_50")
TIME_FEATURES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_overlap", "mins_since_open")
FEATURE_GROUPS = {"silver_price": SILVER_PRICE_FEATURES, "volatility": VOLATILITY_FEATURES, "trend_momentum": TREND_FEATURES, "cross_market": CROSS_MARKET_FEATURES, "time_session": TIME_FEATURES}
SPOT_FEATURE_COLUMNS = tuple(value for group in FEATURE_GROUPS.values() for value in group)


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    return 100 - 100 / (1 + _div(gain, loss))


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Spot feature frame is missing: {', '.join(missing)}")


def add_silver_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame, ["silver_close", "silver_open", "silver_high", "silver_low"])
    result = frame.copy(); close, open_price, high, low = result.silver_close, result.silver_open, result.silver_high, result.silver_low
    log_close = np.log(close.replace(0, np.nan)); candle_range = (high - low).replace(0, np.nan)
    for period, name in ((1, "ret_1"), (5, "ret_5"), (15, "ret_15"), (60, "ret_60")): result[name] = log_close.diff(period)
    result["range_pct"] = _div(high - low, close); result["body_ratio"] = _div(close - open_price, candle_range)
    result["upper_wick"] = _div(high - pd.concat([open_price, close], axis=1).max(axis=1), candle_range)
    result["lower_wick"] = _div(pd.concat([open_price, close], axis=1).min(axis=1) - low, candle_range)
    return result


def add_volatility_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(); close, high, low = result.silver_close, result.silver_high, result.silver_low
    previous = close.shift(1)
    true_range = pd.concat([(high-low).abs(), (high-previous).abs(), (low-previous).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean(); rv20 = result.ret_1.rolling(20, min_periods=20).std(); rv60 = result.ret_1.rolling(60, min_periods=60).std()
    result["atr_14_pct"] = _div(atr, close); result["rv_20"] = rv20; result["vol_ratio"] = _div(rv20, rv60)
    result["parkinson_20"] = np.sqrt((np.log(_div(high, low)).pow(2).rolling(20, min_periods=20).mean()) / (4*np.log(2)))
    return result


def add_trend_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(); close = result.silver_close; atr = result.atr_14_pct * close
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean(); ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean(); macd = ema12 - ema26; signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["rsi_14"] = _rsi(close, 14); result["macd_hist_c"] = _div(macd - signal, close)
    sma20 = close.rolling(20, min_periods=20).mean(); std20 = close.rolling(20, min_periods=20).std(); upper, lower = sma20 + 2*std20, sma20 - 2*std20
    result["bb_pctb"] = _div(close-lower, upper-lower)
    for period in (21,50,200): result[f"ema_dist_{period}"] = _div(close - close.ewm(span=period, adjust=False, min_periods=period).mean(), atr)
    sma60 = close.rolling(60, min_periods=60).mean(); result["zscore_60"] = _div(close-sma60, close.rolling(60, min_periods=60).std())
    return result


def add_cross_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame, ["xauusd_close", "eurusd_close"])
    result = frame.copy(); silver, gold, eur = result.silver_close, result.xauusd_close, result.eurusd_close
    gold_log, eur_log, silver_log = np.log(gold), np.log(eur), np.log(silver)
    for period, name in ((1,"gold_ret_1"),(5,"gold_ret_5"),(15,"gold_ret_15")): result[name] = gold_log.diff(period)
    result["gsr"] = _div(gold, silver); result["gsr_z"] = _div(result.gsr-result.gsr.rolling(100,min_periods=100).mean(), result.gsr.rolling(100,min_periods=100).std()); result["gsr_chg_5"] = np.log(result.gsr).diff(5)
    result["eur_ret_5"] = eur_log.diff(5); result["lead_lag"] = result.gold_ret_1.shift(1) - silver_log.diff().shift(1); result["corr_gold_50"] = silver_log.diff().rolling(50,min_periods=50).corr(gold_log.diff())
    return result


def add_time_session_features(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame, ["timestamp"]); result = frame.copy(); timestamps = pd.to_datetime(result.timestamp, utc=True); hour = timestamps.dt.hour + timestamps.dt.minute/60; dow = timestamps.dt.dayofweek
    result["hour_sin"] = np.sin(2*np.pi*hour/24); result["hour_cos"] = np.cos(2*np.pi*hour/24); result["dow_sin"] = np.sin(2*np.pi*dow/7); result["dow_cos"] = np.cos(2*np.pi*dow/7); result["is_overlap"] = ((hour>=8)&(hour<12)).astype(float); result["mins_since_open"] = hour*60 + timestamps.dt.minute*0
    return result


def generate_spot_features(frame: pd.DataFrame, groups: tuple[str, ...] | list[str] | None = None) -> pd.DataFrame:
    selected = tuple(groups or FEATURE_GROUPS)
    unknown = set(selected).difference(FEATURE_GROUPS)
    if unknown: raise ValueError(f"Unknown spot feature groups: {', '.join(sorted(unknown))}")
    result = frame.copy().sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    result = add_silver_price_features(result)
    if "volatility" in selected or "trend_momentum" in selected: result = add_volatility_features(result)
    if "trend_momentum" in selected: result = add_trend_momentum_features(result)
    if "cross_market" in selected: result = add_cross_market_features(result)
    if "time_session" in selected: result = add_time_session_features(result)
    selected_columns = tuple(column for group in selected for column in FEATURE_GROUPS[group])
    result[list(selected_columns)] = result[list(selected_columns)].replace([np.inf, -np.inf], np.nan)
    retained = [column for column in result.columns if column not in SPOT_FEATURE_COLUMNS or column in selected_columns]
    return result[retained]


__all__ = ["FEATURE_GROUPS", "SPOT_FEATURE_COLUMNS", "generate_spot_features"]

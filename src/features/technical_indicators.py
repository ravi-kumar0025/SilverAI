"""The approved 17-feature engineering pipeline for SilverAI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import prepare_intraday_data


FEATURE_COLUMNS = (
    "return_1",
    "return_3",
    "return_6",
    "volatility_5",
    "volatility_20",
    "volatility_50",
    "sma_distance_20",
    "sma_distance_50",
    "candle_return",
    "candle_range_pct",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "relative_volume",
    "volume_change",
    "hour_sin",
    "hour_cos",
    "day_of_week",
)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}.")


def _safe_percent_change(series: pd.Series, periods: int) -> pd.Series:
    result = series.pct_change(periods, fill_method=None) * 100.0
    previous = series.shift(periods)
    zero_denominator = previous.eq(0) & previous.notna() & series.notna()
    result = result.replace([np.inf, -np.inf], 0.0)
    result.loc[zero_denominator] = 0.0
    return result


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-, three-, and six-bar percentage close returns."""

    _require_columns(df, ("close",))
    result = df.copy()
    close = result["close"]
    result["return_1"] = _safe_percent_change(close, 1)
    result["return_3"] = _safe_percent_change(close, 3)
    result["return_6"] = _safe_percent_change(close, 6)
    return result


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling sample standard deviations of one-bar returns."""

    _require_columns(df, ("return_1",))
    result = df.copy()
    returns = result["return_1"]
    result["volatility_5"] = returns.rolling(window=5).std()
    result["volatility_20"] = returns.rolling(window=20).std()
    result["volatility_50"] = returns.rolling(window=50).std()
    return result


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add close-price distance from the 20- and 50-bar simple moving averages."""

    _require_columns(df, ("close",))
    result = df.copy()
    close = result["close"]
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    result["sma_distance_20"] = np.where(
        sma_20.isna(), np.nan, np.where(sma_20 != 0, (close / sma_20) - 1.0, 0.0)
    )
    result["sma_distance_50"] = np.where(
        sma_50.isna(), np.nan, np.where(sma_50 != 0, (close / sma_50) - 1.0, 0.0)
    )
    return result


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candle return, range, and wick ratios."""

    _require_columns(df, ("open", "high", "low", "close"))
    result = df.copy()
    candle_range = result["high"] - result["low"]
    result["candle_return"] = np.where(
        result["open"] != 0,
        ((result["close"] - result["open"]) / result["open"]) * 100.0,
        0.0,
    )
    result["candle_range_pct"] = np.where(
        result["close"] != 0,
        (candle_range / result["close"]) * 100.0,
        0.0,
    )
    upper_wick = result["high"] - result[["open", "close"]].max(axis=1)
    lower_wick = result[["open", "close"]].min(axis=1) - result["low"]
    result["upper_wick_ratio"] = np.where(
        candle_range > 0, upper_wick / candle_range, 0.0
    )
    result["lower_wick_ratio"] = np.where(
        candle_range > 0, lower_wick / candle_range, 0.0
    )
    return result


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 20-bar relative-volume and one-bar volume-change features."""

    _require_columns(df, ("volume",))
    result = df.copy()
    volume = result["volume"]
    volume_sma_20 = volume.rolling(window=20).mean()
    previous_volume = volume.shift(1)
    result["relative_volume"] = np.where(
        volume_sma_20 > 0, volume / volume_sma_20, 0.0
    )
    result["volume_change"] = np.where(
        previous_volume > 0,
        ((volume - previous_volume) / previous_volume) * 100.0,
        0.0,
    )
    return result


def add_time_features(
    df: pd.DataFrame, timestamp_col: str = "datetime_et"
) -> pd.DataFrame:
    """Add cyclical Eastern-Time hour encodings and day-of-week."""

    _require_columns(df, (timestamp_col,))
    result = df.copy()
    timestamps = result[timestamp_col]
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"Column '{timestamp_col}' must be timezone-aware Eastern Time.")
    hour_float = timestamps.dt.hour + timestamps.dt.minute / 60.0
    result["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)
    result["day_of_week"] = timestamps.dt.dayofweek
    return result


def generate_features(
    df: pd.DataFrame, timestamp_col: str = "datetime_et"
) -> pd.DataFrame:
    """Prepare OHLCV data and add exactly the 17 approved features.

    The input is copied, sorted chronologically, and left unchanged. Rolling
    warm-up values remain NaN; no rows are removed and no values are filled.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    _require_columns(df, (timestamp_col, *_OHLCV_COLUMNS))
    result = prepare_intraday_data(df, timestamp_col=timestamp_col)
    result = add_momentum_features(result)
    result = add_volatility_features(result)
    result = add_trend_features(result)
    result = add_candle_features(result)
    result = add_volume_features(result)
    result = add_time_features(result, timestamp_col=timestamp_col)
    return result


__all__ = [
    "FEATURE_COLUMNS",
    "add_candle_features",
    "add_momentum_features",
    "add_time_features",
    "add_trend_features",
    "add_volatility_features",
    "add_volume_features",
    "generate_features",
]

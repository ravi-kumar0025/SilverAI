"""Validation and non-destructive preparation of SilverAI market data."""

from __future__ import annotations

import re

import pandas as pd


_EASTERN_TIMEZONE = "America/New_York"
_OFFSET_PATTERN = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_OHLCV_COLUMNS = (*_PRICE_COLUMNS, "volume")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}.")


def _parse_eastern_timestamps(values: pd.Series, column: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"Column '{column}' contains missing timestamps.")

    text_values = values.astype("string").str.strip()
    has_offset = text_values.str.contains(_OFFSET_PATTERN, na=False)
    if has_offset.any() and not has_offset.all():
        raise ValueError(f"Column '{column}' mixes timezone-aware and naive timestamps.")

    try:
        if has_offset.all():
            return pd.to_datetime(text_values, errors="raise", utc=True).dt.tz_convert(
                _EASTERN_TIMEZONE
            )

        return pd.to_datetime(text_values, errors="raise").dt.tz_localize(
            _EASTERN_TIMEZONE, ambiguous="raise", nonexistent="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse column '{column}' as timestamps.") from exc


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return row-level OHLCV validation results without removing invalid rows.

    The returned DataFrame retains the input index.  Its ``valid`` column is
    false for rows with nonnumeric or missing prices, nonpositive prices,
    invalid OHLC geometry, or invalid volume.  Volume is checked when present.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    _require_columns(df, _PRICE_COLUMNS)

    numeric = {
        column: pd.to_numeric(df[column], errors="coerce")
        for column in _OHLCV_COLUMNS
        if column in df.columns
    }
    prices_numeric = pd.concat(
        [numeric[column] for column in _PRICE_COLUMNS], axis=1
    ).notna().all(axis=1)
    prices_finite = pd.concat(
        [numeric[column].replace([float("inf"), float("-inf")], pd.NA)
         for column in _PRICE_COLUMNS],
        axis=1,
    ).notna().all(axis=1)
    price_positive = pd.concat(
        [numeric[column] > 0 for column in _PRICE_COLUMNS], axis=1
    ).all(axis=1)
    high_valid = (numeric["high"] >= numeric["open"]) & (
        numeric["high"] >= numeric["close"]
    )
    low_valid = (numeric["low"] <= numeric["open"]) & (
        numeric["low"] <= numeric["close"]
    )

    report = pd.DataFrame(
        {
            "prices_numeric": prices_numeric,
            "prices_finite": prices_finite,
            "prices_positive": price_positive,
            "high_ge_open_close": high_valid,
            "low_le_open_close": low_valid,
        },
        index=df.index,
    )

    if "volume" in numeric:
        volume = numeric["volume"]
        report["volume_numeric"] = volume.notna()
        report["volume_finite"] = volume.replace(
            [float("inf"), float("-inf")], pd.NA
        ).notna()
        report["volume_nonnegative"] = volume >= 0

    report["valid"] = report.all(axis=1)
    return report


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values in each column without filling them."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    return df.isna().sum()


def _prepare_data(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    _require_columns(df, (timestamp_column, *_OHLCV_COLUMNS))

    prepared = df.copy()
    prepared[timestamp_column] = _parse_eastern_timestamps(
        prepared[timestamp_column], timestamp_column
    )
    prepared = prepared.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    prepared.attrs["duplicate_timestamp_count"] = int(
        prepared.duplicated(subset=[timestamp_column]).sum()
    )
    return prepared


def prepare_intraday_data(
    df: pd.DataFrame, timestamp_col: str = "datetime_et"
) -> pd.DataFrame:
    """Parse, sort, and validate the structure of intraday OHLCV data.

    Duplicate timestamps and OHLCV-invalid rows are preserved for inspection;
    use ``df.duplicated(timestamp_col)`` and :func:`validate_ohlcv` to inspect
    them explicitly.
    """

    return _prepare_data(df, timestamp_col)


def prepare_daily_data(df: pd.DataFrame, timestamp_col: str = "date") -> pd.DataFrame:
    """Parse, sort, and validate roll-adjusted daily OHLCV data independently."""

    return _prepare_data(df, timestamp_col)


def check_time_gaps(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Return observed intervals longer than the dataset's typical interval.

    The typical interval is inferred from the mode of positive consecutive
    differences.  The result reports each longer interval and does not label
    market-session breaks as errors.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    _require_columns(df, (timestamp_col,))

    timestamps = _parse_eastern_timestamps(df[timestamp_col], timestamp_col)
    ordered = timestamps.sort_values(kind="stable").reset_index(drop=True)
    deltas = ordered.diff()
    positive_deltas = deltas[deltas > pd.Timedelta(0)]
    if positive_deltas.empty:
        return pd.DataFrame(
            columns=["previous_timestamp", "timestamp", "gap", "gap_minutes"]
        )

    typical_interval = positive_deltas.mode().iloc[0]
    gap_mask = deltas > typical_interval
    gap_rows = pd.DataFrame(
        {
            "previous_timestamp": ordered.shift(1)[gap_mask].reset_index(drop=True),
            "timestamp": ordered[gap_mask].reset_index(drop=True),
            "gap": deltas[gap_mask].reset_index(drop=True),
        }
    )
    gap_rows["gap_minutes"] = gap_rows["gap"].dt.total_seconds() / 60
    gap_rows.attrs["typical_interval"] = typical_interval
    gap_rows.attrs["gap_count"] = len(gap_rows)
    return gap_rows


__all__ = [
    "check_missing_values",
    "check_time_gaps",
    "prepare_daily_data",
    "prepare_intraday_data",
    "validate_ohlcv",
]

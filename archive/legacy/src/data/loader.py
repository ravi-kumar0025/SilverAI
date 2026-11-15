
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_EASTERN_TIMEZONE = "America/New_York"
_OFFSET_PATTERN = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_INTRADAY_COLUMNS = ("datetime_et", *_OHLCV_COLUMNS)
_DAILY_COLUMNS = ("date", *_OHLCV_COLUMNS)
_FEATURE_COLUMNS = (
    *_INTRADAY_COLUMNS,
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
_MODELING_COLUMNS = (*_FEATURE_COLUMNS, "future_return_15m", "target_direction_15m")


def _dataset_path(filename: str, data_dir: str | Path | None) -> Path:
    """Return a dataset path, using the repository's ``data`` directory by default."""

    return Path(data_dir) / filename if data_dir is not None else _DATA_DIR / filename


def _parse_eastern_timestamps(
    values: pd.Series, *, column: str, source: Path
) -> pd.Series:
    """Parse timestamp values and return a timezone-aware Eastern Time series."""

    if values.isna().any():
        raise ValueError(f"Dataset {source} contains missing values in '{column}'.")

    text_values = values.astype("string").str.strip()
    has_offset = text_values.str.contains(_OFFSET_PATTERN, na=False)
    if has_offset.any() and not has_offset.all():
        raise ValueError(
            f"Dataset {source} contains mixed timezone-aware and naive values in '{column}'."
        )

    try:
        if has_offset.all():
            # UTC parsing handles daylight-saving offset changes, then exposes
            # the timestamps in the project's existing Eastern Time zone.
            parsed = pd.to_datetime(text_values, errors="raise", utc=True)
            return parsed.dt.tz_convert(_EASTERN_TIMEZONE)

        parsed = pd.to_datetime(text_values, errors="raise")
        return parsed.dt.tz_localize(
            _EASTERN_TIMEZONE, ambiguous="raise", nonexistent="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not parse '{column}' as timestamps in dataset {source}."
        ) from exc


def _load_csv(
    filename: str,
    *,
    timestamp_column: str,
    required_columns: Iterable[str],
    data_dir: str | Path | None,
) -> pd.DataFrame:
    """Load and validate one CSV dataset without changing its row structure."""

    path = _dataset_path(filename, data_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Expected the SilverAI data file '{filename}'."
        )

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read dataset file {path}: {exc}") from exc

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Dataset {path} is missing required columns: {', '.join(missing)}."
        )

    frame[timestamp_column] = _parse_eastern_timestamps(
        frame[timestamp_column], column=timestamp_column, source=path
    )
    return frame


def load_intraday_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load raw one-minute silver futures OHLCV data.

    The ``datetime_et`` column is returned as timezone-aware
    ``America/New_York`` timestamps.  All source rows and columns are retained.
    """

    return _load_csv(
        "SI_1m_1098d.csv",
        timestamp_column="datetime_et",
        required_columns=_INTRADAY_COLUMNS,
        data_dir=data_dir,
    )


def load_5m_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load validated five-minute silver candle OHLCV data."""

    return _load_csv(
        "silver_5m.csv",
        timestamp_column="datetime_et",
        required_columns=_INTRADAY_COLUMNS,
        data_dir=data_dir,
    )


def load_feature_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the five-minute OHLCV data with its engineered feature columns."""

    return _load_csv(
        "silver_features.csv",
        timestamp_column="datetime_et",
        required_columns=_FEATURE_COLUMNS,
        data_dir=data_dir,
    )


def load_modeling_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the feature dataset with its 15-minute modeling targets."""

    return _load_csv(
        "silver_modeling_data.csv",
        timestamp_column="datetime_et",
        required_columns=_MODELING_COLUMNS,
        data_dir=data_dir,
    )


def load_daily_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load daily roll-adjusted silver OHLCV data.

    The date is represented as midnight in ``America/New_York`` so the return
    value has the same explicit Eastern Time convention as the intraday data.
    """

    return _load_csv(
        "SI_daily_roll-adjusted.csv",
        timestamp_column="date",
        required_columns=_DAILY_COLUMNS,
        data_dir=data_dir,
    )


__all__ = [
    "load_5m_data",
    "load_daily_data",
    "load_feature_data",
    "load_intraday_data",
    "load_modeling_data",
]

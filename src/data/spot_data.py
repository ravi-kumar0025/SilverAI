"""Dukascopy spot data loading, complete-candle construction and alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.spot_config import SPOT_CONFIG, SpotConfig


INSTRUMENTS = ("xagusd", "xauusd", "eurusd")
OHLC = ("open", "high", "low", "close")


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", *OHLC}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}")
    timestamp = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="raise") if pd.api.types.is_numeric_dtype(frame["timestamp"]) else pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.assign(timestamp=timestamp)
    columns = ["timestamp", *OHLC] + (["volume"] if "volume" in frame.columns else [])
    return frame[columns].sort_values("timestamp", kind="mergesort")


def load_raw_instrument(instrument: str, config: SpotConfig = SPOT_CONFIG) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = config.root_data
    direct = root / f"{instrument}_1m.csv"
    paths = [direct] if direct.is_file() else sorted((root / "chunks" / instrument).glob(f"{instrument}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No Dukascopy files found for {instrument} under {root}")
    pieces = [_read_csv(path) for path in paths]
    frame = pd.concat(pieces, ignore_index=True).sort_values("timestamp", kind="mergesort")
    before = len(frame)
    frame = frame.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    numeric = frame.drop(columns=["timestamp"]).apply(pd.to_numeric, errors="coerce")
    frame.loc[:, numeric.columns] = numeric
    frame = frame.dropna(subset=list(OHLC)).reset_index(drop=True)
    gaps = frame["timestamp"].diff().dropna()
    report = {
        "instrument": instrument, "source_files": len(paths), "rows": int(len(frame)),
        "rows_before_deduplication": int(before), "duplicates_removed": int(before - len(frame)),
        "start": str(frame["timestamp"].iloc[0]), "end": str(frame["timestamp"].iloc[-1]),
        "volume_available": "volume" in frame.columns,
        "median_gap_minutes": float(gaps.dt.total_seconds().median() / 60) if not gaps.empty else None,
        "max_gap_hours": float(gaps.dt.total_seconds().max() / 3600) if not gaps.empty else None,
        "gaps_over_5m": int((gaps > pd.Timedelta(minutes=5)).sum()),
    }
    return frame, report


def build_candles(frame: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """Build only complete left-closed candles; volume stays absent if unavailable."""
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    indexed = frame.set_index("timestamp").sort_index()
    aggregations: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in indexed.columns:
        aggregations["volume"] = "sum"
    candles = indexed.resample(f"{minutes}min", closed="left", label="left").agg(aggregations)
    count = indexed["close"].resample(f"{minutes}min", closed="left", label="left").count()
    candles = candles.loc[count.eq(minutes)].dropna(subset=list(OHLC)).reset_index()
    candles["known_at"] = candles["timestamp"] + pd.Timedelta(minutes=minutes)
    return candles


def load_candles(instrument: str, minutes: int = 5, config: SpotConfig = SPOT_CONFIG) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, report = load_raw_instrument(instrument, config)
    candles = build_candles(raw, minutes)
    report.update({"candle_minutes": minutes, "complete_candles": int(len(candles)), "candle_start": str(candles.timestamp.iloc[0]), "candle_end": str(candles.timestamp.iloc[-1])})
    return candles, report


def align_market_data(minutes: int = 5, config: SpotConfig = SPOT_CONFIG) -> tuple[pd.DataFrame, dict[str, Any]]:
    loaded = {instrument: load_candles(instrument, minutes, config) for instrument in INSTRUMENTS}
    target = loaded["xagusd"][0].set_index("timestamp").sort_index()
    aligned = target.rename(columns={column: f"silver_{column}" for column in target.columns})
    coverage: dict[str, Any] = {instrument: loaded[instrument][1] for instrument in INSTRUMENTS}
    for instrument in ("xauusd", "eurusd"):
        other = loaded[instrument][0].set_index("timestamp").sort_index()
        other = other.reindex(aligned.index).ffill(limit=2)
        other = other.rename(columns={column: f"{instrument}_{column}" for column in other.columns})
        aligned = aligned.join(other, how="left")
    cross_columns = [column for column in aligned.columns if column.startswith(("xauusd_", "eurusd_"))]
    aligned["cross_market_complete"] = aligned[cross_columns].notna().all(axis=1)
    coverage["aligned"] = {"rows": int(len(aligned)), "cross_market_complete_rows": int(aligned.cross_market_complete.sum()), "coverage_ratio": float(aligned.cross_market_complete.mean())}
    return aligned.reset_index(), coverage


__all__ = ["INSTRUMENTS", "align_market_data", "build_candles", "load_candles", "load_raw_instrument"]

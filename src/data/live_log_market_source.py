"""Read closed, deduplicated one-minute live-log candles and build complete 5m bars."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.market_model_config import ROOT, SPOT_CONFIG


class LiveLogSource:
    def __init__(self, root: str | Path | None = None, *, clock=None):
        self.root = Path(root or ROOT / "data" / "live_log")
        self.clock = (clock if callable(clock) else (lambda: clock)) if clock is not None else (lambda: datetime.now(timezone.utc))
        self._offsets: dict[str, int] = {}
        self._latest: dict[str, dict[pd.Timestamp, dict[str, float]]] = {}

    def _raw(self, symbol: str) -> pd.DataFrame:
        path = self.root / f"{symbol}_m1_raw.jsonl"
        if not path.is_file(): return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        size = path.stat().st_size
        offset = self._offsets.get(symbol, 0)
        if size < offset:
            offset = 0
            self._latest[symbol] = {}
        latest = self._latest.setdefault(symbol, {})
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            lines = handle.readlines()
            self._offsets[symbol] = handle.tell()
        for line in lines:
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            received = pd.to_datetime(row.get("recv_ms"), unit="ms", utc=True, errors="coerce")
            for item in row.get("candles", []):
                if isinstance(item, list) and len(item) >= 5:
                    item = {"timestamp": item[0], "open": item[1], "high": item[2], "low": item[3], "close": item[4]}
                timestamp = pd.to_datetime(item.get("timestamp"), unit="ms", utc=True, errors="coerce")
                if pd.isna(timestamp) or pd.isna(received) or timestamp + pd.Timedelta(minutes=1) > received: continue
                latest[timestamp.floor("min")] = {key: float(item[key]) for key in ("open", "high", "low", "close")}
        if not latest: return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        return pd.DataFrame([{"timestamp": ts, **values} for ts, values in sorted(latest.items())])

    def candles(self, symbol: str = "xagusd", minutes: int = 5) -> pd.DataFrame:
        raw = self._raw(symbol)
        if raw.empty: return raw
        indexed = raw.set_index("timestamp").sort_index()
        agg = indexed.resample(f"{minutes}min", closed="left", label="left").agg({"open":"first","high":"max","low":"min","close":"last"})
        count = indexed["close"].resample(f"{minutes}min", closed="left", label="left").count()
        return agg.loc[count.eq(minutes)].dropna().reset_index()

    def status(self) -> dict[str, Any]:
        latest = []
        available = set()
        for symbol in ("xagusd", "xauusd", "eurusd"):
            raw = self._raw(symbol)
            if not raw.empty:
                latest.append(raw.timestamp.max())
                available.add(symbol)
        last = max(latest) if latest else None
        now = pd.Timestamp(self.clock())
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        age = None if last is None else max(0.0, (now - last).total_seconds())
        return {"provider": "dukascopy-live-log", "data_mode": "live", "is_live": True, "fresh": age is not None and age <= SPOT_CONFIG.live_stale_after_seconds, "last_update": None if last is None else last.isoformat(), "age_seconds": age, "stale_after_seconds": SPOT_CONFIG.live_stale_after_seconds, "companion_missing": sorted({"xagusd", "xauusd", "eurusd"} - available)}


__all__ = ["LiveLogSource"]

"""Stream the last stored day through the live-log/API path for a demo."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from src.data.market_candle_builder import load_raw_instrument


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--speed", type=float, default=0.05, help="seconds between replay minutes")
    args = parser.parse_args()
    frames = {}
    for symbol in ("xagusd", "xauusd", "eurusd"):
        frame, _ = load_raw_instrument(symbol)
        frames[symbol] = frame.tail(24 * 60).copy()
    events = []
    for symbol, frame in frames.items():
        for row in frame.itertuples(index=False):
            events.append((pd.Timestamp(row.timestamp), symbol, {"symbol": symbol, "timestamp": pd.Timestamp(row.timestamp).isoformat(), "open": float(row.open), "high": float(row.high), "low": float(row.low), "close": float(row.close)}))
    events.sort(key=lambda item: item[0])
    log_dir = Path("data/live_log")
    log_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.api_url, timeout=10) as client:
        for _, symbol, candle in events:
            log = {"poll_id": "replay", "sent_ms": int(time.time() * 1000), "recv_ms": int(time.time() * 1000), "candles": [[int(pd.Timestamp(candle["timestamp"]).timestamp() * 1000), candle["open"], candle["high"], candle["low"], candle["close"]]]}
            with (log_dir / f"{symbol}_m1_raw.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log) + "\n")
            client.post("/api/live/candles", json=candle).raise_for_status()
            if symbol == "xagusd": print(json.dumps(candle), flush=True)
            time.sleep(max(0.0, args.speed))


if __name__ == "__main__":
    main()

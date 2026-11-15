"""Build the reproducibility manifest for the canonical Dukascopy spot data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "dukascopy"
INSTRUMENTS = ("xagusd", "xauusd", "eurusd")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest() -> dict[str, object]:
    files: dict[str, object] = {}
    all_timestamps: list[pd.Series] = []
    for instrument in INSTRUMENTS:
        path = DATA_ROOT / f"{instrument}_1m.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=["timestamp"])
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        all_timestamps.append(timestamps)
        files[path.name] = {
            "sha256": _sha256(path),
            "rows": int(len(frame)),
            "start": str(timestamps.iloc[0]),
            "end": str(timestamps.iloc[-1]),
            "columns": ["timestamp", "open", "high", "low", "close"],
        }
    manifest: dict[str, object] = {
        "manifest_version": "xagusd_spot_manifest_v1",
        "instrument": "XAGUSD_SPOT",
        "instruments": ["XAG/USD", "XAU/USD", "EUR/USD"],
        "provider": "Dukascopy",
        "price_side": "bid",
        "timezone": "UTC",
        "source_frequency": "1m",
        "volume_present": False,
        "files": files,
        "date_range": {"start": str(min(series.iloc[0] for series in all_timestamps)), "end": str(max(series.iloc[-1] for series in all_timestamps))},
        "resampling": {"closed": "left", "label": "left", "complete_candle_rule": "exactly timeframe rows"},
        "git_commit_at_generation": _git_commit(),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["dataset_id"] = hashlib.sha256(canonical).hexdigest()[:16]
    return manifest


if __name__ == "__main__":
    destination = ROOT / "data" / "dataset_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    print(destination)
    print(json.dumps(json.loads(destination.read_text(encoding="utf-8")), indent=2))

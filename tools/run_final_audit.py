"""Read-only final submission audit for the XAG/USD spot project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.market_features import SPOT_FEATURE_COLUMNS
from src.market_model_config import SPOT_CONFIG


def main() -> int:
    data_root = ROOT / "data" / "raw" / "dukascopy"
    required = [data_root / f"{name}_1m.csv" for name in ("xagusd", "xauusd", "eurusd")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing canonical data files: {missing}")
    rows = {}
    for path in required:
        frame = pd.read_csv(path, usecols=["timestamp"])
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        rows[path.stem] = {
            "rows": len(frame),
            "start": str(timestamps.iloc[0]),
            "end": str(timestamps.iloc[-1]),
        }
    report = ROOT / "reports" / "spot_dukascopy_20261115"
    artifacts = [report / name for name in ("metrics.json", "report.md", "model_card.md", "data_quality.md", "spot_lightgbm.joblib", "spot_sigmoid_calibrator.joblib")]
    missing_artifacts = [str(path) for path in artifacts if not path.is_file()]
    if missing_artifacts:
        raise SystemExit(f"Missing submission artifacts: {missing_artifacts}")
    print(json.dumps({
        "instrument": SPOT_CONFIG.instrument,
        "symbol": SPOT_CONFIG.symbol,
        "source": SPOT_CONFIG.data_source,
        "candle_minutes": SPOT_CONFIG.candle_minutes,
        "feature_count": len(SPOT_FEATURE_COLUMNS),
        "features": list(SPOT_FEATURE_COLUMNS),
        "raw_files": rows,
        "artifacts": [path.name for path in artifacts],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

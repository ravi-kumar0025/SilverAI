# Reproducibility

Run commands from the repository root. The checked-in virtual environment is expected to provide the pinned project dependencies.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_alignment.py
.\.venv\Scripts\python.exe scripts/final_verify.py

Set-Location frontend
npm test
npm run build
```

`check_alignment.py` checks the active instrument identity, raw-data contract, required artifacts, and report consistency. `final_verify.py` checks dataset hashes, the 34-feature classifier contract, regression artifacts, indexed report figures, and the live endpoint when the backend is reachable. It writes the timestamped audit files under `reports/spot_v3_20261115/`.

## Input contract

The canonical inputs are `data/raw/dukascopy/xagusd_1m.csv`, `data/raw/dukascopy/xauusd_1m.csv`, and `data/raw/dukascopy/eurusd_1m.csv`. They contain UTC timestamps and bid OHLC without volume. Complete candles are reproducible at 5-, 15-, and 60-minute resolutions in the data pipeline. The API currently exposes 5-, 10-, 15-, and 30-minute market-candle views.

`data/dataset_manifest.json` records file hashes, row counts, date range, resampling rules, and dataset ID. The current manifest identifies the corpus as `4be8732dbab3aaa6`.

## Evidence and holdout policy

Evidence is stored in [`reports/spot_dukascopy_20261115/`](../reports/spot_dukascopy_20261115/), [`reports/spot_v3_20261115/`](../reports/spot_v3_20261115/), and [`reports/spot_v3_20261116_regression/`](../reports/spot_v3_20261116_regression/). The configured final holdout is reserved and marked `used_do_not_reuse`; do not rerun selection or tune thresholds against it.

The final audit is a timestamped repository/provider observation. A live pass means the endpoint was connected and fresh at audit time; it is not a permanent readiness certificate.

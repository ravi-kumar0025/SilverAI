# SilverAI — XAG/USD spot research terminal

SilverAI is a research and presentation project for XAG/USD spot using Dukascopy UTC bid data. It builds complete 5-minute candles from 1-minute files, aligns XAG/USD with XAU/USD and EUR/USD, creates causal features, and evaluates binary Up/Down signals with walk-forward validation.

The model is not a trading guarantee. The checked-in spot study reports per-fold metrics, calibration results, baselines, label audits, and leakage controls. A result above roughly 60% should be treated as a possible leakage or validation problem until independently verified.

## Current dataset

The canonical data lives under `data/raw/dukascopy/`:

| File | Role |
| --- | --- |
| `xagusd_1m.csv` | XAG/USD target, merged 1-minute UTC data |
| `xauusd_1m.csv` | XAU/USD cross-market data |
| `eurusd_1m.csv` | EUR/USD cross-market data |
| `*_5m.csv`, `*_15m.csv`, `*_1h.csv` | Regenerable complete-candle resolutions |

The source files contain bid OHLC and no volume. The pipeline does not invent volume. Candle `T` becomes known at `T + resolution`; features and live inference use only closed candles. Gold and EUR/USD may be forward-filled for at most two candles; XAG/USD is never filled.

The resumable downloader defaults to `2019-11-01` for a future history extension. It was not run in this checkout because network access is unavailable; the current manifest and dataset remain the 2023-11-01 onward corpus until the owner runs it and rebuilds the manifest.

## Research pipeline

```text
data/raw/dukascopy/*_1m.csv
  -> src.data.spot_data
  -> src.features.spot_features (34 switchable features)
  -> src.spot_targets (ATR-scaled binary labels)
  -> src.experiments.spot_dukascopy_study
  -> reports/spot_dukascopy_20261115
```

The 34 core features cover silver price action, volatility, trend/momentum, cross-market relationships, and UTC session context. Horizons are 15, 30, and 60 minutes (`H = 3, 6, 12` on 5-minute candles); the dead-zone multiplier is tested at `k = 0.2, 0.3, 0.5`. The final selected artifact uses `H=12` and `k=0.2`, a 60-minute horizon.

The API and rolling inference contract use the same ordered 34-feature schema. A closed candle produces a binary Up/Down probability; confidence below the configured threshold becomes `No signal`. The model artifact is available for reproducible research inference, but the report does not promote it as a reliable live trading model.

## Run

```powershell
# Backend, from the repository root (no --reload)
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Live helper, in a second terminal; requires the owner-supplied module
$env:SILVERAI_DUKASCOPY_MODULE = "dukascopy-node"
node scripts/live_feed_logger.mjs

# Frontend, in a third terminal
cd frontend
npm run dev
```

Verification commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm test
npm run build
```

The downloader is intentionally not part of the startup path. If it is run
later, use `\.venv\Scripts\python.exe download_dukascopy.py --out-dir <directory>`;
it refuses to replace existing CSVs unless `--force` is explicitly supplied.

Configuration is in [`config.yaml`](config.yaml). Environment defaults are documented in [`.env.example`](.env.example). The live-source decision and verification gate are in [`docs/spot_live_source.md`](docs/spot_live_source.md).

## Repository policy

Only source code, the canonical Dukascopy dataset, the current spot report, and reproducibility documentation are retained. Raw download chunks, old SI/futures datasets, old models, notebooks, caches, and frontend build output are disposable and should not be committed.

## Status

The current spot report is an evidence package, not a promoted production model. Historical data is always labelled historical in the app; no live readiness is implied.

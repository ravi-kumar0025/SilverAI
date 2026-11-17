# SilverAI

## XAG/USD research terminal

SilverAI is an evidence-oriented research platform for XAG/USD spot. It combines a causal market-data pipeline, a FastAPI prediction service, and a React/Vite financial terminal for historical analysis, controlled live-feed integration, and paper-only evaluation.

> Research software, not financial advice. The checked-in model is not presented as profitable, production-ready, or suitable for automated execution.

## Project status

| Area | Status |
| --- | --- |
| Historical application | Ready to run |
| Instrument | XAG/USD spot (`XAGUSD_SPOT`) |
| Historical source | Dukascopy bid OHLC, UTC, no volume |
| Model contract | 34 causal features; selected 60-minute horizon |
| Verification | 59 Python tests, 4 frontend tests, frontend build passing |
| Live mode | Conditional on an external provider and current feed verification |

Historical mode has been smoke-tested through the backend API and serves real checked-in candles with `data_status: historical`. Live mode remains an integration path, not a production-readiness claim.

## What the project does

- Builds complete 5-minute candles from one-minute XAG/USD, XAU/USD, and EUR/USD bid data.
- Generates a fixed 34-feature causal schema covering price action, volatility, trend/momentum, cross-market relationships, and UTC session context.
- Evaluates binary Up/Down targets at 15-, 30-, and 60-minute candidate horizons using chronological validation.
- Provides calibrated classifier probabilities, one-hour return estimates, confidence-based abstention, and explicit `NO_CLEAR_SIGNAL` states.
- Persists confirmed closed-candle forecasts and evaluates outcomes only when complete future windows are available.
- Exposes historical/live market status, diagnostics, paper-trading state, and model metadata through FastAPI.
- Presents the workflow in a chart-first React/Vite terminal.

The source contains bid OHLC only. SilverAI does not invent volume or use synthetic volume features.

## Quick start

Requirements: Python 3.12, Node.js, npm, and the checked-in dependencies. Run these commands from the repository root.

```powershell
# Terminal 1: start the backend in safe Historical mode
$env:SILVERAI_DATA_MODE = "historical"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: start the frontend
Set-Location frontend
npm run dev
```

Open the Vite URL printed by the frontend. The backend is available at `http://127.0.0.1:8000` and its interactive API documentation is available at `/docs`.

Important: a local `.env` may override the default data mode. Set `SILVERAI_DATA_MODE=historical` explicitly for a reproducible evaluator run.

## Optional live mode

Live mode requires an owner-supplied Dukascopy-compatible provider module. The bridge requests bid one-minute data for `xagusd`, `xauusd`, and `eurusd`, writes raw events to `data/live_log/`, and sends normalized closed candles to the backend.

```powershell
# Terminal 1: backend
$env:SILVERAI_DATA_MODE = "live"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: provider bridge
$env:SILVERAI_DUKASCOPY_MODULE = "dukascopy-node"
node scripts/live_feed_logger.mjs

# Terminal 3: frontend
Set-Location frontend
npm run dev
```

The system discards forming candles, de-duplicates events, requires companion markets for new scoring, and marks the feed stale after 180 seconds. Verify provider connection, companion availability, freshness, reconnect behavior, and persisted closed-candle predictions before describing live mode as verified.

## Verification

Run the complete local checks:

```powershell
# Backend, data, model, and report checks
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_alignment.py
.\.venv\Scripts\python.exe scripts/final_verify.py --skip-live

# Frontend checks
Set-Location frontend
npm test
npm run build
```

Use `scripts/final_verify.py` without `--skip-live` only when the backend and live endpoint are intentionally running. Its live result is a timestamped observation, not a permanent readiness certificate.

## Evidence and limitations

The current dataset manifest identifies the checked-in corpus as `4be8732dbab3aaa6`, covering 2023-11-01 through 2026-11-13. The selected research artifact uses a 60-minute forecast horizon (`H=12`) and is marked development/research-only.

The original final holdout is marked `used_do_not_reuse` and must not be reused for selection or threshold tuning. Missing or gapped future candles remain pending. The live bridge and paper-trading paths do not establish profitability or a completed five-day live evaluation.

The stored report package includes an explicit readiness review; two indexed report figures remain unavailable. This limitation is preserved rather than hidden or replaced with fabricated output.

## Repository layout

```text
backend/       FastAPI application, routes, and prediction service
frontend/      React/Vite financial terminal
src/           Data, features, models, prediction, and evaluation modules
data/          Canonical Dukascopy files and dataset manifest
reports/       Stored research, regression, threshold, and audit artifacts
docs/          API, architecture, methodology, reproducibility, and user guide
scripts/       Verification, live bridge, replay, manifest, and report helpers
tools/         Study runners and final audit tools
presentation/  Evaluator-facing presentation asset
```

## Documentation

- [API reference](docs/api.md)
- [Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Reproducibility](docs/reproducibility.md)
- [User guide](docs/user_guide.md)
- [Live-source verification gate](docs/spot_live_source.md)
- [Project decisions](DECISIONS.md)
- [Evaluation report](reports/spot_dukascopy_20261115/report.md)
- [Model card](reports/spot_dukascopy_20261115/model_card.md)
- [Readiness summary](reports/spot_v3_20261115/readiness_summary.md)

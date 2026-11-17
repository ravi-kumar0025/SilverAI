# SilverAI — XAG/USD research terminal

SilverAI is a research and presentation project for XAG/USD spot. It combines a FastAPI backend, a React/Vite market terminal, a reproducible Dukascopy data pipeline, and an evidence-oriented evaluation package.

The project is intentionally conservative about claims: historical data is labelled historical, live data is labelled live only when the configured provider is connected and fresh, and the model is not presented as a profitable or production-ready trading system.

## Evaluation snapshot

The checked-in artifacts describe the state of the repository at their generation time:

| Area | Current evidence |
| --- | --- |
| Instrument | XAG/USD spot (`XAGUSD_SPOT`) |
| Historical source | Dukascopy bid OHLC, UTC, 1-minute files resampled to complete 5-minute candles |
| Dataset | 2023-11-01 through 2026-11-13; dataset ID `4be8732dbab3aaa6` |
| Model input | 34 ordered causal features across price action, volatility, trend/momentum, cross-market context, and UTC session context |
| Selected forecast | Binary Up/Down study at a 60-minute horizon (`H=12`) |
| Checked-in audit | 13 passed, 1 failed; overall static/report status `REVIEW` because `fig_3_7_feature_groups_heatmap.png` and `fig_4_5_reliability.png` are missing |
| Live observation | The audit recorded a connected, fresh live-log observation at audit time; this is not a current or permanent readiness claim |
| Trading status | Research and paper-trading infrastructure only; no profitability or production-trading claim |

See the [readiness summary](reports/spot_v3_20261115/readiness_summary.md), [final audit](reports/spot_v3_20261115/final_audit.json), and [model card](reports/spot_v3_20261115/model_card.md) for supporting evidence and limitations.

## Current run readiness

The project is ready to run and evaluate in Historical mode. The following checks were completed in the current checkout:

| Check | Result |
| --- | --- |
| Backend import | Passed; FastAPI app exposes 28 routes |
| Backend smoke test | `/api/health` returned 200 with `status: ok` |
| Historical candle API | Returned 24 real candles with `data_status: historical` |
| Alignment audit | Passed |
| Python test suite | 59 passed |
| Frontend tests | 4 passed |
| Frontend production build | Passed with Vite |

The local `.env` sets `SILVERAI_DATA_MODE=live` and a live-provider module. To guarantee a safe historical evaluator run, set `SILVERAI_DATA_MODE=historical` in the backend terminal before starting Uvicorn; the process environment overrides the `.env` default.

The project is not certified as production-ready or profitable. Live mode remains conditional on a genuine provider connection, all three companion markets, closed-candle handling, freshness within 180 seconds, reconnect behavior, and persisted signal verification in the current environment.

## What is included

- A causal spot-data pipeline using `data/raw/dukascopy/{xagusd,xauusd,eurusd}_1m.csv`.
- Complete-candle resampling at 5-minute, 15-minute, and 1-hour resolutions. A candle is used only after its source interval has closed.
- A shared 34-feature schema used by research and API inference. The authoritative order is in [`src/features/spot_features.py`](src/features/spot_features.py).
- ATR dead-zone binary targets across 15-, 30-, and 60-minute candidate horizons and multiple dead-zone multipliers.
- Versioned classifier and regression artifacts, calibration support, prediction persistence, outcome evaluation, paper-trading state, and frozen-evaluation controls.
- A React dashboard with historical/live labels, candlestick data, predictions, diagnostics, evaluation, model, and paper-trading views.
- Read-only audit and alignment scripts for evaluator review.

The source dataset has no volume. SilverAI does not invent volume and does not use it as a feature.

## Repository map

```text
backend/                  FastAPI application and service layer
frontend/                 React/Vite terminal
src/data/                 Historical and live-provider data contracts
src/features/             Causal rolling and cross-market features
src/models/               Classifier, regression, calibration, and artifact contracts
src/prediction/           Persistence, orchestration, signals, and scenarios
src/evaluation/           Baselines, paper trading, selectivity, and frozen mode
data/                     Canonical data and dataset manifest
reports/                  Reproducibility and evaluation artifacts
docs/                     API, architecture, methodology, and user guidance
scripts/                  Dataset, live-feed, and verification helpers
tools/                    Study and final-audit entry points
```

## Run the application

From the repository root, use the checked-in virtual environment or an equivalent Python 3.12 environment.

```powershell
# Terminal 1: backend; historical mode is the safe default
$env:SILVERAI_DATA_MODE = "historical"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: frontend
Set-Location frontend
npm run dev
```

Open the Vite URL printed by the frontend. Historical mode serves real checked-in XAG/USD candles and reports `data_status: historical`.

### Optional live mode

Live mode requires an owner-supplied Dukascopy-compatible module and a running feed logger. It is an explicit opt-in and should be treated as unavailable until provider, companion symbols, closed-candle handling, freshness, and reconnect behavior are verified in the current environment.

```powershell
# Terminal 1: backend
$env:SILVERAI_DATA_MODE = "live"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: live logger; requires the configured external module
$env:SILVERAI_DUKASCOPY_MODULE = "dukascopy-node"
node scripts/live_feed_logger.mjs

# Terminal 3: frontend
Set-Location frontend
npm run dev
```

The frontend falls back to Historical mode when live status is unavailable. Never interpret a historical candle or a stale live feed as a live signal.

## Verification

These commands are read-only apart from normal test caches and generated audit output:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_alignment.py
.\.venv\Scripts\python.exe scripts/final_verify.py

# Static audit when no live backend/provider is running
.\.venv\Scripts\python.exe scripts/final_verify.py --skip-live

Set-Location frontend
npm test
npm run build
```

`check_alignment.py` checks active instrument identity, no-volume compliance, required artifacts, and manifest/report consistency. `final_verify.py` checks dataset hashes, the 34-feature artifact contract, regression artifacts, report figures, and the live endpoint when one is available. The final audit should be read as a timestamped observation, not a permanent readiness certificate.

Do not rerun model selection, threshold tuning, or holdout evaluation against the permanently closed holdout. New research requires an extended dataset and a newly frozen evaluation period.

## API and documentation

- [API reference](docs/api.md)
- [Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Reproducibility](docs/reproducibility.md)
- [User guide](docs/user_guide.md)
- [Live-source decision and verification gate](docs/spot_live_source.md)
- [Project decisions](DECISIONS.md)

Important endpoints include `GET /api/health`, `GET /api/market/candles`, `GET /api/market/status`, `GET /api/models`, `POST /api/predict/confirmed`, `GET /api/predictions`, `GET /api/evaluation/metrics`, and `GET /api/diagnostics`.

## Data and model limitations

- The historical corpus contains bid OHLC only; there is no volume field.
- Missing or gapped future candles remain unevaluated rather than being silently filled.
- Cross-market forward filling is bounded and XAG/USD is never fabricated.
- The configured holdout is marked `used_do_not_reuse`; it must not be reused for model selection.
- The checked-in model is development/research evidence. Its reported metrics do not establish live performance, profitability, or deployability.
- The live bridge and paper-trading paths are safeguards and integration points, not evidence of a five-day live validation result.

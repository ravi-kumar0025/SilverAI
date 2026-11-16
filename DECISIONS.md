# Spot XAG/USD decisions

## 2026-11-15 - switch instrument

- Decision: use Dukascopy XAG/USD bid candles as the target instrument.
- Evidence: complete XAG/USD, XAU/USD and EUR/USD 1-minute histories are present from 2023-11-01 through 2026-11-13/15 and are aligned on UTC.
- SI models: rejected and isolated. The spot study writes artifacts under `reports/spot_dukascopy_20261115`; no file under `saved_models/` is loaded.

## 2026-11-15 - volume features

- Decision: exclude volume features.
- Evidence: Dukascopy exports contain timestamp and OHLC only; no volume column exists in the raw files. Creating synthetic volume would violate data integrity.

## 2026-11-15 - model promotion gate

- Decision: do not promote the first spot candidate to the app.
- Evidence: the best cross-market candidate was `H=12`, `k=0.2`, with 54.28% mean walk-forward validation accuracy but 49.33% untouched holdout accuracy. The calibrated artifact produced no signals at thresholds 0.55-0.70 because the calibration block was not sufficiently discriminative.
- Next experiment: improve calibration and selective signal gating on validation only, then evaluate one frozen candidate on the holdout.

## 2026-11-15 - final feature and horizon contract

- Decision: use the complete 34-feature spot schema in training, rolling inference, the API, and the dashboard.
- Decision: expose the selected `H=12` configuration as the final 60-minute horizon while retaining 15/30/60-minute candidate comparisons in the report.
- Evidence: the selected artifact stores the same ordered 34-feature list; the final audit and backend model endpoint validate that order.

## Live-source gate

The historical source is Dukascopy bid data. A live server is not yet enabled because a verified real-time XAG/USD, XAU/USD and EUR/USD source with the same bid-price semantics has not been connected. yfinance is not an acceptable live match for this requirement.

## 2026-11-15 - live bridge boundary

- Decision: use a Node `getRealTimeRates` polling helper as the network boundary and a Python `DukascopyRealtimeSource` as the normalized server receiver.
- Contract: XAG/USD, XAU/USD and EUR/USD; m1; bid; volumes false; 20-second polling; forming candles rejected; duplicate timestamps rejected; gaps counted; heartbeat/freshness exposed through `/api/health`, `/api/market/status`, and `/api/provider/health`.
- Decision: `ReplaySource` must pass through the same receiver guards and is the only offline validation path in this checkout.
- Evidence: `scripts/live_feed_logger.mjs`, `src/data/live_market_provider.py`, `backend/routes/prediction_api_routes.py`, `tests/test_live_bridge.py`.
- Limitation: no internet or provider credentials were available. The live helper's installed `getRealTimeRates` module and unattended network behavior remain unverified; `pytest` could not run because pytest is not installed in the active Python environment. `scripts/verify_market_data_alignment.py` passed after this step.

## 2026-11-15 - no synthetic volume

- Decision: remove the chart's zero-volume bars, volume gradients, divider and `VOLUME` label.
- Decision: the candles API and user-facing docs state that Dukascopy spot volume is unavailable; no zero-valued volume is presented as data.
- Evidence: `frontend/src/ChartPanel.jsx`, `frontend/src/styles.css`, `docs/api.md`, `frontend/src/market_terminal_app.jsx`.
- Limitation: generic internal market-update contracts still accept optional quote volume for provider-neutral compatibility; it is not part of the spot API or feature schema. `scripts/verify_market_data_alignment.py` passed after this step.

## 2026-11-15 - single configuration source

- Decision: `config.yaml` is authoritative for spot identity, timeframe, candidate/final horizons, target k, confidence thresholds, costs, holdout dates/status, dataset/model paths, seed and evaluation settings.
- Decision: `src/project_config.py` and backend paths are compatibility views of `src.market_model_config.SPOT_CONFIG`; environment variables no longer override those research-contract values. `SILVERAI_DATA_MODE`, CORS and live-bridge runtime settings remain operational environment controls.
- Decision: frontend dependencies are pinned to the versions already represented in `frontend/package-lock.json` (`@vitejs/plugin-react` 6.2.1, Vite 8.3.6, React/React DOM 19.3.1).
- Holdout rule: the original 2026-10-01 through 2026-11-13 holdout is marked `used_do_not_reuse` and will never be evaluated again.
- Limitation: legacy archiving and a clean commit are deferred until the existing unrelated dirty/deleted worktree changes are classified; no pre-existing user changes will be staged implicitly. `scripts/verify_market_data_alignment.py` passed after this step.

## 2026-11-15 - first holdout permanently closed

- Decision: treat the 2026-10-01 through 2026-11-13 21:59 UTC holdout as used and never evaluate it again.
- Reconciliation: the stored model metadata and `metrics.json` select H=12, k=0.3. `config.yaml`, report, model card and metadata sidecar now identify that candidate and the original holdout ledger.
- Evidence: `reports/spot_dukascopy_20261115/metrics.json`, `reports/spot_dukascopy_20261115/report.md`, `reports/spot_dukascopy_20261115/model_card.md`, `reports/spot_dukascopy_20261115/artifact_metadata.json`, `reports/holdout_ledger.json`.
- Limitation: no one-run retraining/regeneration was performed because the existing study would reopen the used holdout. Holdout actual class rate and always-Up/majority baseline are UNKNOWN in stored artifacts. `scripts/verify_market_data_alignment.py` passed after this step.

## 2026-11-15 - history extension prepared, not executed

- Decision: the resumable Dukascopy downloader defaults to `2019-11-01` and retains 60-day chunks for retry/resume.
- Decision: downloader outputs exclude volume so the historical source remains OHLC-only.
- Limitation: no internet is available in this environment, so no chunks were downloaded, the manifest/dataset ID were not rebuilt, and no expanded-data evaluation was run. `scripts/verify_market_data_alignment.py` passed after this step.

## 2026-11-15 - advanced labels and selective reporting scaffolding

- Decision: add causal triple-barrier labels, overlap-aware sample-uniqueness weights, Wilson confidence intervals, validation-only selective metrics and act/abstain meta-label helpers.
- Study contract: configuration now records 5/15/60-minute candidate timeframes, five validation folds, ATR dead-zone and triple-barrier label methods. These are declared contracts only; no study was run.
- Holdout rule: the original holdout remains closed; no new label or selectivity code was evaluated against it.
- Evidence: `src/directional_labels.py`, `src/evaluation/selectivity.py`, `tests/test_advanced_targets.py`, `config.yaml`.
- Limitation: expanded-data, multi-timeframe, >3-fold, session/regime and reliability-plot results are not available until the owner supplies the extended data and a new frozen holdout. `scripts/verify_market_data_alignment.py` passed after this step.

## 2026-11-15 - new holdout and honest result boundary

- Decision: do not freeze or evaluate a new holdout until the owner runs the 2019 history extension and the resulting latest block is inspected.
- Decision: report the stored model as having no consistent confident-signal edge; do not claim costs, profitability, separate-period performance, reliability plots or session/regime edge without results.
- Evidence: `reports/holdout_ledger.json`, `reports/spot_dukascopy_20261115/report.md`, `reports/spot_dukascopy_20261115/model_card.md`.
- Limitation: 1x/2x/3x cost sensitivity and >3-fold/multi-timeframe results remain UNKNOWN/NOT STARTED. `scripts/verify_market_data_alignment.py` passed after this step.

## 2026-11-15 - legacy archive

- Decision: move the explicitly identified legacy SI/futures tree from `legacy/` to `archive/legacy/` so it is retained but not presented as active source.
- Evidence: the archive move was performed with exact resolved paths; active alignment scanning excludes the archive.
- Limitation: a clean commit was not created. The checkout still contains unrelated pre-existing modifications and deletions, so staging all changes would risk committing user work without classification.

## 2026-11-15 - target schema and freshness correction

- Decision: both ATR and triple-barrier targets use the explicit `config.yaml` price-column mapping (`timestamp`, `silver_open`, `silver_high`, `silver_low`, `silver_close`). Triple-barrier windows require exact configured timestamps through the time barrier; missing candles and weekend gaps are dropped.
- Decision: live freshness uses the injected replay/source clock and the configured `live_stale_after_seconds=90`; freshness is true through the exact 90-second boundary and false after it.
- Development-only result: on 256,222 pre-holdout XAG/USD 5m rows, H=12 and 0.3 barriers produced 153,853 labels: 77,853 down, 76,000 up, up fraction 49.3978%; 1,932 rows lacked an exact future window. See `reports/spot_dukascopy_20261115/triple_barrier_development.md`.
- Evidence: `src/directional_labels.py`, `src/data/live_market_provider.py`, `tests/test_advanced_targets.py`, `tests/test_live_bridge.py`, `config.yaml`.

## 2026-11-15 - serialized artifact runtime contract

- Decision: spot classifier and sigmoid calibrator artifacts must be versioned envelopes containing Python, joblib, NumPy, pandas, scikit-learn and LightGBM versions; the loader refuses missing metadata or any runtime mismatch.
- Root cause: the current stored artifacts were serialized under scikit-learn 1.7.1 while the project pins 1.6.1. The active spot artifact test loads `reports/spot_dukascopy_20261115/spot_lightgbm.joblib` and its sigmoid calibrator through `load_spot_artifacts()`; it does not use `LabelEncoder` or the multiclass calibrator.
- Isolation: `src/models/probability_calibration.py` is generic multiclass test infrastructure and is not imported by the active spot pipeline; archived SI code is under `archive/legacy/`. The active spot study uses binary LightGBM plus binary one-feature sigmoid logistic calibration.
- Limitation: regeneration was not performed because this host has Python 3.13.5, scikit-learn 1.7.1, pandas 2.3.1 and joblib 1.5.1, not the required Python 3.12/pinned environment; the corrected study now refuses to regenerate under that mismatch. `scripts/verify_market_data_alignment.py` passed after this correction.

## 2026-11-15 - test environment contract

- Decision: `requirements.txt` remains the pinned runtime contract; `requirements-dev.txt` adds `httpx==0.28.1`; `requirements.lock` records the same exact direct versions plus httpx.
- Exact intended commands: `py -3.12 -m venv .venv312`; `.venv312\\Scripts\\python -m pip install -r requirements-dev.txt`; `.venv312\\Scripts\\python -m pytest -q`.
- Evidence: the owner reported 36 passed and 2 failed in a fresh Python 3.12 environment before these fixes. This host has no Python 3.12 and no pytest, so a post-fix full pytest result cannot be claimed here.
- `scripts/verify_market_data_alignment.py` passed after the test-contract update.
## 2026-11-16 - active 15-minute model and chart presentation

- Decision: the active XAG/USD model is `xagusd_5m_15m_multiclass_regression_v1`, using three 5-minute bars for a 15-minute Down/Neutral/Up forecast plus mean and 10/50/90% return regression.
- Evidence: `reports/spot_15m_20261116_multiclass/metadata.json`, `data/dataset_manifest.json`, and `config.yaml`.
- Selected parameters: `k=0.5`, neutral confidence threshold `0.34`, 34-feature schema, dataset ID `3b0693b8941795ee`.
- Presentation: chart prediction-marker dots are disabled; prediction records and regression context remain available in the terminal pages.
- Scope: retired binary/60-minute reports remain historical and are not loaded by the active API.

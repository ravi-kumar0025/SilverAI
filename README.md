# SilverAI - XAG/USD 15-minute research terminal

SilverAI is a research and paper-trading terminal for XAG/USD spot. It uses UTC bid OHLC from Dukascopy, builds complete 5-minute candles, aligns XAG/USD with XAU/USD and EUR/USD, and generates a causal 34-feature input for the active 15-minute model.

The active artifact is `xagusd_5m_15m_multiclass_regression_v1`:

- `pred_1/classifier.joblib`: base Down/Neutral/Up LightGBM classifier.
- `pred_2/calibrated_predictor.joblib`: calibrated classifier, selected neutral threshold, and 15-minute mean plus 10/50/90% return regressors.
- Selected `k`: `0.5`; neutral confidence threshold: `0.34`.
- Forecast horizon: 15 minutes, or three 5-minute bars.

The model is research-only. Current untouched-test results are recorded in `reports/spot_15m_20261116_multiclass/metadata.json`; they do not establish profitability or live trading readiness.

## Current data

The canonical raw data is under `data/raw/dukascopy/`:

| File | Role |
| --- | --- |
| `xagusd_1m.csv` | XAG/USD target bid OHLC |
| `xauusd_1m.csv` | XAU/USD cross-market bid OHLC |
| `eurusd_1m.csv` | EUR/USD cross-market bid OHLC |
| `*_5m.csv`, `*_15m.csv`, `*_1h.csv` | Regenerable complete candles |

Dataset identity is recorded in `data/dataset_manifest.json`. The current canonical dataset ID is `3b0693b8941795ee`, covering 2023-11-01 through 2026-11-16 14:21 UTC. Training records any closed live-log extension used during the run in model metadata. No volume is invented or used.

## Active pipeline

```text
data/raw/dukascopy + data/live_log
  -> complete 5-minute UTC candles
  -> 34 causal features
  -> purged chronological train/calibration/validation/test split
  -> Down/Neutral/Up classifier + train-only calibration
  -> 15-minute return and 10/50/90% quantile regressors
  -> FastAPI + React paper-trading terminal
```

Incomplete feature rows are excluded before training so training and production inference share the same valid-input contract. The final test block is untouched by selection. The chart displays candles without signal-marker dots; signals and regression details remain available in the dashboard and prediction pages.

## Run

Use the same Python 3.13 runtime used to serialize the artifacts:

```powershell
# Backend
& "C:\Users\ravik\AppData\Local\Programs\Python\Python313\python.exe" -m uvicorn backend.server:app --host 127.0.0.1 --port 8000

# Live bridge, second terminal
$env:SILVERAI_DUKASCOPY_MODULE = "dukascopy-node"
node scripts/live_feed_logger.mjs

# Frontend, third terminal
cd frontend
npm run dev -- --host 127.0.0.1
```

The downloader is separate from startup. After changing canonical raw files, rebuild `data/dataset_manifest.json` before retraining.

## Verification

```powershell
& "C:\Users\ravik\AppData\Local\Programs\Python\Python313\python.exe" -m py_compile backend\services\prediction_pipeline_service.py scripts\train_spot_15m_multiclass.py
cd frontend
npm test
npm run build
```

The API reports the active model through `/api/models`, persists only closed-candle forecasts, and keeps live/paper status separate from historical evaluation. A live five-day test requires a verified provider, continuous closed-candle operation, reconnection handling, and fresh runtime evidence.

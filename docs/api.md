# SilverAI API

Start the FastAPI service from the repository root with the Python 3.13 runtime used for the serialized model:

```powershell
& "C:\Users\ravik\AppData\Local\Programs\Python\Python313\python.exe" -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

`GET /api/market/candles?limit=288` returns complete UTC XAG/USD candles. Dukascopy spot files contain bid OHLC only; volume is unavailable and is not returned.

`GET /api/market/status` and `GET /api/live/status` report mode, freshness, accepted closed candles, companion availability, duplicate counts, and gap events.

`GET /api/models` reports the active 34-feature contract, 15-minute horizon, calibration status, selected `k`, neutral threshold, classifier, regressor, and regression target. The active regression target is `log(close[t+3]/close[t])`.

`POST /api/predict` accepts one valid closed-candle feature observation. It returns calibrated Down/Neutral/Up probabilities and, when a close price is supplied, the 15-minute expected return, predicted price, and 10/50/90% regression range.

`POST /api/predict/confirmed` is the persistence boundary for a closed candle. `GET /api/predictions` returns immutable active-model forecasts. `POST /api/predictions/evaluate` settles eligible historical outcomes; live cycles settle outcomes after three complete future bars. Missing or gapped windows remain pending.

`GET /api/evaluation/metrics` reports active-model validation metrics, untouched-test regression metrics, and current recorded outcomes. Retired binary or 60-minute reports are not loaded. `GET /api/paper-trading` is analytical and paper-only.

The canonical instrument is `XAGUSD_SPOT` / `XAG/USD`, with UTC timestamps, 5-minute input candles, a 15-minute forecast horizon, and the ordered schema in `src/features/market_features.py`. Missing or invalid feature values are rejected rather than silently filled.

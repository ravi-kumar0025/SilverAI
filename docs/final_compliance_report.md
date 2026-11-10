# SilverAI final compliance report

## Scope

SilverAI remains committed to COMEX Silver Futures (SI). The retained research setup is five-minute candles with a three-bar, 15-minute forecast because the repository's feature, target, saved-model metadata, and reports are all built around that contract. Changing it would invalidate the current artifacts and is not justified by the available evidence.

## Model and data evidence

- Selected classifier: `hist_gradient_boosting_classifier`.
- Selected regressor: `linear_regression`.
- Input schema: 17 features in the saved metadata order.
- Horizon: 3 x 5-minute candles / 15 minutes.
- Dataset: `silver_5m.csv`, with timezone-aware Eastern timestamps and OHLCV validation.
- Validation: chronological train/validation/test design with purged boundaries.
- Known research limitation: regression performance is approximately equal to the zero-return baseline, and the current artifacts do not include calibrated probabilities.

## Implemented live-testing foundations

`src/data/market_data.py` defines a provider-neutral candle contract and an explicit historical provider. `src/prediction/store.py` provides append-only SQLite prediction persistence, duplicate prevention, and full-horizon outcome evaluation. `src/evaluation/paper_trading.py` computes paper-only gross/net P&L with transaction costs and slippage. `POST /api/predict/confirmed` is the closed-candle inference boundary intended for a future live adapter. `GET /api/market/status`, `GET /api/predictions`, `POST /api/predictions/evaluate`, and `GET /api/paper-trading` expose the validation state.

## External setup still required

The repository does not contain credentials or a verified live COMEX SI data provider. A provider must be selected, authenticated, tested against actual SI futures data, and connected to the confirmed-candle endpoint before the five-day test. The provider must supply contract/roll metadata, timestamps, volume, session state, reconnect behavior, and delayed/live status.

## Verification

```text
pytest -q                 27 passed, 1 warning
cd frontend; npm run build  passed
GET /api/market/status     historical, is_live=false
GET /api/paper-trading     paper_only=true
```

## Verdict

**NOT READY FOR FIVE-DAY LIVE TESTING.** The project is reproducible for historical research and has durable validation foundations, but it must not be represented as live-ready until the external market-data setup, calibrated confidence, live closed-candle loop, historical marker overlay, and end-to-end live verification are complete.

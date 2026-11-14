# SilverAI final compliance report

## Source brief and delivery option

The authoritative brief is `ps.pdf`, dated 27 October 2026. SilverAI uses Option B, the recommended Python-backed live chart web app, for one declared asset: COMEX Silver Futures (SI). The brief leaves timeframe and forecast horizon to the team; this project selects five-minute candles and a three-bar / 15-minute horizon because the available data, feature contract, target construction, saved artifacts, and chronological experiments all use that configuration.

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

The test-set baseline results are: always-UP accuracy 0.353815 / macro-F1
0.174231; always-DOWN accuracy 0.376152 / macro-F1 0.182224; majority-class
accuracy 0.376152 / macro-F1 0.182224; RSI-14 accuracy 0.306402 / macro-F1
0.269801. The selected classifier's recorded test macro-F1 is 0.37312. The
zero-return regression baseline has MAE 0.220179, RMSE 0.319404, and R2
-0.000103, so the selected regression model is not treated as evidence of a
useful continuous forecasting edge.

## Implemented live-testing foundations

`src/data/market_data.py` defines a provider-neutral candle contract and an explicit historical provider. `src/prediction/store.py` provides append-only SQLite prediction persistence, duplicate prevention, and full-horizon outcome evaluation. `src/evaluation/paper_trading.py` computes paper-only gross/net P&L with transaction costs and slippage. `POST /api/predict/confirmed` is the closed-candle inference boundary intended for a future live adapter. `GET /api/market/status`, `GET /api/predictions`, `POST /api/predictions/evaluate`, and `GET /api/paper-trading` expose the validation state.

## PDF live-test requirement

For silver, the brief requires five trading days. The submitted indicator cannot be changed during that test, every signal is checked against the realized movement over the stated horizon, and results are compared with simple baselines such as always-Up and an RSI rule. Paper trading only is permitted.

## External setup still required

The repository does not contain credentials or a verified live COMEX SI data provider. A provider must be selected, authenticated, tested against actual SI futures data, and connected to the confirmed-candle endpoint before the five-day test. The provider must supply contract/roll metadata, timestamps, volume, session state, reconnect behavior, and delayed/live status.

## Verification

```text
pytest -q                 31 passed, 2 warnings
cd frontend; npm run build  passed
GET /api/market/status     historical, is_live=false
GET /api/paper-trading     paper_only=true
```

## Verdict

**NOT READY FOR THE FIVE-TRADING-DAY LIVE TEST.** The project has reproducible research and durable validation foundations, but live submission remains blocked by provider credentials/entitlements, final frozen-artifact verification, automatic live-feed execution, and end-to-end live testing.

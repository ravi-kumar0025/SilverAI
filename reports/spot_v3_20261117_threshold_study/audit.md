# SilverAI regression threshold-study audit

## Scope and current contract

- Instrument: XAG/USD spot from the Dukascopy-derived 5-minute candle pipeline.
- Features: the ordered 34-column schema in `src/features/spot_features.py`.
- Target: `log(close[t+12] / close[t])`, a 12-bar / 60-minute horizon.
- Regression artifacts: mean, Q10, Q50, and Q90 LightGBM regressors under `reports/spot_v3_20261116_regression/`.
- Current model training: `scripts/train_spot_regression.py`; development data ends before the configured final holdout beginning `2026-10-01T00:00:00Z`.
- Current market path: closed-candle feature generation, historical CSV provider for replay, and an owner-run Dukascopy-compatible live-log bridge. The backend reports live freshness explicitly.
- Persistence: SQLite prediction store at `data/spot_predictions.sqlite3`; records are deduplicated and outcomes are evaluated after the configured horizon when future candles are available.
- Current paper trading: analytical historical evaluation in `src/evaluation/paper_trading.py` exposed by `GET /api/paper-trading`; it is not yet a stateful live paper-trading account.

## Incomplete relative to the pasted request

- No prior persisted chronological OOS prediction table for the four regressors.
- No threshold comparison at the requested bps grid, with per-fold results and frozen regression threshold.
- The live prediction service still combines the regression estimate with the existing classifier/scenario path; it does not yet make a selected regression threshold the sole BUY/SELL gate.
- Paper trading is an analytical replay over evaluated predictions, not a backend-managed running/stopped virtual account with open positions, account equity, duplicate-entry protection, and one-hour exits.
- No dedicated transactional `Clear Testing Data` endpoint or frontend confirmation flow was found.
- Live readiness remains conditional on the owner-run provider bridge and companion-market freshness; historical data is not silently treated as live.

## Integrity decisions for this phase

- The final holdout is excluded before all fitting and threshold selection.
- The study uses the existing LightGBM architecture and exact feature ordering.
- Thresholds are post-prediction parameters in decimal log-return units (`basis_points / 10000`).
- Trade-return metrics use non-overlapping one-hour windows and configured spread/slippage costs.
- No original model artifact, dataset, validation artifact, or application database was deleted or overwritten.

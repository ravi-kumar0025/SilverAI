# Spot methodology

The target is `log(close[t+H] / close[t])`. A row is `Up` when the return exceeds `k * ATR_pct`, `Down` when it is below `-k * ATR_pct`, and is removed otherwise. The study compares `H = 3, 6, 12` and `k = 0.2, 0.3, 0.5` on 5-minute candles.

Validation is chronological walk-forward with a purge and embargo of at least the forecast horizon. The final holdout is touched once. Reported metrics include accuracy, AUC, log loss, Brier score, class balance, dropped rows, calibration, confidence-threshold signal counts, baselines, and cross-market ablation.

The main leakage controls are closed-candle timestamps, train-only model fitting, no forward fill on XAG/USD, a two-candle maximum fill on companion markets, and no use of the final holdout for selection. See `reports/spot_dukascopy_20261115/report.md` for the current evidence.

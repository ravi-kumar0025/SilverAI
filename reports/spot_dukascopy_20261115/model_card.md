# XAG/USD spot model card

Metadata: dataset_id `4be8732dbab3aaa6`; commit `cc1e500c68af57dff4101570f56b9428dea4fe69`; config SHA-256 `1735aeb39347d811307e01577a006102e5aa3745bd34e01ddc613e50b92f2aa1`; seed `42`; holdout status `used_do_not_reuse`.

## Intended use

Research and classroom demonstration of a trader-facing indicator for XAG/USD spot. The artifact is not financial advice, an execution system, or evidence of profitability.

## Training contract

- Source: Dukascopy bid OHLC, UTC.
- Target: XAG/USD.
- Resolution: 5-minute complete candles.
- Horizons: 15, 30, and 60 minutes.
- Labels: Up/Down after dropping moves inside `k * ATR_pct`.
- Candidate `k`: 0.2, 0.3, and 0.5.
- Features: 34 causal features across five groups.
- Model: conservative LightGBM binary classifier.
- Calibration: sigmoid calibration fitted on a later chronological block.

## Data integrity

XAG/USD has 1,334,781 raw rows and 266,937 complete 5-minute candles. XAU/USD and EUR/USD are aligned on the XAG/USD index with a maximum two-candle forward fill. XAG/USD is never filled. Source volume is unavailable and no volume features are used.

## Results

The stored selected candidate was `H=12`, `k=0.3`, selected using walk-forward validation. Mean cross-market validation accuracy was 53.27%; stored untouched holdout accuracy was 49.03%, AUC 0.5070, and balanced accuracy 50.29%. Stored calibrated holdout accuracy was 54.61% at the same AUC, with 1,689 signals at .55, 19 at .60, 1 at .65 and zero at .70. The original holdout is permanently closed and was not re-evaluated.

## Limitations

The candidate is not promoted. The dataset is historical, the final holdout is near chance, the calibration artifact is not sufficiently selective, and live feed equivalence has not been verified. Overall accuracy materially above 60% should trigger a leakage audit.

No new holdout, multi-period or cost-sensitivity result is claimed. The original holdout is closed; future evaluation requires the extended dataset and a newly frozen period.

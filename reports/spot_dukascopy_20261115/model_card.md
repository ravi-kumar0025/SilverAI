# XAG/USD spot model card

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

The selected candidate was `H=12`, `k=0.2`, selected using walk-forward validation. Mean cross-market validation accuracy was 54.28%; untouched holdout accuracy was 49.33%, AUC 0.5186, and balanced accuracy 50.54%. Calibrated holdout accuracy was 48.11% with zero signals at confidence thresholds 0.55-0.70.

## Limitations

The candidate is not promoted. The dataset is historical, the final holdout is near chance, the calibration artifact is not sufficiently selective, and live feed equivalence has not been verified. Overall accuracy materially above 60% should trigger a leakage audit.

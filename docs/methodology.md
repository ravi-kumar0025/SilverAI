# Spot methodology

## Instrument and target

The active instrument is XAG/USD spot using Dukascopy bid OHLC in UTC. The raw contract is one-minute data; complete 5-minute candles are the default modeling resolution. The binary target is:

```text
forward_return = log(close[t + H] / close[t])
```

For a configured dead-zone multiplier `k`, a row is labelled `Up` when the forward return exceeds `k × ATR_pct`, `Down` when it is below `-k × ATR_pct`, and is removed when it falls inside the dead zone. Candidate horizons are `H = 3, 6, 12` bars (15, 30, and 60 minutes); candidate multipliers are `k = 0.2, 0.3, 0.5`.

The selected development artifact uses the configured 60-minute horizon (`H=12`) and binary LightGBM classifier. A separate regression artifact estimates the 1-hour forward return and converts it to an estimated price range; it is not a guarantee or trading recommendation.

## Features

The shared schema contains 34 ordered features in five groups:

- silver price and return structure;
- volatility;
- trend and momentum;
- XAU/USD and EUR/USD cross-market context;
- UTC session context.

The authoritative order is [`src/features/spot_features.py`](../src/features/spot_features.py). The source has no volume, so no synthetic volume features are created.

## Validation and leakage controls

Validation is chronological walk-forward. The configured split uses a three-bar purge and three-bar embargo; these are explicit bar counts and are not described as a full-horizon purge. The final holdout is reserved and marked `used_do_not_reuse` in configuration; the development artifact metadata marks it as not evaluated for development-only selection.

Controls include:

- closed-candle `known_at` timestamps;
- train-only fitting and calibration;
- no forward fill on XAG/USD;
- at most two-candle companion-market filling;
- missing/gapped future windows left unevaluated;
- validation-only selection and no reuse of the reserved holdout.

## Evidence and interpretation

Read the stored reports as research evidence, not proof of live performance. The current report package includes fold metrics, baselines, calibration, confidence-threshold analysis, data-quality checks, ablations, and an explicit readiness summary. Missing report assets and audit failures remain visible rather than being treated as passes.

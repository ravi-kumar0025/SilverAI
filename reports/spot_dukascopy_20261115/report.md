# Final XAG/USD Dukascopy spot study

> dataset_id: 4be8732dbab3aaa6 | instrument: XAGUSD_SPOT | provider: Dukascopy | price_side: bid | timezone: UTC | timeframe: 5m | feature_set: xagusd_spot_34_v1 | seed: 42

## Executive result

The final project uses binary Up/Down labels for XAG/USD spot. The selected candidate is `H=12` candles, or 60 minutes on 5-minute data, with `k=0.2`. It achieved 54.28% mean calibrated walk-forward accuracy with cross-market features, but only 49.33% accuracy on the untouched final holdout. The candidate is therefore not promoted as a reliable trading model.

This is a deliberate evidence-based conclusion. The earlier SI result was a three-class UP/NEUTRAL/DOWN experiment and is not comparable to this binary spot result.

## Data quality

| Instrument | 1-minute rows | Volume | Complete 5-minute rows | Alignment coverage |
|---|---:|---|---:|---:|
| XAG/USD | 1,334,781 | unavailable | 266,937 | 100.00% target index |
| XAU/USD | 1,334,790 | unavailable | 266,937 | 100.00% after <=2-bar fill |
| EUR/USD | 1,342,487 | unavailable | 266,937 | 100.00% after <=2-bar fill |

Files were concatenated, sorted, and de-duplicated in UTC. Five-minute candles are left-closed and left-labeled and are retained only when all five one-minute rows exist. Gold and EUR/USD are forward-filled by at most two candles; XAG/USD is never filled. Dukascopy supplies OHLC without volume, so no synthetic volume was used.

## Feature contract

The final model uses 34 causal features in five switchable groups: eight silver price/return features, four volatility features, seven trend/momentum features, nine cross-market features, and six UTC session features. A candle stamped `T` is usable only at `T+5 minutes`.

## Label and validation design

`fwd_ret = log(close[t+H] / close[t])`. A row is Up when `fwd_ret > k*ATR_pct`, Down when `fwd_ret < -k*ATR_pct`, and is dropped inside the dead zone. The study compares `H={3,6,12}` and `k={0.2,0.3,0.5}`. Walk-forward folds are chronological and include purge and embargo gaps; the final 20% holdout is touched once.

## Candidate results

| Candidate | Core mean accuracy | Core+cross mean accuracy |
|---|---:|---:|
| H=3, k=0.2 | 53.55% | 53.55% |
| H=3, k=0.3 | 53.49% | 53.15% |
| H=3, k=0.5 | 53.84% | 52.87% |
| H=6, k=0.2 | 54.16% | 53.13% |
| H=6, k=0.3 | 54.12% | 53.53% |
| H=6, k=0.5 | 54.19% | 52.68% |
| H=12, k=0.2 | 54.49% | **54.28%** |
| H=12, k=0.3 | 54.56% | 53.59% |
| H=12, k=0.5 | 54.43% | 53.38% |

## Label and leakage checks

For the selected cross-market candidate, the mean shuffled-label accuracy was 54.53% and the mean flip-test accuracy was 46.30%. Because the labels are mildly imbalanced, shuffled-label accuracy is not expected to be exactly 50%; it is a diagnostic, not evidence of skill. The flip test behaves as expected for a directional model.

The per-fold JSON contains always-Up, always-Down, majority-class, and RSI (`RSI<30 => Up`, `RSI>70 => Down`) baselines. It also records class balance and dropped rows for every horizon and threshold.

## Final holdout

| Metric | Raw | Calibrated |
|---|---:|---:|
| Accuracy | 49.33% | 48.11% |
| Balanced accuracy | 50.54% | 50.00% |
| AUC | 0.5186 | 0.5186 |
| Log loss | 0.6977 | 0.6964 |
| Brier score | 0.2523 | 0.2516 |

At confidence thresholds 0.55, 0.60, 0.65, and 0.70, the calibrated artifact emitted zero holdout signals. The dashboard consequently supports `No signal` and does not force a trade.

## Trader-facing output

The dashboard shows XAG/USD price action, session high/low, trend and volatility context, cross-market coverage, binary Up/Down probabilities, confidence, and a no-signal state. Predictions are persisted only for confirmed closed candles. Historical data is labelled historical; no live readiness or profitability claim is made.

## Reproduction

```powershell
python -m pip install -r requirements.txt
python tools/run_final_audit.py
pytest -q
python -m src.experiments.spot_dukascopy_study
```

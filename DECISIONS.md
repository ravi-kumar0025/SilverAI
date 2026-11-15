# Spot XAG/USD decisions

## 2026-11-15 - switch instrument

- Decision: use Dukascopy XAG/USD bid candles as the target instrument.
- Evidence: complete XAG/USD, XAU/USD and EUR/USD 1-minute histories are present from 2023-11-01 through 2026-11-13/15 and are aligned on UTC.
- SI models: rejected and isolated. The spot study writes artifacts under `reports/spot_dukascopy_20261115`; no file under `saved_models/` is loaded.

## 2026-11-15 - volume features

- Decision: exclude volume features.
- Evidence: Dukascopy exports contain timestamp and OHLC only; no volume column exists in the raw files. Creating synthetic volume would violate data integrity.

## 2026-11-15 - model promotion gate

- Decision: do not promote the first spot candidate to the app.
- Evidence: the best cross-market candidate was `H=12`, `k=0.2`, with 54.28% mean walk-forward validation accuracy but 49.33% untouched holdout accuracy. The calibrated artifact produced no signals at thresholds 0.55-0.70 because the calibration block was not sufficiently discriminative.
- Next experiment: improve calibration and selective signal gating on validation only, then evaluate one frozen candidate on the holdout.

## 2026-11-15 - final feature and horizon contract

- Decision: use the complete 34-feature spot schema in training, rolling inference, the API, and the dashboard.
- Decision: expose the selected `H=12` configuration as the final 60-minute horizon while retaining 15/30/60-minute candidate comparisons in the report.
- Evidence: the selected artifact stores the same ordered 34-feature list; the final audit and backend model endpoint validate that order.

## Live-source gate

The historical source is Dukascopy bid data. A live server is not yet enabled because a verified real-time XAG/USD, XAU/USD and EUR/USD source with the same bid-price semantics has not been connected. yfinance is not an acceptable live match for this requirement.

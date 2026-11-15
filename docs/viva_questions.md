# Viva preparation

## Why XAG/USD spot?

It matches the final Dukascopy data contract and removes futures-only concerns such as contract rolls, settlement, and maintenance breaks.

## Why binary labels?

The target asks whether the next horizon move is more likely up or down. Returns inside the ATR-scaled dead zone are dropped instead of being treated as a misleading directional class.

## Why 34 features?

They cover price action, volatility, momentum, cross-market information from gold and EUR/USD, and UTC session context. Each group can be enabled or disabled in `config.yaml`.

## How is leakage prevented?

Only closed candles are used, companion markets are forward-filled by at most two candles, labels use future prices only after the feature row, and walk-forward folds include purge and embargo gaps. The final holdout is touched once.

## What does the result mean?

The best mean walk-forward accuracy was 54.28% for the cross-market candidate, but the untouched holdout accuracy was 49.33%. That gap means the candidate is not promoted as a reliable trading model.

## Why is volume unavailable?

Dukascopy files supplied for this project contain timestamp and OHLC only. Synthetic volume would create information that was not present in the source.

## Is it live?

The dashboard and provider contract are prepared for live integration, but the checked-in data source is historical and the Dukascopy-compatible real-time source still requires verification. The application must be described as historical until that gate is complete.

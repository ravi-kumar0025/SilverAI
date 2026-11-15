# XAG/USD live-source decision

## Training source

Training uses Dukascopy bid OHLC data downloaded through `dukascopy-node`.
The historical files are UTC and do not contain volume, so the spot model does
not use volume features.

## Proposed live source

Use the same Dukascopy feed through `dukascopy-node`'s `getRealTimeRates` API
for `xagusd`, `xauusd`, and `eurusd`, with the same `bid` side and 5-minute
aggregation. This is the closest source match to the training data. The API is
documented as experimental and may return the still-forming candle, so the live
adapter must discard that candle and emit predictions only after the next
interval confirms closure.

## Current gate status

The live source is not verified yet. The project must measure latency, missing
candles, duplicate ticks, and historical/live candle agreement for one session
before the server is used for the graded five-day test. Until then the app must
show `NOT READY` and must not claim a live connection.

## Rejected source

`yfinance` is rejected as the live source because it is not the same Dukascopy
bid feed used for training and may be delayed or differently constructed.

# XAG/USD live-source decision

## Training source

Training uses Dukascopy bid OHLC data downloaded through `dukascopy-node`.
The historical files are UTC and do not contain volume, so the spot model does
not use volume features.

## Live source

The owner runs `scripts/live_feed_logger.mjs`, which calls
`getRealTimeRates` for `xagusd`, `xauusd`, and `eurusd` using bid OHLC and
appends the raw one-minute events to `data/live_log/*.jsonl`. `LiveLogSource`
discards forming candles, de-duplicates events, requires complete five-minute
candles, and feeds the same feature/inference path as replay tests.

## Current gate status

The bridge has been verified to ingest live candles with companions and to
generate persisted forecasts. Freshness is injectable and config-driven by
`config.yaml` (`live_stale_after_seconds: 180`); the UI reports `STALE` when
that limit is exceeded. A two-minute provider delay is accepted within this
window, but the system must still be treated as paper trading and research-only.
The live helper must be running for continuous updates.

When a new closed candle arrives, pending forecasts whose complete one-hour
future window is available are evaluated automatically. Missing or gapped
future candles remain `pending` rather than being guessed.

## Rejected source

`yfinance` is rejected as the live source because it is not the same Dukascopy
bid feed used for training and may be delayed or differently constructed.

# XAG/USD live-source decision

## Training source

Training uses Dukascopy bid OHLC for XAG/USD, XAU/USD, and EUR/USD. The active model consumes complete 5-minute candles and forecasts the next 15 minutes. Spot files contain no volume, so no volume feature is invented.

## Live source

The owner runs `scripts/live_feed_logger.mjs`, which calls `getRealTimeRates` for `xagusd`, `xauusd`, and `eurusd` and appends raw one-minute events to `data/live_log/*.jsonl`. `LiveLogSource` discards forming candles, de-duplicates events, requires complete five-minute candles, and feeds the same 34-feature path used in training.

When a closed 5-minute candle arrives, the active classifier emits calibrated Down/Neutral/Up probabilities and the active regressor emits the 15-minute expected return and 10/50/90% range. Outcomes are evaluated only after three complete future bars.

Freshness is controlled by `live_stale_after_seconds` and is shown as `STALE` when exceeded. Missing companions, gaps, and reconnect failures are reported rather than guessed. The system remains paper-only and research-only until continuous provider verification, reconnection evidence, and a full live test are complete.

`yfinance` remains rejected because it is not the same Dukascopy bid feed used for training.

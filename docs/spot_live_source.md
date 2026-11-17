# XAG/USD live-source decision

## Historical/training source

Training and reproducible historical replay use Dukascopy bid OHLC for XAG/USD, XAU/USD, and EUR/USD. Source timestamps are UTC, source rows are one-minute, and the files contain no volume. The active model therefore has no volume features.

## Live bridge

The owner runs [`scripts/live_feed_logger.mjs`](../scripts/live_feed_logger.mjs) with an installed provider module named by `SILVERAI_DUKASCOPY_MODULE` (default: `dukascopy-node`). The bridge requests bid, one-minute data for `xagusd`, `xauusd`, and `eurusd`, writes raw events to `data/live_log/*.jsonl`, and sends normalized closed one-minute candles to the backend.

The backend and `LiveLogSource`:

- discard forming candles;
- de-duplicate minute events;
- require complete five-minute bars;
- track gaps, outages, duplicate counts, and accepted candles;
- require companion symbols for new feature generation;
- reuse the same feature and inference path as historical/replay tests;
- persist finalized forecasts and settle eligible future windows.

## Freshness and verification gate

The configured stale window is `live_stale_after_seconds: 180` in `config.yaml`. The UI and API must expose a stale state after that limit; stale data is not converted into a signal. The live bridge polls according to `SILVERAI_POLL_MS` (default 20,000 ms) and uses bounded backoff after errors.

Before describing live mode as verified, check:

1. `GET /api/market/status` and `GET /api/live/status` report a genuine live provider.
2. XAG/USD, XAU/USD, and EUR/USD companions are available.
3. Last-update age is below 180 seconds.
4. Closed-candle and duplicate/gap counters are behaving as expected.
5. A confirmed prediction is persisted once per closed candle.
6. Future outcomes settle only when a complete window exists.
7. Reconnect and stale-feed behavior are observed in the current environment.

The bridge and paper-trading paths are research/integration infrastructure. They do not establish profitability, production readiness, or a completed five-day live evaluation.

## Rejected source

`yfinance` is not used for the live contract because it is not the same Dukascopy bid feed used for training and may be delayed or differently constructed.

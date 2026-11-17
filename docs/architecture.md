# Architecture

SilverAI keeps one causal data path for historical research, API inference, and live integration.

```text
Dukascopy 1m bid OHLC
        │
        ├── src/data/spot_data.py
        │      sort, de-duplicate, audit gaps, build complete candles
        │
        ├── src/features/spot_features.py
        │      34 ordered causal features across three instruments
        │
        ├── src/spot_targets.py
        │      ATR-scaled binary Up/Down labels
        │
        ├── src/experiments/spot_dukascopy_study.py
        │      chronological studies and versioned evidence
        │
        └── backend/services/prediction_service.py
               API inference, persistence, live scoring, paper state
```

## Data flow

1. `download_dukascopy.py` downloads and merges UTC 1-minute bid OHLC files for XAG/USD, XAU/USD, and EUR/USD.
2. `src.data.spot_data` sorts and de-duplicates source rows, reports gaps, and builds complete candles. A candle is usable only after its interval closes.
3. `src.features.spot_features` aligns the three instruments and generates the shared 34-feature schema. XAG/USD is never fabricated; companion-market filling is bounded to two candles.
4. `src.spot_targets` defines the forward log-return target and removes the no-move class using the configured ATR dead zone.
5. `src.experiments.spot_dukascopy_study` runs the configured chronological studies and writes report artifacts under `reports/`.
6. In live mode, `scripts/live_feed_logger.mjs` sends normalized closed one-minute events. `src.data.live_log_source` de-duplicates them, discards forming candles, builds complete 5-minute bars, and feeds the same feature path.
7. The FastAPI service exposes market data, model metadata, predictions, persistence, evaluation, diagnostics, and paper-trading state. The React frontend consumes those contracts and shows historical/live/fresh/stale states explicitly.
8. Finalized forecasts are persisted in SQLite. Eligible one-hour outcomes are settled automatically; missing or gapped future windows remain pending.

## Boundaries

All timestamps are UTC. The source has bid OHLC and no volume. The active model artifact is a research artifact and is not promoted as a production trading model. The live bridge is an opt-in integration path and does not itself establish live performance or profitability.

# Architecture

The spot pipeline has one causal path shared by research and live integration:

1. `download_dukascopy.py` downloads and merges UTC 1-minute bid OHLC files.
2. `src.data.spot_data` sorts, de-duplicates, reports gaps, builds complete candles, and aligns XAG/USD, XAU/USD, and EUR/USD.
3. `src.features.spot_features` generates the switchable 34-feature schema.
4. `src.spot_targets` creates ATR-scaled binary labels and drops the no-move class.
5. `src.experiments.spot_dukascopy_study` runs purged walk-forward LightGBM studies and writes the evidence report.
6. The FastAPI and React layers expose historical candles and trader-facing context while keeping live status explicit.

All timestamps are UTC. A candle is usable only at its `known_at` timestamp. No SI model or futures-specific session logic belongs in this version.

# Architecture

SilverAI uses one causal data path for training, replay, and live inference:

1. `download_dukascopy_market_data.py` downloads UTC 1-minute bid OHLC for XAG/USD, XAU/USD, and EUR/USD.
2. `src.data.market_candle_builder` sorts, de-duplicates, reports gaps, builds complete 5-minute candles, and aligns the instruments.
3. `src.features.market_features` generates the ordered 34-feature schema.
4. `scripts/train_spot_15m_multiclass.py` selects `k` and the neutral threshold chronologically, fits train-only calibration, and trains the 15-minute return regressors.
5. `reports/spot_15m_20261116_multiclass/pred_1` stores the base classifier; `pred_2` stores the calibrated classifier and nested mean/q10/q50/q90 regressors.
6. `src.data.live_log_market_source` reads closed-candle logs and feeds the same valid feature path.
7. FastAPI exposes market, model, prediction, persistence, evaluation, and paper-trading routes; React renders the terminal without chart signal-marker dots.
8. `src.prediction.store` persists immutable closed-candle forecasts and evaluates only complete three-bar future windows.

All timestamps are UTC. A candle is usable only after its `known_at` time. No volume is fabricated, and no retired binary/60-minute artifact is part of the active model path.

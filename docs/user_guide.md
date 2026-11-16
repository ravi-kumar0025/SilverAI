# User guide

Start the backend with Python 3.13, start `node scripts/live_feed_logger.mjs` for the live bridge, and start the frontend with `npm run dev` from `frontend/`.

The dashboard uses the active 15-minute model. Its primary output is Down, Neutral, or Up with calibrated probabilities. The regression panel reports the expected 15-minute return, predicted price, direction, and 10–90% price range.

Historical candles are labelled historical. Live candles display freshness and become stale after the configured window. Paper-trading endpoints are analytical only and do not place orders.

The chart intentionally contains candles and price context only; prediction-marker dots are disabled. Prediction history, model probabilities, regression outputs, and evaluation remain available on their dedicated pages.

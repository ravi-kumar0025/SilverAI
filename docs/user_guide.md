# User guide

The dashboard supports Historical and Live XAG/USD modes. Start the backend with `\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`, start `node scripts/live_feed_logger.mjs` for live data, then start the frontend with `npm run dev` from `frontend/`.

Treat `No signal` as a valid result: it means the confidence threshold, closed-candle rule, data freshness, or cross-market completeness gate was not satisfied. Historical candles are labelled historical; live candles show their freshness and become stale after the configured 180-second window. Paper-trading endpoints are analytical only and do not place orders.

Trained regression models predict the 1-hour return, converted to a price forecast with a 10-90% range; estimates only.

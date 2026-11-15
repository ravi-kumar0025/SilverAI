# User guide

The dashboard is a historical XAG/USD spot terminal. Start the backend with `python -m uvicorn backend.main:app --reload`, then start the frontend with `npm run dev` from `frontend/`.

Treat `No signal` as a valid result: it means the confidence threshold, closed-candle rule, data freshness, or cross-market completeness gate was not satisfied. Historical candles are not live prices. Paper-trading endpoints are analytical only and do not place orders.

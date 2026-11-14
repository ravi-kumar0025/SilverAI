# SilverAI user guide

## Application modes

SilverAI opens in historical mode without credentials. The Dashboard displays
the real `data/silver_5m.csv` OHLCV candles, with timeframe aggregation,
volume, crosshair, zoom, pan, and immutable persisted prediction markers.
Historical mode is not live trading.

The navigation provides Dashboard, Predictions, Analysis, Evaluation, Paper
Trading, Model Information, Data & Diagnostics, and Settings. Pages display
backend state and show unavailable messages when report artifacts or model
outputs are missing.

## Live configuration

Copy the root `.env.example` to the environment used by the backend. Historical
mode requires no key. Live mode requires an authorized provider and the
following variables: `SILVERAI_DATA_MODE`, `SILVERAI_LIVE_PROVIDER`,
`SILVERAI_LIVE_API_KEY`, `SILVERAI_LIVE_WS_URL`, and `SILVERAI_LIVE_SYMBOL`.
Do not commit secrets. A configured but unconnected provider is shown as
disconnected; it is never presented as live.

## Current operating mode

The default application uses the checked-in `silver_5m.csv` file. The dashboard labels this source as **historical**. It is not a live market feed and must not be used as evidence of live connectivity.

## Run the application

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend uses `VITE_API_BASE_URL=http://localhost:8000` by default.

## Reading the indicator

- `UP`: classifier direction is upward.
- `DOWN`: classifier direction is downward.
- `NEUTRAL`: the classifier predicts the neutral class.
- `MIXED`: classifier and regression evidence disagree.
- `INSUFFICIENT_DATA`: required evidence or reliability information is missing.

The displayed class probabilities are model outputs. They are not calibrated confidence and are not a guarantee of an outcome. Historical accuracy and probability output must remain separate.

## Live provider setup

No live provider is configured in the current repository. A provider adapter must supply authenticated COMEX SI candles, contract/roll metadata, timestamps, volume, session status, and reconnect behavior. The adapter must implement the `MarketDataProvider` contract in `src/data/market_data.py` and must report `is_live=True` only after real connection testing.

## Paper validation

Recorded, evaluated predictions can be inspected with:

```text
GET /api/predictions
GET /api/predictions?outcome_status=pending
GET /api/paper-trading
```

Paper trading is simulated only. Costs and slippage are explicit inputs to `evaluate_paper_trades`; no real orders are placed.

## When not to trust the indicator

Do not interpret a signal as reliable when the feed is delayed, historical-only, stale, disconnected, missing candles, outside the model's training distribution, or when the reliability payload reports limitations or model disagreement.

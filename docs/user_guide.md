# User guide

## Historical mode

Historical mode is the safe default and uses the checked-in Dukascopy XAG/USD data.

```powershell
# Terminal 1: from the repository root
$env:SILVERAI_DATA_MODE = "historical"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: from the repository root
Set-Location frontend
npm run dev
```

Open the Vite URL printed in Terminal 2. The dashboard should show historical candles and a historical data status. Historical data is real checked-in data, but it is not a live market feed.

## Live mode

Live mode requires an owner-supplied provider module and the Node bridge:

```powershell
# Terminal 1: backend
$env:SILVERAI_DATA_MODE = "live"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: live bridge
$env:SILVERAI_DUKASCOPY_MODULE = "dukascopy-node"
node scripts/live_feed_logger.mjs

# Terminal 3: frontend
Set-Location frontend
npm run dev
```

The bridge requests bid one-minute data for XAG/USD, XAU/USD, and EUR/USD. The backend only scores complete closed candles and marks the feed stale after the configured 180-second window. If live status cannot be verified, the frontend remains in Historical mode.

## Reading the dashboard

- `Historical` and `Live` identify the configured data mode.
- `Fresh`, `Stale`, and companion availability describe the current feed state.
- `No clear signal` is a valid result. It can mean that confidence is below threshold, a candle is not finalized, the feed is stale, or a companion market is unavailable.
- A forecast is persisted only at the confirmed closed-candle boundary.
- Missing or gapped future candles remain pending instead of receiving guessed outcomes.
- Paper trading is analytical only and does not place orders.

## Resetting test state

The Testing/Clear action and `POST /api/testing/clear` remove live-testing predictions, outcomes, signals, paper-trading history, and live-testing log files when live mode is active. They do not remove models, historical datasets, validation results, or threshold files.

## Troubleshooting

1. Check `GET http://127.0.0.1:8000/api/health`.
2. Check `GET http://127.0.0.1:8000/api/market/status` and `GET http://127.0.0.1:8000/api/live/status`.
3. Confirm `SILVERAI_API_URL`, `SILVERAI_CORS_ORIGINS`, and `SILVERAI_DATA_MODE` match the running backend.
4. In live mode, confirm the provider module is installed and all three companion symbols are arriving.
5. Review `GET /api/diagnostics` for non-fatal scoring or persistence errors.

For the model contract and endpoint payloads, see [`docs/api.md`](api.md). For the live verification gate, see [`docs/spot_live_source.md`](spot_live_source.md).

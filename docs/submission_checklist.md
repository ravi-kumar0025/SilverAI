# Final submission checklist

## Brief alignment

- [x] One declared asset: XAG/USD spot.
- [x] Option B web application: FastAPI backend plus React/Vite dashboard.
- [x] Dukascopy UTC bid data is retained with reproducible loading and resampling.
- [x] Signals use closed candles and expose a no-signal state.
- [x] The report includes model choices, validation, baselines, limitations, and failed approaches.
- [x] A trader guide and demonstration script are included.

## Evidence

- [x] 34 causal features are listed in the model card.
- [x] 5-minute, 15-minute, and 1-hour candle construction is supported.
- [x] XAG/USD, XAU/USD, and EUR/USD data quality is reported.
- [x] Walk-forward validation uses purge and embargo boundaries.
- [x] The final holdout is kept separate from candidate selection.
- [x] The final holdout result is reported honestly; no performance claim is made.

## Run before presentation

```powershell
python -m pip install -r requirements.txt
python tools/run_final_audit.py
pytest -q
python -m src.experiments.spot_dukascopy_study
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The checked-in feed is historical. Do not describe the demo as a verified live five-day test until the provider gate in `docs/spot_live_source.md` has been completed.

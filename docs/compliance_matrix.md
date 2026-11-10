# SilverAI compliance matrix

This matrix is based on the official `ps.pdf` brief (4 pages, dated 27 October 2026). SilverAI selects Option B: a Python-backed live web app for one declared asset, COMEX Silver Futures (SI).

| Requirement area | Status | Evidence / remaining gap |
| --- | --- | --- |
| Declared asset: COMEX Silver Futures | COMPLETE | `data/silver_5m.csv`, loader, API metadata, and frontend instrument identity use COMEX SI. |
| Historical training data | COMPLETE | `src/data/loader.py` validates timezone-aware OHLCV; `src/data/market_data.py` sorts, deduplicates, and validates candles. |
| Chosen timeframe and horizon | PARTIALLY COMPLETE | The team-selected contract is 5-minute candles and a 15-minute / 3-bar horizon. It is centralized and documented, but the PDF leaves this choice open and the research report should present the selection evidence explicitly. |
| Contract/vendor provenance | INCOMPLETE | CSV provenance and contract-roll policy are not encoded in the repository. Must be documented from the original vendor before live testing. |
| Chronological model validation | COMPLETE | `src/models/train.py` uses chronological train/validation/test partitions with a three-row purge. |
| Trained classifier and regressor | COMPLETE | Saved artifacts exist under `saved_models/`; metadata records feature order and horizon. |
| Baseline comparison | PARTIALLY COMPLETE | Majority, zero-return, always-UP, always-DOWN, and RSI baselines exist. A fresh consolidated report still needs to include all required comparisons. |
| Calibrated confidence | PARTIALLY COMPLETE | `src/models/calibration.py` now provides train-suffix chronological calibration, validation metrics, and optional artifact loading. The current checked-in model set has no calibrator, so the API continues to report calibration as unavailable. |
| Live market provider | MISSING | Provider-neutral interface and explicit historical provider exist; no authenticated live CME-compatible provider is configured or verified. |
| Live candle construction/reconnection | PARTIALLY COMPLETE | `ClosedCandleBuilder` and `ClosedCandleOrchestrator` now enforce closed-candle inference and restart deduplication; authenticated vendor transport and reconnect testing remain external prerequisites. |
| Chart with real historical candles | COMPLETE | `GET /api/market/candles` feeds the React chart with validated OHLCV data. |
| Chart signal overlay | PARTIALLY COMPLETE | The dashboard has the live-style chart and prediction monitor; persisted historical markers are stored but still need final visual overlay wiring. |
| No-repainting storage | INCOMPLETE | SQLite append-only prediction journal and immutability tests now exist; live closed-candle orchestration is not wired to a provider. |
| Prediction persistence | INCOMPLETE | `src/prediction/store.py` and `GET /api/predictions` exist; API observations without a candle close cannot yet support complete outcome evaluation. |
| Outcome evaluation | PARTIALLY COMPLETE | Store evaluation now requires the exact elapsed horizon and contiguous observed intervals; automatic provider scheduling remains pending. |
| Paper trading | INCOMPLETE | Deterministic cost/slippage-aware evaluator exists at `GET /api/paper-trading`; no live-validation feed is driving it yet. |
| Live performance dashboard | INCOMPLETE | Market status/history APIs exist; dedicated live-validation UI, confusion matrix, and metrics views remain. |
| Documentation and reproducibility | INCOMPLETE | Existing docs cover research workflow; live provider setup, frozen configuration, and user guide are being added. |
| Five-trading-day live readiness | NOT READY | The PDF requires five trading days for silver. A genuine live feed, frozen submission manifest, automatic evaluation, and end-to-end verification remain outstanding. |

## Current verdict

**NOT READY FOR THE PDF'S FIVE-TRADING-DAY LIVE TEST.** The repository now records the Option B and frozen-evaluation contract, but must not claim submission or live-test completion until a genuine live provider is configured, the final indicator is frozen, every closed-candle signal is persisted, outcomes are evaluated over the stated horizon, and baseline comparisons are reported.

# SilverAI compliance matrix

This matrix is based on the supplied official project brief pasted into the repository task. The referenced PDF file itself was not present in the workspace, so PDF-specific wording and page citations could not be independently checked.

| Requirement area | Status | Evidence / remaining gap |
| --- | --- | --- |
| Declared asset: COMEX Silver Futures | COMPLETE | `data/silver_5m.csv`, loader, API metadata, and frontend instrument identity use COMEX SI. |
| Historical five-minute data | COMPLETE | `src/data/loader.py` validates timezone-aware OHLCV; `src/data/market_data.py` sorts, deduplicates, and validates candles. |
| Contract/vendor provenance | INCOMPLETE | CSV provenance and contract-roll policy are not encoded in the repository. Must be documented from the original vendor before live testing. |
| Chronological model validation | COMPLETE | `src/models/train.py` uses chronological train/validation/test partitions with a three-row purge. |
| Trained classifier and regressor | COMPLETE | Saved artifacts exist under `saved_models/`; metadata records feature order and horizon. |
| Baseline comparison | INCOMPLETE | Majority and zero-return baselines exist; independent always-UP, always-DOWN, and RSI baseline implementation is now present but needs a fresh report run. |
| Calibrated confidence | MISSING | The current artifacts expose classifier probabilities but no calibration model or calibration report. |
| Live market provider | MISSING | Provider-neutral interface and explicit historical provider exist; no authenticated live CME-compatible provider is configured or verified. |
| Live candle construction/reconnection | MISSING | Requires a real vendor adapter, credentials, session rules, and reconnect testing. |
| Chart with real historical candles | COMPLETE | `GET /api/market/candles` feeds the React chart with validated OHLCV data. |
| Chart signal overlay | INCOMPLETE | Prediction monitor exists, but persistent historical signal markers are not yet rendered on candles. |
| No-repainting storage | INCOMPLETE | SQLite append-only prediction journal and immutability tests now exist; live closed-candle orchestration is not wired to a provider. |
| Prediction persistence | INCOMPLETE | `src/prediction/store.py` and `GET /api/predictions` exist; API observations without a candle close cannot yet support complete outcome evaluation. |
| Outcome evaluation | INCOMPLETE | Store evaluation handles completed horizons; a scheduler/provider loop is still required. |
| Paper trading | INCOMPLETE | Deterministic cost/slippage-aware evaluator exists at `GET /api/paper-trading`; no live-validation feed is driving it yet. |
| Live performance dashboard | INCOMPLETE | Market status/history APIs exist; dedicated live-validation UI, confusion matrix, and metrics views remain. |
| Documentation and reproducibility | INCOMPLETE | Existing docs cover research workflow; live provider setup, frozen configuration, and user guide are being added. |
| Five-day live readiness | NOT READY | Blocked by missing verified live provider credentials, calibration, live orchestration, and end-to-end signal/outcome validation. |

## Current verdict

**NOT READY FOR FIVE-DAY LIVE TESTING.** Historical replay and model research are available, but the project must not claim live readiness until a real COMEX SI-compatible provider is configured and tested, confidence calibration is completed, closed-candle inference is persisted, outcomes are evaluated after the full horizon, and the live-validation dashboard is verified.

# SilverAI completion status

This report separates implementation from verification. It does not claim live readiness or trading profitability.

## Implemented and verified

- Option B FastAPI/React architecture for COMEX Silver Futures (SI).
- Historical OHLCV loading, timezone validation, causal feature contract, chronological 70/15/15 split, and three-row purge.
- Three-class DOWN/NEUTRAL/UP modeling and baseline comparisons.
- Prediction orchestration, scenarios, final analytical signals, SQLite persistence, immutable forecast fields, outcome evaluation, and paper-only trading.
- Persisted final-signal chart markers restricted to explicitly finalized candles; marker history survives dashboard refresh.
- Leakage-safe train-suffix sigmoid and isotonic calibration candidates, validation-only method selection, multiclass Brier/log-loss/ECE metrics, and safe missing/corrupt-artifact handling.
- Historical mode remains available without credentials and is never labeled live.

## Implemented but not active in the default artifact set

- The training workflow can generate and persist a selected classification calibrator. The checked-in saved model metadata does not reference one, so the default application correctly reports calibration as unavailable.
- A separate calibration experiment was executed under `saved_models/experiments/calibration_20261114_v1`; it does not replace or activate the production artifacts.

## Blocked by external dependency

- Genuine live COMEX SI events require an approved vendor, account/entitlement, credentials, transport adapter, and contract/roll verification. The repository contains a credential-gated provider boundary, but no vendor transport is claimed as connected.
- The official five-trading-day live test cannot be completed without that feed and must not be substituted with historical replay.

## Not implemented

- A verified production vendor adapter with automatic network reconnect and real SI event ingestion.
- Verified live five-day performance, confidence intervals from repeated live periods, or profitability claims.

## Verification commands

```powershell
pytest -q
cd frontend
npm test
npm run build
```

## Separate calibration experiment

The selected method was sigmoid. On the validation partition, raw versus calibrated results were:

| Metric | Raw | Calibrated |
|---|---:|---:|
| Multiclass log loss | 1.04136985 | 1.02942791 |
| Mean Brier | 0.21185323 | 0.20884040 |
| ECE | 0.03638301 | 0.00484379 |

The experiment used 160,573 validated rows, a 112,398 / 24,083 / 24,086 chronological train/validation/test split, and did not use the final test partition. The classifier and calibrator hashes and per-class metrics are in the experiment `metrics.json` file.

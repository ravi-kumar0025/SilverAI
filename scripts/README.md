# Verification and live-feed scripts

These scripts support evaluator review and local operation. Run them from the repository root with the project virtual environment.

| Script | Purpose |
| --- | --- |
| `check_alignment.py` | Checks active XAG/USD identity, no-volume compliance, required artifacts, and manifest/report consistency. |
| `final_verify.py` | Runs the read-only final audit for dataset hashes, the 34-feature artifact contract, regression artifacts, report figures, and the live endpoint when available. |
| `build_dataset_manifest.py` | Computes hashes, row counts, date range, resampling metadata, and `dataset_id` for the canonical Dukascopy files. |
| `live_feed_logger.mjs` | Receives owner-configured live data and writes the live log consumed by the backend. Requires `SILVERAI_DUKASCOPY_MODULE`. |
| `replay_live.py` | Replays a controlled feed for integration checks; it is not a live-performance evaluation. |
| `build_project_report.py` | Builds report assets from the configured project outputs. |

Typical verification:

```powershell
.\.venv\Scripts\python.exe scripts/check_alignment.py
.\.venv\Scripts\python.exe scripts/final_verify.py
```

The scripts do not make a production-readiness claim. Read the generated audit and readiness summary for the timestamped result.

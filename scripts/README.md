# Submission scripts

- `build_dataset_manifest.py` computes the canonical Dukascopy file hashes, row counts, date range, resampling rule, and `dataset_id`.
- `verify_market_data_alignment.py` audits the active repository for spot identity, no-volume compliance, required artifacts, and manifest/report consistency.
- `train_spot_15m_multiclass.py` trains the active calibrated Down/Neutral/Up classifier and the 15-minute mean/q10/q50/q90 regressors. It writes `pred_1` and `pred_2` under `reports/spot_15m_20261116_multiclass/`.

# Reproducibility

From the repository root:

```powershell
\.venv\Scripts\python.exe -m pytest -q
\.venv\Scripts\python.exe scripts\check_alignment.py
\.venv\Scripts\python.exe scripts\final_verify.py
```

The input contract is `data/raw/dukascopy/{xagusd,xauusd,eurusd}_1m.csv` with millisecond UTC timestamps and bid OHLC. Derived candles are reproducible at 5-minute, 15-minute, and 1-hour resolution. Evidence is stored in `reports/spot_dukascopy_20261115/`, `reports/spot_v3_20261115/`, and `reports/spot_v3_20261116_regression/`. Do not rerun studies against the used holdout; the readiness summary records the current holdout policy and unavailable report sources.

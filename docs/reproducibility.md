# Reproducibility

From the repository root:

```powershell
python -m pip install -r requirements.txt
pytest -q
python -m src.experiments.spot_dukascopy_study
```

The input contract is `data/raw/dukascopy/{xagusd,xauusd,eurusd}_1m.csv` with millisecond UTC timestamps and bid OHLC. Derived candles are reproducible at 5-minute, 15-minute, and 1-hour resolution. The output directory is `reports/spot_dukascopy_20261115/`.

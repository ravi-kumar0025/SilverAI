# Reproducibility

The raw input contract is `data/raw/dukascopy/{xagusd,xauusd,eurusd}_1m.csv`, containing UTC bid OHLC. The current manifest is `data/dataset_manifest.json`; its dataset ID is `3b0693b8941795ee`.

Rebuild the manifest after changing canonical raw files:

```powershell
& "C:\Users\ravik\AppData\Local\Programs\Python\Python313\python.exe" scripts\build_dataset_manifest.py
```

Retrain the active 15-minute artifact:

```powershell
& "C:\Users\ravik\AppData\Local\Programs\Python\Python313\python.exe" scripts\train_spot_15m_multiclass.py
```

The output is `reports/spot_15m_20261116_multiclass/`. Its metadata records source rows, live-log extension rows, feature drops, split sizes, selected parameters, runtime versions, classifier metrics, calibration metrics, and regression metrics.

Retired `spot_v3` and `spot_dukascopy` reports remain historical research records. They are not loaded by the active API or displayed by the active terminal.

# Triple-barrier development audit

This is a development-only label audit. It excludes every candle at or after the permanently used original holdout start (`2026-10-01T00:00:00Z`). No holdout labels were opened or evaluated.

Command:

```powershell
python -c "import pandas as pd; from src.data.market_candle_builder import load_candles; from src.directional_labels import make_triple_barrier_labels; frame,_=load_candles('xagusd',5); dev=frame[pd.to_datetime(frame.timestamp,utc=True) < pd.Timestamp('2026-10-01',tz='UTC')].rename(columns={'open':'silver_open','high':'silver_high','low':'silver_low','close':'silver_close'}); labeled, report=make_triple_barrier_labels(dev, horizon_bars=12, profit_multiple=0.3); print({'development_rows':len(dev), **report, 'down_count':int((labeled.triple_barrier_label==0).sum()), 'up_count':int((labeled.triple_barrier_label==1).sum())})"
```

Observed output:

| Quantity | Value |
|---|---:|
| Development rows | 256,222 |
| Horizon | 12 bars / 60 minutes |
| Profit/stop multiple | 0.3 / 0.3 |
| Rows kept | 153,853 |
| Rows dropped | 102,369 |
| Missing future-window rows | 1,932 |
| Down labels | 77,853 |
| Up labels | 76,000 |
| Up fraction | 49.3978% |

The implementation uses the configured `timestamp`, `silver_open`, `silver_high`, `silver_low`, and `silver_close` mapping. It requires every exact timestamp in the horizon window, so missing candles and weekend gaps do not silently become valid time-barrier samples.

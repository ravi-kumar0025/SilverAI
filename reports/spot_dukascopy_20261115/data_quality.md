# Data quality report

| Instrument | Raw 1-minute rows | Range | Volume | Complete 5-minute rows |
|---|---:|---|---|---:|
| XAG/USD | 1,334,781 | 2023-11-01 to 2026-11-13 | unavailable | 266,937 |
| XAU/USD | 1,334,790 | 2023-11-01 to 2026-11-13 | unavailable | 266,937 |
| EUR/USD | 1,342,487 | 2023-11-01 to 2026-11-13 | unavailable | 266,937 |

Files were concatenated, sorted in UTC, and de-duplicated by timestamp. Candles use left-closed, left-labeled resampling and are retained only when all five 1-minute rows exist. Cross-market coverage after alignment was 100% of the 266,937-candle XAG/USD target index with a maximum two-candle forward fill for XAU/USD and EUR/USD. Silver was never forward-filled.

The largest raw gaps are session/data-availability gaps. They are reported by the loader and are not silently filled. No synthetic volume was created.

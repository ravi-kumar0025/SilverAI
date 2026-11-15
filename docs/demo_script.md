# Final demonstration script

1. Show `config.yaml` and identify the asset as XAG/USD spot, the Dukascopy source, UTC timestamps, and the 5-minute candle resolution.
2. Run `python tools/run_final_audit.py` and show the three input row counts and 34-feature schema.
3. Start the FastAPI server and open the React dashboard.
4. Point out the historical data badge, latest XAG/USD price, session high/low, chart, and closed-candle timestamp.
5. Explain the trader context panel: trend, bias, volatility regime, cross-market coverage, and why volume is unavailable rather than fabricated.
6. Explain that the indicator can return `No signal` when confidence, freshness, or cross-market completeness is insufficient.
7. Open the study report and show the candidate table, untouched holdout result, calibration threshold table, and leakage warning.
8. State clearly that the report does not claim profitability or live readiness; paper trading is analytical only.

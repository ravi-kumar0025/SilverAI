# SilverAI timeframe and horizon experiment

This report is generated from the checked-in `data/silver_5m.csv` artifact using
`src/experiments/timeframe_horizon.py`. It is a non-live coverage and
timestamp-validity experiment; it does not use the official live test.

## Policy

The target generator requires the exact future timestamp and every expected
interval between the anchor and exit. Rows crossing a market-session break,
missing candle, duplicate, or irregular interval are unresolved and are not
silently treated as a valid next-row forecast.

The PDF leaves timeframe and horizon to the team. The existing model artifacts
use five-minute candles and a 15-minute / three-bar horizon. The experiment
therefore evaluates nearby alternatives without changing the frozen artifacts.

| Candle | Horizon | Candles | Valid targets | Coverage | Class balance (DOWN / NEUTRAL / UP) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 5m | 10m | 189,798 | 167,653 | 88.33% | 29.25% / 40.00% / 30.75% |
| 5m | 15m | 189,798 | 160,612 | 84.62% | 29.08% / 40.00% / 30.91% |
| 5m | 30m | 189,798 | 145,674 | 76.75% | 28.91% / 40.00% / 31.09% |
| 10m | 20m | 101,061 | 94,072 | 93.08% | 29.09% / 40.00% / 30.91% |
| 10m | 30m | 101,061 | 91,646 | 90.68% | 29.02% / 40.00% / 30.98% |
| 15m | 15m | 68,820 | 66,704 | 96.93% | 29.10% / 40.00% / 30.90% |
| 15m | 30m | 68,820 | 65,008 | 94.46% | 28.96% / 40.00% / 31.04% |
| 30m | 30m | 35,100 | 34,072 | 97.07% | 29.02% / 40.00% / 30.98% |

## Interpretation and limitation

Longer resampled intervals have higher exact-target coverage, while the
five-minute/15-minute setting retains substantially more observations and is
the only configuration with checked-in features, targets, saved models, and
reports already aligned. This table does not establish that it is the most
predictive configuration: a full candidate comparison must rebuild the
feature/target datasets and run the same chronological model-selection process
for each candidate. Until that experiment is completed, the five-minute and
15-minute values remain a documented research configuration, not a proven
optimal choice.

The generated machine-readable outputs are
`reports/timeframe_horizon_experiment.csv` and
`reports/timeframe_horizon_experiment.json`.

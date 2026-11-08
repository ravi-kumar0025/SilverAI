# SilverAI methodology

## Data preparation

The input workflow uses five-minute silver OHLCV candles from `data/silver_5m.csv` and the derived `data/silver_features.csv`. Intraday timestamps are timezone-aware Eastern Time. Loading and preparation validate schema, preserve rows, sort chronologically, and report duplicate timestamps, missing values, and intervals longer than the modal positive interval. Long intervals are reported, not automatically treated as errors, because futures sessions contain closures and breaks.

OHLCV checks require numeric positive prices, high values at least as large as open and close, low values no larger than open and close, and nonnegative volume when volume exists. Missing values are reported rather than filled.

## Feature engineering

The approved feature set has six groups:

| Group | Features |
| --- | --- |
| Momentum | `return_1`, `return_3`, `return_6` |
| Volatility | `volatility_5`, `volatility_20`, `volatility_50` |
| Trend | `sma_distance_20`, `sma_distance_50` |
| Candle | `candle_return`, `candle_range_pct`, `upper_wick_ratio`, `lower_wick_ratio` |
| Volume | `relative_volume`, `volume_change` |
| Time | `hour_sin`, `hour_cos`, `day_of_week` |

Returns and candle percentages are expressed in percent; moving-average distances and wick ratios use their implementation-defined ratio forms. Time features use the Eastern-localized timestamp. Rolling features are causal: no future rows are used. The first warm-up rows remain missing and are removed only when training data preparation requires finite features and targets.

## Target construction

The target notebook requires the current bar and the next three consecutive five-minute timestamps. Rows near session gaps and the final three rows are excluded from target creation. For valid rows:

```text
future_return_15m = 100 * (close[t+3] - close[t]) / close[t]
```

The notebook's `target_direction_15m` uses a provisional `±0.05%` band for diagnostics. Training does not trust that provisional label: `src.models.train` calibrates a neutral threshold from the absolute continuous returns in the purged training partition only. The saved training threshold is `0.0776397515528126%`.

## Splitting and fitting

Rows are sorted chronologically and divided by raw row boundaries into nominal 70% training, 15% validation, and 15% test partitions. Three rows are purged from the end of the training and validation partitions. The resulting saved partition sizes are 112,398, 24,083, and 24,086 rows.

Candidate models are fitted on training data and scored on validation data. Logistic and linear regression pipelines use training-fitted median imputation; standardized logistic regression also fits its scaler on training data. Tree models use the finite prepared feature matrix. No test result is used for model selection.

## Models and selection

The classifier candidates are logistic regression, random forest, and histogram gradient boosting. The regressor candidates are linear regression, random forest, and histogram gradient boosting. Classification selection ranks validation macro F1. Regression selection ranks validation RMSE. The saved selected models are histogram gradient boosting classification and linear regression.

## Evaluation

Evaluation uses fixed class order `DOWN (-1)`, `NEUTRAL (0)`, `UP (1)`, including absent classes in confusion matrices and reports. It calculates accuracy, balanced accuracy, macro F1, per-class metrics, MAE, RMSE, and safe R². Baselines are a majority class learned from training labels and a zero-return regression prediction.

The latest executed test evaluation returned:

| Model | Accuracy | Balanced accuracy | Macro F1 | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected classifier | 0.38375 | 0.38565 | 0.37312 | — | — | — |
| Majority classifier baseline | 0.27003 | 0.33333 | 0.14175 | — | — | — |
| Selected regressor | — | — | — | 0.22040 | 0.31952 | -0.00084 |
| Zero-return baseline | — | — | — | 0.22018 | 0.31940 | -0.00010 |

The regression result is effectively baseline-level and should not be presented as useful forecasting performance.

## Prediction interpretation

Classifier probabilities are accepted only when all three class probabilities are finite, in range, and sum to one. Regression direction uses the saved neutral threshold when supplied. Scenarios are descriptive, not additional models. If classifier and regression directions disagree, the final signal is `MIXED`; if inputs or historical reliability are insufficient, it is `INSUFFICIENT_DATA`.

The project currently has no calibrated confidence module. `pipeline.py` therefore reports this limitation rather than fabricating reliability.

## Research justification

The existing [`reports/feature_research_analysis.md`](../reports/feature_research_analysis.md) retains four relevant papers:

1. Ji et al. (2022), adaptive technical-indicator feature selection.
2. Cohen (2022), lagged dependence and algorithmic strategies for precious-metal futures.
3. Gono, Napitupulu, and Firdaniza (2023), silver forecasting with XGBoost.
4. Varshini, Kayal, and Maiti (2024), model and regime comparisons for metal futures.

These papers justify testing compact interpretable features, multiple horizons, chronological evaluation, and model/regime robustness. They do not validate SilverAI's exact 5-minute features or establish novelty, profitability, or transferable predictive performance.

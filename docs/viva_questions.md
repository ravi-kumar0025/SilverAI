# SilverAI viva questions and answers

## Why COMEX silver futures?

Silver is the project's chosen market because the available artifacts contain intraday COMEX SI data and the research report includes precious-metal forecasting literature. The project does not claim that silver is easier to predict than another market.

## Why five-minute candles?

Five-minute candles provide a practical intraday sampling interval for a short-horizon experiment while reducing the noise and volume of raw one-minute observations. The choice is a project design decision, not a claim that five minutes is optimal.

## Why a 15-minute horizon?

The target looks three consecutive five-minute candles ahead. Three bars give a clearly defined short horizon while retaining enough time for a return to exceed tiny price-tick noise. Rows without three consecutive future bars are excluded.

## Why these 17 features?

They form a compact, interpretable set from six groups: multi-horizon momentum, rolling volatility, trend distance, candle geometry, volume context, and Eastern-localized time. The research report treats them as candidates supported by domain reasoning and related literature, not as proven predictors.

## How was leakage prevented?

Feature calculations use current and prior bars. Target creation uses future bars only for the target columns, and those columns are excluded from the model feature list. Threshold calibration occurs on the purged training partition. The test partition is never used for model fitting or selection.

## Why chronological splitting?

Random shuffling would let observations from later market regimes influence training for earlier observations and would undermine a forward-looking evaluation. The project sorts by Eastern timestamp and uses nominal 70% / 15% / 15% chronological partitions.

## Why purge rows?

Targets use a three-bar future horizon. Purging three rows at the train and validation boundaries reduces boundary contamination from labels whose future window approaches the next partition.

## Why these algorithms?

Logistic/linear regression provide simple interpretable baselines. Random forests test bagged nonlinear trees. Histogram gradient boosting tests a more expressive boosting model. The project compares families rather than assuming that one algorithm is universally best.

## How were models selected?

The classifier with the highest validation macro F1 and the regressor with the lowest validation RMSE were selected. The current selected artifacts are `hist_gradient_boosting_classifier` and `linear_regression`. Test metrics were generated after selection and did not choose the models.

## Why macro F1 instead of accuracy?

The three direction classes are imbalanced. Macro F1 gives each class equal weight and reveals whether performance is concentrated in one class. Accuracy and balanced accuracy are also reported.

## How should probabilities be interpreted?

They are the classifier's class probabilities for the current feature row, mapped explicitly to `DOWN`, `NEUTRAL`, and `UP`. The pipeline accepts them only when all three values are finite, within [0, 1], and sum to one. They are not historical reliability or a probability of profit.

## How are scenarios generated?

`scenarios.py` creates deterministic Bullish, Bearish, and Neutral descriptions from the classifier direction, valid class probabilities, and regression-implied direction. It reports supporting evidence, contradicting evidence, forecast information, and limitations. It does not invent price targets or market evidence.

## What happens when models disagree?

The disagreement remains visible in scenario limitations and produces the `MIXED` signal. For example, an UP classifier with a negative regression forecast does not become automatically bullish or bearish.

## What does reliability mean in the current system?

Reliability currently means whether probability data, model agreement, and supplied limitations are available. A calibrated historical confidence module is not implemented, so the pipeline explicitly reports limited reliability instead of fabricating a score.

## Why is the regression result weak?

The selected regressor's untouched-test RMSE is 0.31952 versus 0.31940 for the zero-return baseline, and R² is -0.00084. The correct conclusion is that this implementation does not yet demonstrate useful continuous-return forecasting on the available test partition.

## Why does the dashboard show no live price?

The current backend exposes model predictions and metadata, not a live market-price feed. The frontend displays an unavailable state rather than presenting a stale or fabricated price.

## Why is historical analysis empty?

The backend stores only the latest prediction in process memory. It does not persist a prediction history or actual future outcomes, so the dashboard cannot honestly render a forecast-versus-actual series yet.

## What are the main design trade-offs?

- A compact feature set improves interpretability but may omit useful information.
- Chronological evaluation is realistic but gives fewer independent validation regimes than random cross-validation.
- In-memory latest state keeps the prototype simple but loses history on restart.
- Lazy cached model loading avoids repeated reloads but requires valid artifacts at runtime.
- A separate classifier and regressor expose disagreement, but the combined signal is intentionally conservative.

## What would improve the project next?

Add a reusable tested target-creation module, calibrate confidence using validation-only data, evaluate rolling regimes, persist predictions and outcomes, add live data with clear provenance, and assess transaction costs and slippage in a separate trading study. None of those improvements should be presented as completed today.

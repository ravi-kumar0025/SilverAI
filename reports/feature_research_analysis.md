# SilverAI Feature Research Analysis

**Project:** SilverAI - COMEX silver futures machine learning  
**Asset:** COMEX Silver Futures (SI)  
**Primary timeframe:** 5-minute candles  
**Inputs:** `data/silver_5m.csv`, `data/silver_features.csv`  
**Feature-engineering source:** `notebooks/05_feature_engineering.ipynb`  
**Status:** Literature-based justification; no model selection has been performed

## 1. Project Objective

SilverAI aims to predict short-term movement in COMEX silver futures from information available at the close of each 5-minute candle. The current feature set is deliberately compact: it represents multi-horizon momentum, volatility, trend distance, candle geometry, volume context, and intraday calendar effects without adding a large library of overlapping indicators.

This report evaluates that candidate set against a small group of verifiable studies. Published findings are treated as evidence about methods or related markets, not as proof that a feature predicts SilverAI's target. Final inclusion or removal will require chronological out-of-sample testing on silver futures.

## 2. Initial Feature Engineering Strategy

SilverAI currently has 17 candidate features in six groups. Their existing definitions are preserved below.

| Group | Exact features | Information represented |
|---|---|---|
| Momentum | `return_1`, `return_3`, `return_6` | Percentage price change over 1, 3, and 6 bars. |
| Volatility | `volatility_5`, `volatility_20`, `volatility_50` | Rolling close-to-close volatility over short, medium, and longer windows. |
| Trend | `sma_distance_20`, `sma_distance_50` | Percentage distance of close from the 20- and 50-bar simple moving averages. |
| Candle | `candle_return`, `candle_range_pct`, `upper_wick_ratio`, `lower_wick_ratio` | Direction, total intrabar range, and normalized rejection/buying-pressure geometry. |
| Volume | `relative_volume`, `volume_change` | Current volume relative to its reference level and its bar-over-bar change. |
| Time | `hour_sin`, `hour_cos`, `day_of_week` | Cyclical time-of-day and weekly-session information. |

The strategy is to begin with economically interpretable, scale-aware OHLCV features, then test whether each group contributes incremental predictive information. No feature is considered validated merely because it is common in technical analysis.

## 3. Literature Review

The review is limited to four relevant papers. The first is the mandatory methodological reference; the other three concern precious metals or metal futures directly.

### 3.1 Ji et al. (2022): adaptive feature selection

**Full citation:** Gang Ji, Jingmin Yu, Kai Hu, Jie Xie, and Xunsheng Ji, “An Adaptive Feature Selection Schema Using Improved Technical Indicators for Predicting Stock Price Movements,” *Expert Systems with Applications*, 200, Article 116941 (2022). [DOI](https://doi.org/10.1016/j.eswa.2022.116941)

**Objective and methodology.** Ji et al. study direction classification for four **stock indices** - SSEC, HSI, DJI, and S&P 500 - using Random Forest. They begin with 18 technical indicators, apply wavelet denoising to the price series, recalculate improved indicators, and compare them with indicators calculated from the original series. Their two-stage adaptive selector uses repeated permutation-importance rankings and evaluates candidate subset sizes rather than assuming that the full indicator set or one fixed top-k value is optimal. The study also varies the historical time window.

The paper's 18-indicator panel, using its paper/library naming conventions, is: AD, ADOSC, AROON, ATR, BBANDS, CCI, CMO, DX, EMA, FASTK, MACD, MFI, MOM, OBV, ROC, RSI, ULTOSC/ULTISC, and WILLR/WR. The repeatedly retained indicators highlighted by the paper are **WR, FASTK, CCI, ULTISC, and ROC**.

**Findings relevant to SilverAI.** The paper reports that denoising can improve the quality of derived technical indicators, that selected subsets can outperform the full set, and that appropriately increasing the historical window can help. The authors prefer permutation importance to Gini importance because permutation importance measures the performance loss after disrupting a feature, whereas Gini importance measures tree impurity reduction and can be distorted by feature properties and correlated predictors.

**Relation and limitation.** This is methodological guidance, not silver-futures evidence. Its daily stock-index data, target, sampling frequency, and market microstructure differ from SilverAI's 5-minute COMEX futures task. Permutation importance is also not universally reliable: correlated features can substitute for one another, rankings can vary by fold and random seed, and repeated permutations are computationally costly. SilverAI should test selection stability inside chronological folds rather than copy the five retained indicators.

### 3.2 Cohen (2022): precious-metal futures and lagged dependence

**Full citation:** Gil Cohen, “Algorithmic Strategies for Precious Metals Price Forecasting,” *Mathematics*, 10(7), 1134 (2022). [DOI](https://doi.org/10.3390/math10071134)

**Objective and methodology.** Cohen evaluates algorithmic forecasting/trading approaches on 20 years of daily futures data for gold, silver, copper, platinum, and palladium. The study uses lagged multi-dimensional regression, Darvas boxes, and Bollinger-band rules, and examines autocorrelation and lagged dependence among the precious metals.

**Findings relevant to SilverAI.** The paper reports short-term dependence in precious-metal returns and cross-metal relationships, and finds that silver was comparatively forecastable in its daily setting. The lag structure supports testing multiple horizons rather than assuming that one return window contains all useful information. Its range and band-based methods also motivate treating volatility and candle range as distinct from direction.

**Relation and limitation.** This is direct precious-metal futures evidence, but it uses daily data and includes other metals. SilverAI currently uses only SI 5-minute OHLCV and does not claim that gold, copper, or platinum variables are available. The study does not validate SilverAI's specific 1-, 3-, and 6-bar features, so those remain hypotheses to test.

### 3.3 Gono, Napitupulu, and Firdaniza (2023): silver price forecasting with XGBoost

**Full citation:** Dylan Norbert Gono, Herlina Napitupulu, and Firdaniza, “Silver Price Forecasting Using Extreme Gradient Boosting (XGBoost) Method,” *Mathematics*, 11(18), 3813 (2023). [DOI](https://doi.org/10.3390/math11183813)

**Objective and methodology.** This open-access study forecasts daily silver prices with XGBoost and grid-search hyperparameter tuning. It uses daily silver prices together with gold, platinum, and dollar/euro exchange-rate data, compares the tuned model with Random Forest and CatBoost, and evaluates MAPE, RMSE, MAE, and SI.

**Findings relevant to SilverAI.** The study demonstrates that tree ensembles and careful hyperparameter tuning are viable for silver forecasting, and that related precious-metal and exchange-rate variables may contain information at a daily horizon. It supports using a tuned baseline model and evaluating more than one error or directional metric.

**Relation and limitation.** The target is daily price forecasting, not 5-minute direction or return prediction. Its cross-asset inputs are not part of SilverAI's current data scope, and its reported metrics cannot be transferred to this project. It supports the validation workflow and the importance of clear baselines, not any specific SilverAI feature.

### 3.4 Varshini, Kayal, and Maiti (2024): model and regime robustness for metal futures

**Full citation:** Anu Varshini, Parthajit Kayal, and Moinak Maiti, “How Good Are Different Machine and Deep Learning Models in Forecasting the Future Price of Metals? Full Sample versus Sub-Sample,” *Resources Policy*, 92, Article 105040 (2024). [DOI](https://doi.org/10.1016/j.resourpol.2024.105040)

**Objective and methodology.** The study forecasts gold, silver, copper, platinum, palladium, and aluminium futures using Stacked LSTM, ConvLSTM, BiLSTM, SVR, XGBoost, and GRU. It compares a roughly 20-year full sample with two ten-year subsamples and examines different input periods, including 30- and 60-day inputs.

**Findings relevant to SilverAI.** Model performance varies materially by metal, sample period, and input choice; no single model is uniformly best. This directly supports regime-aware evaluation and testing more than one historical context rather than selecting features from one aggregate score.

**Relation and limitation.** The study is daily and model-comparative, not a feature-ablation study of 5-minute COMEX data. Its input-window findings cannot determine SilverAI's correct bar lookback. It supports robustness checks and chronological subperiod analysis, not a claim that any one of SilverAI's 17 features is superior.

## 4. Feature-wise Justification

| Group | Why it may matter for 5-minute silver | What the selected papers support | SilverAI interpretation |
|---|---|---|---|
| Momentum | Captures immediate continuation or reversal across several intraday horizons. | Cohen supports examining lagged return dependence in precious metals; the other papers support testing input history but do not validate these exact horizons. | Keep as a candidate group; test whether 1, 3, and 6 bars add complementary information. |
| Volatility | Distinguishes quiet conditions from active price discovery and scales the meaning of a move. | Cohen uses range/band information; the metal-futures studies show performance varies with period and regime. | Keep all three initially; test whether `volatility_50` adds value beyond shorter windows. |
| Trend | Measures whether price is stretched relative to recent local baselines. | Ji's indicator panel includes moving-average and band concepts, but its stock-index results do not prove SMA distance for SI. | Retain as a compact trend representation; test redundancy between 20 and 50 bars. |
| Candle | Encodes intrabar direction, range, and rejection that close-only returns omit. | The selected papers do not isolate these exact wick ratios, so support is methodological rather than direct. | Treat as domain-motivated candidates; test group and feature-level ablations. |
| Volume | Separates high-participation moves from low-participation noise. | Ji includes volume/accumulation indicators; its stock-index findings motivate testing volume information, not these exact formulas. | Compare `relative_volume` with `volume_change`; do not assume both are needed. |
| Time | Represents session liquidity and recurring weekly behavior. | The selected papers do not establish these exact encodings for COMEX SI. | Keep as a market-structure hypothesis and test against a no-time-feature baseline. |

## 5. Feature Evidence Matrix

| Group | Captures | Potential silver relevance | Evidence strength | Limitations / redundancy | Experimental test |
|---|---|---|---|---|---|
| Momentum | Multi-horizon returns | Trend continuation or mean reversion after recent bars | Moderate, via precious-metal lag dependence | Horizons may be correlated and target-horizon dependent | Group ablation plus individual permutation importance for 1/3/6 bars |
| Volatility | Recent dispersion | Regime and risk intensity | Moderate, via range/band and regime-window studies | Close-to-close measures may overlap; long window may lag | Compare 5+20, 5+20+50, and single-window baselines |
| Trend | Distance from moving averages | Local stretch and directional context | Indirect, via Ji's indicator methodology | Overlaps with momentum and volatility | Correlation matrix and nested ablations |
| Candle | Intrabar shape | Rejection, pressure, and bar-level range | Hypothesis only in this selected set | Ratios can be unstable when ranges are tiny | Test candle group and each ratio with minimum-range safeguards |
| Volume | Participation and acceleration | Conviction versus noise | Indirect, via Ji's volume indicators | `relative_volume` and `volume_change` may duplicate activity shocks | Compare each alone and together using OOS permutation importance |
| Time | Session/weekly seasonality | Liquidity, maintenance, and settlement patterns | Hypothesis for this project | Calendar effects can decay or be regime-specific | Compare full model with time features removed across periods |

## 6. Feature Redundancy and Limitations

Several candidate features describe related information. `return_1`, `return_3`, and `return_6` may share momentum; the SMA distances may overlap with returns; volatility and candle range both respond to activity; and the two volume features may respond to the same participation shock. Correlation is useful for detecting obvious redundancy but cannot establish predictive redundancy in a nonlinear model.

The main limitations are market and sampling mismatch in the literature, non-stationary relationships, multiple-testing risk, and the possibility that a feature appears important only because a correlated substitute is absent. The report therefore makes no claim that any feature is experimentally proven. It also does not add Ji et al.'s 18 indicators to SilverAI and does not alter the existing definitions.

## 7. SilverAI Feature Selection Methodology

SilverAI should move from candidates to final features using the following sequence:

1. **Domain review:** retain the 17 candidates because each has an interpretable market rationale and is computable from the permitted OHLCV/time inputs.
2. **Data-quality audit:** check missing values, infinities, warm-up periods, timestamp order, and feature availability at the prediction timestamp.
3. **Correlation and redundancy analysis:** inspect pairwise and grouped correlations on training data only. Use this to form hypotheses, not to remove features automatically.
4. **Baseline model:** train a fixed, documented baseline on the full 17-feature set using chronological splits.
5. **Permutation importance:** compute repeated importance on validation or test folds, not on training predictions. Record both the magnitude and selection frequency across folds.
6. **Ablation testing:** remove one feature or one group at a time and compare out-of-sample performance with the baseline.
7. **Chronological validation:** repeat the analysis across rolling or expanding time splits and relevant market subperiods.
8. **Final selection:** keep a feature only when its contribution is stable, economically plausible, and improves out-of-sample performance under the predeclared metric and validation protocol.

Feature importance alone is insufficient because importance is conditional on the model, metric, split, feature correlation structure, and random seed. A high score can be unstable, a low score can reflect information shared with a substitute, and a feature can be useful only in one regime. Ji et al. specifically motivate repeated permutation-based selection, but also show why selected subsets should not be treated as universally optimal.

## 8. Experimental Validation Plan

No models are trained and no features are selected in this report. The next analysis should:

- define the prediction horizon and target before feature selection;
- use chronological train/validation/test partitions with no shuffle leakage;
- fit preprocessing and any feature-selection step inside each training fold;
- compare the 17-feature baseline with one-feature, grouped, and leave-one-group-out ablations;
- report directional and error metrics with confidence intervals or fold dispersion;
- inspect permutation-importance stability across folds and market subperiods;
- test whether `volatility_50`, `volume_change`, or any other candidate adds incremental value;
- separate statistical improvement from practical usefulness by considering turnover, slippage, and transaction costs if a trading rule is later evaluated.

The relevant hypothesis from Ji et al. is that improved feature quality and adaptive selection may help. For SilverAI, wavelet denoising and any additional indicators should be separate experiments, fitted causally within the training window. The stock-index results must not be presented as evidence of silver-futures predictability.

## 9. Conclusion

SilverAI's 17 features form a coherent, compact starting set for 5-minute COMEX silver futures. The literature supports testing multi-horizon information, volatility context, model baselines, and regime-aware validation, but it does not justify declaring any current feature optimal. Ji et al. (2022) adds a useful adaptive-selection framework and emphasizes permutation importance, while its stock-index setting limits direct transfer. The final feature set should be determined only after chronological out-of-sample permutation importance and ablation testing demonstrate stable incremental value.

## 10. References

1. Ji, G., Yu, J., Hu, K., Xie, J., & Ji, X. (2022). *An Adaptive Feature Selection Schema Using Improved Technical Indicators for Predicting Stock Price Movements*. **Expert Systems with Applications, 200**, 116941. [https://doi.org/10.1016/j.eswa.2022.116941](https://doi.org/10.1016/j.eswa.2022.116941)
2. Cohen, G. (2022). *Algorithmic Strategies for Precious Metals Price Forecasting*. **Mathematics, 10**(7), 1134. [https://doi.org/10.3390/math10071134](https://doi.org/10.3390/math10071134)
3. Gono, D. N., Napitupulu, H., & Firdaniza. (2023). *Silver Price Forecasting Using Extreme Gradient Boosting (XGBoost) Method*. **Mathematics, 11**(18), 3813. [https://doi.org/10.3390/math11183813](https://doi.org/10.3390/math11183813)
4. Varshini, A., Kayal, P., & Maiti, M. (2024). *How Good Are Different Machine and Deep Learning Models in Forecasting the Future Price of Metals? Full Sample versus Sub-Sample*. **Resources Policy, 92**, 105040. [https://doi.org/10.1016/j.resourpol.2024.105040](https://doi.org/10.1016/j.resourpol.2024.105040)

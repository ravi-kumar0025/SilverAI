# SilverAI Error Analysis and Market-Regime Robustness

## Scope and evidence status

This is a descriptive analysis of the persisted, frozen historical test predictions in `reports/final_holdout_20261114_v2/predictions.csv`. No model, calibrator, threshold, feature set, or production artifact was fitted or changed. The test contains **24,086 rows**, mapped one-to-one to `data/silver_modeling_data.csv`, from **2026-05-06 08:35:00-04:00** through **2026-11-03 01:45:00-05:00**.

The test partition was not used for calibration or selection according to the holdout integrity record. Results below are descriptive only.

## Calibration result and DOWN collapse

Raw predictions selected DOWN for **5,510** rows and UP for **12,091**. Sigmoid-calibrated argmax predictions selected DOWN for only **416** rows and UP for **17,536**. Thus the calibrated decision rule moved 5,094 rows away from DOWN and 5,445 rows toward UP.

The raw-to-calibrated change is visible in the persisted confusion matrices and class counts, not inferred from model internals. Probability quality improved modestly: raw versus calibrated log loss is reported in the holdout output as 1.08417 versus 1.08044, mean Brier 0.21947 versus 0.21868, and ECE 0.04201 versus 0.04052. However, accuracy changed from 0.383999 to 0.374741, balanced accuracy from 0.385300 to 0.380478, and DOWN F1 from 0.3164 to 0.0371.

The calibration experiment selected sigmoid on the 24,083-row validation partition after fitting on a 22,480-row chronological train suffix. Validation log loss improved from 1.04137 raw to 1.02943 calibrated and validation ECE from 0.03638 to 0.00484; isotonic was retained as a candidate but not selected. The final test results above are a separate descriptive holdout evaluation.

This establishes a probability-quality/argmax tradeoff. Calibration is not guaranteed to preserve class rankings in a multiclass one-vs-rest transformation, so a small improvement in proper scoring rules can coexist with a large shift in predicted classes. The evidence identifies the decision-distribution effect; it does not establish that calibration alone is the only causal mechanism beyond the persisted calibrated probabilities and argmax outcomes.

## Overall and class behavior

| Variant | Accuracy | Balanced accuracy | Macro-F1 | DOWN predictions | NEUTRAL predictions | UP predictions |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 0.383999 | 0.385300 | 0.375232 | 5,510 | 6,485 | 12,091 |
| Sigmoid calibrated | 0.374741 | 0.380478 | 0.301121 | 416 | 6,134 | 17,536 |

Actual support is DOWN 9,060, NEUTRAL 6,504, UP 8,522. Raw DOWN recall/F1 are 0.2544/0.3164; calibrated DOWN recall/F1 are 0.0194/0.0371. Calibrated UP recall rises to 0.7693, while NEUTRAL recall changes from 0.3659 to 0.3527. The model therefore does not provide balanced directional coverage under the calibrated argmax.

## Regime and sequence findings

Full metrics, counts, confusion matrices, confidence bands, class-conditioned confidence, chronological quartiles, local-time buckets, volatility tertiles, volume tertiles, and error-run statistics are machine-readable in `metrics.json`. Volatility uses the existing `volatility_20` feature; volume uses existing `relative_volume`. Tertile cut points are descriptive test-distribution summaries and were not used to tune anything.

Session analysis is limited to four Eastern local-time buckets because no official exchange-session identifier is stored. Small groups must not be over-interpreted; every group includes an `n` count in `metrics.json`.

Consecutive error statistics are calculated in chronological test order for both variants. They describe instability periods, not future performance.

## Robustness and trader-oriented interpretation

Chronological and volatility stability are supported only as descriptive subgroup analyses here. `src/experiments/feature_ablation.py` supports validation-only permutation diagnostics, but no new ablation experiment was executed in this task. Threshold sensitivity was not executed because changing or selecting a threshold would be a model-decision experiment and the frozen test must not be used for that purpose.

The current evidence does not justify unqualified actionable directional signals. Raw predictions have weak, uneven class performance; sigmoid calibration improves probability scoring but produces a severe DOWN-class collapse under argmax. NEUTRAL is present but only moderately discriminated. Before a trader could trust a signal, the project would need validation-only decision-policy analysis, larger-sample stability checks, explicit session/regime definitions, monitored calibration drift, and the separately required genuine live-market evaluation. None of these findings establish profitability or support real-money trading.

## Reproduction and limitations

The analysis is reproducible with `python -c "from src.experiments.error_analysis import write_report; write_report()"`. Source implementation is `src/experiments/error_analysis.py`; feature definitions are in `src/features/technical_indicators.py`, target/split logic in `src/models/train.py`, and validation-only ablation support in `src/experiments/feature_ablation.py`. No artifacts under `saved_models/` were written.

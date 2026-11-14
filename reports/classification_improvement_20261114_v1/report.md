# SilverAI Classification Improvement — Validation-Only Study

## Scope

This experiment uses the existing target, 5-minute candles, 15-minute horizon, threshold, 17-feature contract, chronological split and three-row purge. It loaded 160,573 validated rows to reconstruct the split boundaries. Candidate fitting and scoring used only the 112,398-row training partition and 24,083-row validation partition; zero test rows were passed to candidate fitting, tuning, scoring, or selection. The previously evaluated 24,086-row test partition was not used for model decisions.

Candidate development used three chronological folds inside training. The validation partition was then used once for final candidate comparison. No production model, default metadata, calibrator, threshold, or test output was changed.

## Validation comparison

| Candidate | Accuracy | Balanced accuracy | Macro F1 | Log loss | DOWN F1 | NEUTRAL F1 | UP F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dummy_most_frequent | 0.197276 | 0.333333 | 0.109847 | 28.933102 | 0.0000 | 0.3295 | 0.0000 |
| dummy_stratified | 0.321305 | 0.332848 | 0.317413 | 24.462630 | 0.3251 | 0.2610 | 0.3661 |
| logistic_regression | 0.429141 | 0.395521 | 0.363980 | 1.044333 | 0.1986 | 0.3322 | 0.5612 |
| hist_gradient_boosting_current | 0.421584 | 0.386803 | 0.389685 | 1.041370 | 0.3758 | 0.2957 | 0.4975 |
| hist_gradient_boosting_leaf31 | 0.426774 | 0.390546 | 0.391889 | 1.033786 | 0.3666 | 0.2990 | 0.5101 |
| hist_gradient_boosting_leaf63 | 0.421002 | 0.385128 | 0.387762 | 1.038967 | 0.3756 | 0.2898 | 0.4978 |
| random_forest_balanced | 0.417058 | 0.379697 | 0.383534 | 1.041669 | 0.4012 | 0.2708 | 0.4786 |
| random_forest_unweighted | 0.416892 | 0.382585 | 0.386411 | 1.043332 | 0.3944 | 0.2842 | 0.4807 |
| extra_trees_balanced | 0.422414 | 0.380041 | 0.382957 | 1.034746 | 0.4094 | 0.2543 | 0.4852 |
| hist_gradient_boosting_extended | 0.423120 | 0.368215 | 0.362048 | 1.038523 | 0.3651 | 0.2058 | 0.5153 |

The highest validation accuracy under the stated ordering was **logistic_regression**. Full per-class precision, recall, F1, confusion matrices, predicted distributions, artifact hashes, and chronological-fold results are in `metrics.json`.

The current HistGradientBoosting baseline is included as `hist_gradient_boosting_current`; majority and stratified dummy baselines are included. Random Forest, Extra Trees, logistic regression, two HGB configurations, and a causal extended-feature HGB candidate were tested. XGBoost availability: **False**. LightGBM availability: **True**.

## Decision-rule study

Class-probability multipliers were selected using only the three chronological training folds for the current HGB model. The selected multipliers and all candidates are recorded in `metrics.json`; the adjusted result is reported separately from underlying classifier performance. It was not evaluated on the historical test partition and was not activated.

## Interpretation

Validation improvement is only meaningful if it is consistent across chronological folds and does not materially sacrifice balanced accuracy, macro F1, or DOWN/NEUTRAL performance. This study does not justify replacing the default model until the candidate’s fold stability, validation trade-offs, and an independently frozen future evaluation are reviewed. No validation result establishes live performance or profitability.

## Reproduction

```powershell
python -c "from src.experiments.classification_improvement import run_experiment; run_experiment()"
```

Artifacts are stored under this versioned directory. The exact dataset hash, feature schema, split boundaries, configuration, commands, and test-protection assertions are in `metrics.json`.

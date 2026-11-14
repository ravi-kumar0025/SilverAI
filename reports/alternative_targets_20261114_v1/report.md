# SilverAI Alternative Target and Volatility-Adjusted Label Experiments

## Integrity and baseline

This campaign stage preserves the official 5-minute / 15-minute / three-class target, 17 approved features, and training-derived threshold `0.0776397515528126%`. It used chronological development folds inside the original training partition and a controlled validation comparison. Zero rows from the frozen historical test partition were passed to fitting, tuning, scoring, or selection.

The original-target frozen test baseline remains 38.40% accuracy. Alternative-target results below are validation results for different label definitions and must not be compared as if they were the same prediction problem.

## Executed target comparisons

| Target | Model | Train n | Validation n | Accuracy | Balanced accuracy | Macro F1 | DOWN F1 | NEUTRAL F1 | UP F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_5m | hgb_current | 123122 | 26039 | 0.4153 | 0.3847 | 0.3890 | 0.4018 | 0.2951 | 0.4701 |
| fixed_5m | hgb_tuned | 123122 | 26039 | 0.4212 | 0.3903 | 0.3936 | 0.3909 | 0.3027 | 0.4871 |
| fixed_5m | logistic | 123122 | 26039 | 0.4254 | 0.3948 | 0.3773 | 0.2700 | 0.3197 | 0.5422 |
| fixed_10m | hgb_current | 117061 | 24948 | 0.4248 | 0.3928 | 0.3962 | 0.3989 | 0.3006 | 0.4892 |
| fixed_10m | hgb_tuned | 117061 | 24948 | 0.4321 | 0.3982 | 0.3994 | 0.3803 | 0.3062 | 0.5118 |
| fixed_10m | logistic | 117061 | 24948 | 0.4271 | 0.3971 | 0.3745 | 0.2432 | 0.3317 | 0.5487 |
| fixed_15m | hgb_current | 112395 | 24080 | 0.4194 | 0.3850 | 0.3874 | 0.3702 | 0.2957 | 0.4964 |
| fixed_15m | hgb_tuned | 112395 | 24080 | 0.4260 | 0.3907 | 0.3921 | 0.3617 | 0.3043 | 0.5104 |
| fixed_15m | logistic | 112395 | 24080 | 0.4291 | 0.3955 | 0.3641 | 0.1992 | 0.3322 | 0.5610 |
| volatility_adjusted_15m | hgb_current | 112395 | 24080 | 0.4181 | 0.3614 | 0.3014 | 0.1320 | 0.5744 | 0.1978 |
| volatility_adjusted_15m | hgb_tuned | 112395 | 24080 | 0.4176 | 0.3579 | 0.2881 | 0.1075 | 0.5760 | 0.1808 |
| volatility_adjusted_15m | logistic | 112395 | 24080 | 0.4212 | 0.3583 | 0.2758 | 0.0263 | 0.5785 | 0.2227 |
| dynamic_neutral_15m | hgb_current | 112395 | 24080 | 0.3925 | 0.3503 | 0.3208 | 0.3741 | 0.0975 | 0.4908 |
| dynamic_neutral_15m | hgb_tuned | 112395 | 24080 | 0.3904 | 0.3451 | 0.3100 | 0.3395 | 0.0863 | 0.5042 |
| dynamic_neutral_15m | logistic | 112395 | 24080 | 0.3970 | 0.3395 | 0.2570 | 0.1628 | 0.0565 | 0.5517 |
| triple_barrier_15m | hgb_current | 112395 | 24080 | 0.4501 | 0.3884 | 0.3286 | 0.2012 | 0.6136 | 0.1708 |
| triple_barrier_15m | hgb_tuned | 112395 | 24080 | 0.4522 | 0.3896 | 0.3275 | 0.2076 | 0.6146 | 0.1603 |
| triple_barrier_15m | logistic | 112395 | 24080 | 0.4440 | 0.3804 | 0.3131 | 0.2038 | 0.6046 | 0.1309 |
| trend_scanning_15m | hgb_current | 112395 | 24080 | 0.4391 | 0.3778 | 0.3203 | 0.1917 | 0.6016 | 0.1675 |
| trend_scanning_15m | hgb_tuned | 112395 | 24080 | 0.4427 | 0.3801 | 0.3198 | 0.1879 | 0.6040 | 0.1676 |
| trend_scanning_15m | logistic | 112395 | 24080 | 0.4319 | 0.3642 | 0.2848 | 0.0998 | 0.5959 | 0.1588 |
| fixed_30m | hgb_current | 102540 | 22202 | 0.4192 | 0.3885 | 0.3921 | 0.3959 | 0.2986 | 0.4820 |
| fixed_30m | hgb_tuned | 102540 | 22202 | 0.4206 | 0.3887 | 0.3915 | 0.3834 | 0.2993 | 0.4918 |
| fixed_30m | logistic | 102540 | 22202 | 0.4238 | 0.3921 | 0.3464 | 0.1353 | 0.3387 | 0.5651 |
| fixed_60m | hgb_current | 89765 | 19823 | 0.4128 | 0.3868 | 0.3892 | 0.3997 | 0.2971 | 0.4709 |
| fixed_60m | hgb_tuned | 89765 | 19823 | 0.4131 | 0.3850 | 0.3870 | 0.3775 | 0.2986 | 0.4848 |
| fixed_60m | logistic | 89765 | 19823 | 0.4173 | 0.3886 | 0.3386 | 0.1110 | 0.3448 | 0.5602 |

The machine-readable report contains confusion matrices, class proportions, prediction distributions, log loss, chronological fold metrics, parameters and artifact hashes for every row above.

Definitions executed:

- `fixed`: fixed threshold estimated from the fitting partition.
- `volatility_adjusted`: threshold equals a training-fitted return-volatility scale times causal `volatility_20`.
- `dynamic_neutral`: prior 200-bar absolute-return quantile, shifted by one row, with a training-only fallback.
- `triple_barrier`: volatility-scaled profit/stop barriers with a horizon-specific maximum holding period.
- `trend_scanning`: bounded 1/2/3/6/12-bar forward scan, retained as an explicitly experimental approximation rather than a claim of a full trend-scanning implementation.
- `two_stage_official_15m`: movement versus neutral followed by UP versus DOWN conditional on movement; both stages and end-to-end results are reported separately.

## Conclusions

Alternative label accuracy is not automatically evidence of improvement on the official target. A label definition can be easier, change class balance, or remove ambiguous rows. The report therefore emphasizes label coverage, class proportions, sample counts, directional class metrics and chronological stability alongside accuracy.

The executed evidence does not activate any candidate or establish a defensible route to 60% accuracy. The strongest candidates for further review are those that improve DOWN and NEUTRAL discrimination without relying on reduced coverage or materially changed labels; exact candidates are identified from `metrics.json` rather than selected here for production.

Observed results from the executed validation comparisons:

- For fixed labels, the 10-minute horizon with tuned HGB had the highest fixed-target accuracy at approximately 43.21%; 15-minute tuned HGB reached approximately 42.60%. These are separate targets and are not evidence that the 10-minute target improves the official 15-minute task.
- Volatility-adjusted labels produced roughly 40.12% neutral coverage and weak directional F1, so they did not improve DOWN/NEUTRAL discrimination.
- Dynamic neutral labels changed class balance substantially and reduced NEUTRAL F1; they are not a demonstrated improvement.
- Triple-barrier labels reached approximately 45.22% validation accuracy with tuned HGB, but NEUTRAL accounted for approximately 40.99% of labels and directional F1 was only about 0.21 DOWN / 0.16 UP. This is an easier, materially different target rather than a 45% solution to the official problem.
- The bounded trend-scanning approximation produced approximately 40.21% NEUTRAL labels and about 44.27% validation accuracy with tuned HGB, but directional F1 remained weak. It is not a full trend-scanning implementation.
- The two-stage official-15-minute model detected movement at about 80.36% accuracy, but end-to-end accuracy was about 42.28% for tuned HGB and 42.69% for logistic regression. The movement score therefore must not be presented as directional performance.

None of the executed alternatives supports the 60% stretch goal. The strongest remaining research opportunities are stable causal feature selection, proper target-specific walk-forward validation, and acquisition of aligned external market information; none should be activated without a new untouched evaluation period.

## Reproduction

```powershell
python -c "from src.experiments.alternative_target_experiments import run_experiment; run_experiment()"
```

All candidate artifacts are isolated under this directory. No default model, metadata, calibration artifact, or prior report was overwritten.

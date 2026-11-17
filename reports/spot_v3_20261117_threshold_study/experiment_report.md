# Regression threshold study

Status: **no_defensible_threshold**.

This isolated study uses chronological OOS predictions from the existing LightGBM mean/Q10/Q50/Q90 architecture, the exact 34-feature schema, and target `log(close[t+12]/close[t])`. The configured final holdout was excluded before fitting and selection.

Thresholds tested: 2 bp, 3 bp, 4 bp, 5 bp, 6 bp, 8 bp, 10 bp, 12 bp, 15 bp, 20 bp; decimal log-return conversion is `bp / 10000`. Costs: 2.0 bp spread + 1.0 bp slippage per side.

See `threshold_comparison.csv`, `threshold_comparison_per_fold.csv`, and `selected_threshold.json` for the complete evidence.

# 15-minute multiclass and regression model

`pred_1` is the base Down/Neutral/Up classifier. `pred_2` contains the calibrated classifier, selected `k=0.5` and neutral threshold `0.34`, plus 15-minute mean and 10/50/90% quantile return regressors. The final test blocks are untouched by selection. Training uses the current manifest dataset ID `3b0693b8941795ee` and records the live-log extension in `metadata.json`.

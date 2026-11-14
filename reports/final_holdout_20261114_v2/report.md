# SilverAI final holdout evaluation

This is a historical evaluation of frozen `20261114_v1` artifacts. It is not the official five-day live-market test.

- Dataset rows evaluated: **24086**
- Dataset SHA-256: `fa5f5a9270324d10efc7add52964ade00f908507137ea3b4ee0f0d64109879a5`
- Classifier SHA-256: `20bf078b9836f8d41910af8430eed80c5394ef14184aa5318a4fbe30b9bf1c38`
- Sigmoid calibrator SHA-256: `01df86c8b1b0e1a62a673a0348defe19e00940ad4055278df8ed0719e724a306`
- Test rows excluded: **0**
- Test rows used for fitting, selection, or calibration: **0**
- Actual class distribution: `{"DOWN": 9060, "NEUTRAL": 6504, "UP": 8522}`
- Raw predicted class distribution: `{"DOWN": 5510, "NEUTRAL": 6485, "UP": 12091}`
- Calibrated predicted class distribution: `{"DOWN": 416, "NEUTRAL": 6134, "UP": 17536}`

## Metrics

| Metric | Raw | Calibrated |
|---|---:|---:|
| Multiclass log loss | 1.08416559 | 1.08044059 |
| Mean Brier | 0.21947113 | 0.21867652 |
| ECE | 0.04201483 | 0.04051516 |
| Accuracy | 0.38399900 | 0.37474051 |
| Balanced accuracy | 0.38529957 | 0.38047835 |

## Per-class results

| Class | Raw precision | Raw recall | Raw F1 | Calibrated precision | Calibrated recall | Calibrated F1 |
|---|---:|---:|---:|---:|---:|---:|
| DOWN | 0.41833031 | 0.25441501 | 0.31640357 | 0.42307692 | 0.01942605 | 0.03714648 |
| NEUTRAL | 0.36700077 | 0.36592866 | 0.36646393 | 0.37398109 | 0.35270603 | 0.36303213 |
| UP | 0.37747085 | 0.53555503 | 0.44282734 | 0.37385949 | 0.76930298 | 0.50318520 |

## Confusion matrices

Raw:

```text
[[2305, 2182, 4573], [1170, 2380, 2954], [2035, 1923, 4564]]
```

Calibrated:

```text
[[176, 2031, 6853], [83, 2294, 4127], [157, 1809, 6556]]
```

## Reproduction

`python -c "from src.experiments.final_holdout_evaluation import run_final_holdout_evaluation; run_final_holdout_evaluation('saved_models/experiments/calibration_20261114_v1', 'reports/final_holdout_20261114')"`

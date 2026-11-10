# SilverAI

SilverAI is an Option B live-web-app implementation of the project brief in `ps.pdf`: a research and prototype prediction system for one declared asset, COMEX silver futures. It currently consumes five-minute OHLCV data, creates causal features, trains chronological baseline models, and exposes predictions through a FastAPI backend and React dashboard.

The project is analytical research software, not a trading system or a claim of profitable prediction. The current untouched-test results show modest classification performance and regression performance approximately equal to a zero-return baseline.

## Problem and objective

SilverAI investigates whether information available at the close of a five-minute silver candle can support a forecast for the next three consecutive five-minute candles (15 minutes). It produces two independent outputs:

- a three-class direction: `DOWN`, `NEUTRAL`, or `UP`;
- a continuous predicted 15-minute return, expressed in percent.

The outputs are combined into descriptive scenarios and an analytical signal. Model disagreement is retained as `MIXED`; the system does not force a directional conclusion.

## Data

The repository currently contains these local CSV artifacts under `data/`:

| File | Role |
| --- | --- |
| `SI_1m_1098d.csv` | Raw one-minute silver OHLCV data. |
| `silver_5m.csv` | Five-minute OHLCV candles used by the feature workflow. |
| `silver_features.csv` | Five-minute candles plus the 17 engineered features. |
| `silver_modeling_data.csv` | Feature data plus `future_return_15m` and `target_direction_15m`. |
| `SI_daily_roll-adjusted.csv` | Separate daily roll-adjusted OHLCV context data. |

The notebooks inspect these files locally. The external acquisition provenance is not recorded in the source code, so users should treat the checked-in CSVs as supplied research inputs and document their own data vendor/source when reproducing the dataset.

All intraday timestamps are interpreted as timezone-aware `America/New_York` values. The loader preserves all source rows and columns while validating required schema.

## Architecture

```text
data/*.csv
   -> src.data.loader
   -> src.data.preprocessing
   -> src.features.technical_indicators
   -> notebook 06 target creation / silver_modeling_data.csv
   -> src.models.train
   -> saved_models + reports/model_comparison
   -> src.models.predict
   -> src.prediction.pipeline
      -> scenarios -> signals
   -> backend FastAPI
   -> frontend React/Vite dashboard
```

Detailed responsibilities are in [`docs/architecture.md`](docs/architecture.md). The notebook sequence and reproducibility commands are in [`docs/reproducibility.md`](docs/reproducibility.md).

## Features and target

The approved 17-feature input is:

`return_1`, `return_3`, `return_6`, `volatility_5`, `volatility_20`, `volatility_50`, `sma_distance_20`, `sma_distance_50`, `candle_return`, `candle_range_pct`, `upper_wick_ratio`, `lower_wick_ratio`, `relative_volume`, `volume_change`, `hour_sin`, `hour_cos`, `day_of_week`.

Features use only the current and historical candle information. Rolling warm-up values remain missing and are not silently filled by the feature module. The target notebook requires three consecutive five-minute observations, then computes:

```text
future_return_15m = (close[t+3] - close[t]) / close[t] * 100
```

The training workflow recalibrates the neutral band from the purged training partition only. The saved threshold is `±0.0776397515528126%`; the original notebook's `±0.05%` label is explicitly provisional and is not the final training threshold.

## Models and evaluation

The training module evaluates three classifiers and three regressors:

- classifiers: logistic regression, random forest, histogram gradient boosting;
- regressors: linear regression, random forest, histogram gradient boosting.

Selection uses validation performance only: highest macro F1 for classification and lowest RMSE for regression. The split is chronological with nominal fractions 70% / 15% / 15% and a three-row purge at train/validation boundaries. Test data is not used for selection.

The selected saved artifacts are:

- `hist_gradient_boosting_classifier`
- `linear_regression`

Evaluation on the untouched test partition produced:

| Family | Metric | Selected model | Zero/majority baseline |
| --- | ---: | ---: | ---: |
| Classification | Accuracy | 0.38375 | 0.27003 majority accuracy |
| Classification | Balanced accuracy | 0.38565 | 0.33333 |
| Classification | Macro F1 | 0.37312 | 0.14175 |
| Regression | MAE | 0.22040 | 0.22018 zero-return MAE |
| Regression | RMSE | 0.31952 | 0.31940 zero-return RMSE |
| Regression | R² | -0.00084 | -0.00010 |

These figures are descriptive results from the available artifacts, not evidence of deployable trading performance. Full tables are under `reports/model_comparison/`.

## Prediction, scenarios, and reliability

`src.models.predict` loads the selected classifier and regressor, validates the exact feature order, and returns independent outputs. `src.prediction.pipeline` adds model metadata, scenarios, reliability limitations, and a final signal.

Scenario probabilities are copied from valid classifier probabilities only. The scenario module reports supporting evidence, contradicting evidence, forecast information, and model disagreement. `signals.py` emits `BULLISH`, `BEARISH`, `NEUTRAL`, `MIXED`, or `INSUFFICIENT_DATA`.

`src/models/calibration.py` provides an optional chronological train-suffix calibrator. The current checked-in model artifacts do not include a calibrator, so the runtime reports calibrated probabilities as unavailable rather than treating raw probabilities as confidence.

## Backend and frontend

The FastAPI backend is organized as:

- `backend/main.py` — application and CORS setup;
- `backend/routes/api.py` — validated REST contracts;
- `backend/services/prediction_service.py` — lazy, cached model access;
- `backend/core/config.py` — environment-driven paths and CORS.

Available routes are documented in [`docs/api.md`](docs/api.md). Confirmed predictions are persisted in SQLite through the confirmed-candle route and prediction journal; the dashboard can recover the latest durable record after restart.

The React/Vite frontend in `frontend/` consumes backend health, model, journal, evaluation, and latest-prediction endpoints. Historical CSV data remains explicitly labeled historical; it is never presented as live data.

## Installation and execution

Prerequisites: Python 3.10+ with the project scientific dependencies available, Node.js 22+ and npm, and the supplied CSV/model artifacts.

From the repository root:

```powershell
python -m pip install -r requirements.txt
pytest -q
```

Start the API:

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Start the dashboard in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Set `frontend/.env` when the API is not on the default URL:

```env
VITE_API_BASE_URL=http://localhost:8000
```

The backend expects the selected artifacts at `saved_models/` and selection metadata at `reports/model_comparison/evaluation_selection.json`. These can be overridden with `SILVERAI_MODEL_ROOT`, `SILVERAI_SELECTION_PATH`, and `SILVERAI_CORS_ORIGINS`.

To regenerate models and evaluation reports, use the Python modules from the repository root:

```powershell
python -c "from src.models.train import run_training; run_training()"
python -c "from src.models.evaluate import evaluate_saved_models; evaluate_saved_models()"
```

The notebook workflow is described in [`docs/reproducibility.md`](docs/reproducibility.md).

## Limitations and future work

- The data acquisition provenance is not encoded in the repository.
- The target-creation workflow remains notebook-based rather than a reusable source module.
- Regression test performance is effectively at the zero-return baseline.
- Confidence calibration and authenticated live-provider integration are not implemented.
- A SQLite prediction journal, closed-candle orchestrator, frozen evaluation manifest, and cost/slippage-aware paper evaluator exist; they are not a live trading system.
- Authenticated live data, contract-roll verification, and the PDF's five-trading-day continuous test remain outside the verified scope until a genuine live feed is configured.
- Research papers justify hypotheses and validation practices; they do not establish SilverAI feature superiority or predictive novelty.

The existing literature rationale and four-paper review are in [`reports/feature_research_analysis.md`](reports/feature_research_analysis.md).

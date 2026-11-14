# SilverAI reproducibility

## Prerequisites

- Windows PowerShell or an equivalent shell.
- Python 3.10 or newer with pandas, NumPy, scikit-learn, joblib, matplotlib, pytest, FastAPI, and Uvicorn available.
- Node.js and npm for the dashboard.
- The five CSV datasets in `data/`.
- Saved models under `saved_models/` and selection metadata under `reports/model_comparison/` for prediction/API execution.

The complete Python dependency file is `requirements.txt`; `backend/requirements.txt` is the smaller runtime subset. The frontend dependency lockfile is `frontend/package-lock.json`.

## Install

From the project root:

```powershell
python -m pip install -r requirements.txt
```

Install the scientific stack used by the source modules if it is not already present. A typical environment is:

```powershell
python -m pip install pandas numpy scikit-learn joblib matplotlib pytest
```

Install the frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## Data and notebook order

Place the supplied files in `data/` with these exact names:

```text
SI_1m_1098d.csv
SI_daily_roll-adjusted.csv
silver_5m.csv
silver_features.csv
silver_modeling_data.csv
```

Run notebooks in this order from the repository root or open them in Jupyter with the repository as the working directory:

1. `notebooks/01_data_collection.ipynb` — inspect local raw datasets.
2. `notebooks/02_data_understanding.ipynb` — inspect distributions and schema.
3. `notebooks/03_data_preprocessing.ipynb` — inspect and prepare five-minute data.
4. `notebooks/04_exploratory_data_analysis.ipynb` — exploratory analysis.
5. `notebooks/05_feature_engineering.ipynb` — create `silver_features.csv`.
6. `notebooks/06_target_creation.ipynb` — create `silver_modeling_data.csv` with valid three-bar targets.
7. `notebooks/07_model_training.ipynb` — train and save model artifacts.
8. `notebooks/08_model_evaluation.ipynb` — evaluate selected models and write reports.
9. `notebooks/09_scenario_analysis.ipynb` — inspect scenario behavior.

The reusable source modules can be used instead of notebook cells for the implemented parts:

```powershell
python -c "from src.models.train import run_training; run_training()"
python -c "from src.models.evaluate import evaluate_saved_models; evaluate_saved_models()"
python -c "from src.data.loader import load_5m_data; from src.experiments.timeframe_horizon import run_timeframe_horizon_experiment, save_experiment_report; save_experiment_report(run_timeframe_horizon_experiment(load_5m_data(), [(5,10),(5,15),(5,30),(10,20),(10,30),(15,15),(15,30),(30,30)]), 'reports/timeframe_horizon_experiment')"
```

Target creation is currently still implemented in notebook 06, so that notebook must run before training unless an existing `silver_modeling_data.csv` is supplied.

## Verification

Run the automated tests:

```powershell
pytest -q
```

The verified workspace result is `36 passed` with two warnings. The tests cover model metrics, prediction pipeline behavior, scenarios, signals, backend endpoint contracts, candle validation, timestamp-aware targets, prediction persistence and immutability, frozen manifests, calibration, baselines, and paper-trading calculations.

Run the production frontend build:

```powershell
cd frontend
npm run build
cd ..
```

The verified build completed successfully with Vite.

## Train and evaluate

Training writes model artifacts below `saved_models/` and metadata under `saved_models/metadata/`:

```powershell
python -c "from src.models.train import run_training; result=run_training(); print(result['saved_paths'])"
python -c "from src.models.evaluate import evaluate_saved_models; result=evaluate_saved_models(); print(result['output_paths'])"
```

Evaluation rewrites the declared report outputs under `reports/model_comparison/` and `reports/figures/`. Do not use test metrics for model selection.

## Run the application

Start the backend from the project root:

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal start the frontend:

```powershell
cd frontend
npm run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173`. The frontend uses `VITE_API_BASE_URL=http://localhost:8000` by default. The backend can also be checked at `http://localhost:8000/docs` and `http://localhost:8000/api/health`.

## Generate a prediction directly

The API requires all 17 engineered feature values. The frontend does not fabricate these values. To call the API programmatically, construct a JSON object matching [`docs/api.md`](api.md) and send it to `/api/predict` or send multiple objects to `/api/predict/batch`.

The Python service can also be used directly after loading the selected artifacts:

```powershell
python -c "import pandas as pd; from backend.services.prediction_service import PredictionService; from src.features.technical_indicators import FEATURE_COLUMNS; row=pd.read_csv('data/silver_modeling_data.csv')[list(FEATURE_COLUMNS)].tail(1).to_dict('records'); print(PredictionService().predict(row))"
```

## Reproducibility caveats

Model artifacts are binary files and dataset contents are not committed by the repository ignore policy. Exact results require the same CSV data, saved metadata, library versions, and model artifacts. Predictions are persisted in the SQLite path configured by `SILVERAI_PREDICTION_DB_PATH` (default `data/silverai_predictions.sqlite3`). Calibration support includes leakage-safe sigmoid/isotonic candidates, validation-only selection, multiclass Brier/log-loss/ECE metrics, and an optional persisted artifact. The current checked-in artifacts do not include a calibrator, so historical raw probabilities remain explicitly uncalibrated. Historical CSV data is never reported as live.

The non-destructive final holdout command is:

```powershell
python -c "from src.experiments.final_holdout_evaluation import run_final_holdout_evaluation; run_final_holdout_evaluation('saved_models/experiments/calibration_20261114_v1', 'reports/final_holdout_20261114')"
```

It writes a versioned report and metrics file without changing the default production artifacts. The holdout report is historical evidence only and must not be presented as the official live-market test.

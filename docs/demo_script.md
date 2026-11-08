# SilverAI live demonstration script

## Before the session

From the repository root, confirm that `data/silver_modeling_data.csv`, `saved_models/metadata/training_metadata.json`, and `reports/model_comparison/evaluation_selection.json` exist. Start the API in one terminal and the frontend in another:

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
cd frontend
npm run dev
```

Open `http://localhost:5173` and `http://localhost:8000/docs`. Keep the presentation open at `docs/presentation.html`. The demonstration uses local artifacts only and does not require an external data service.

## Sequence and speaking notes

### 1. Problem statement

“SilverAI studies a narrow question: can information known at the close of a five-minute COMEX silver candle support a 15-minute directional and return forecast? This is a research prototype, not a profitability claim.”

Use slides 1–2.

### 2. Dataset and preprocessing

Show slide 3. Point out the one-minute, five-minute, engineered, modeling, and separate daily files. Explain that timestamps remain in Eastern Time, data rows are not silently deleted by the loader, and session gaps are reported rather than treated as automatic errors.

### 3. Feature engineering

Show slide 5. Explain the six groups: momentum, volatility, trend, candle geometry, volume, and time. Emphasize that rolling features use current and prior rows only.

### 4. Target construction

Show slide 6. Explain that the target uses the close three valid five-minute bars ahead. A row crossing a timestamp gap is excluded. The notebook's diagnostic band is provisional; training recalibrates the neutral threshold from training data only.

### 5. Training and evaluation

Show slides 7–8. State the chronological 70/15/15 split and three-row purge. Name the selected models. Present the actual untouched-test results without rounding them into a stronger claim: classification macro F1 is 0.37312 and regression R² is -0.00084, while regression RMSE is essentially the zero-return baseline.

### 6. Launch and check the application

In the browser, refresh `http://localhost:8000/api/health` or open Swagger. Show that the selected artifacts are available. Then open the frontend. Point out the explicit unavailable states for live price and history.

### 7. Generate a real prediction

Use a real row from the modeling CSV without adding fabricated values:

```powershell
python -c "import pandas as pd; from backend.services.prediction_service import PredictionService; from src.features.technical_indicators import FEATURE_COLUMNS; row=pd.read_csv('data/silver_modeling_data.csv')[list(FEATURE_COLUMNS)].tail(1).to_dict('records'); print(PredictionService().predict(row))"
```

The verified example returned `UP`, a 15-minute return of approximately `0.006886%`, and a `MIXED` signal. Explain that the mixed signal reflects model disagreement rather than a forced directional label.

To demonstrate the REST contract, use Swagger's `POST /api/predict` with the 17 feature fields from `docs/api.md`. Then refresh the dashboard. The dashboard reads the result from the API and renders probabilities, scenarios, evidence, limitations, and metadata.

### 8. Explain the output

Use slides 9–10. Clarify that class probabilities describe the current classifier output, not historical reliability. Scenario probabilities appear only when valid class probabilities exist. Reliability remains limited because no calibrated confidence module exists.

### 9. Close with limitations

Use slide 11. State the negative regression R², lack of persistence and live price feed, missing confidence calibration, and absence of trading-cost analysis. Finish with the next research steps rather than claiming deployment readiness.

## Recovery notes

- If `/api/health` reports degraded status, verify the saved model root and selection JSON path.
- If the dashboard shows “No prediction loaded,” generate a prediction through Swagger or the direct Python command, then click `Refresh latest`.
- If the frontend cannot reach the API, check `VITE_API_BASE_URL` and the backend CORS origin.
- If a feature request returns HTTP 422, compare its fields exactly with `docs/api.md`.

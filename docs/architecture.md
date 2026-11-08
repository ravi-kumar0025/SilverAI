# SilverAI architecture

## System boundary

SilverAI has four executable layers and a notebook research layer:

```text
CSV datasets
  -> data loading and quality preparation
  -> causal feature generation
  -> notebook target creation
  -> chronological model training/evaluation
  -> saved artifacts
  -> prediction orchestration
  -> FastAPI
  -> React dashboard
```

The daily roll-adjusted dataset remains separate from the intraday feature/model path. No daily price levels are combined with five-minute prices in the current implementation.

## Components

### Data layer

`src/data/loader.py` resolves paths relative to the repository, loads the five dataset variants, validates required columns, and parses timestamps into `America/New_York`. It does not engineer features or create targets.

`src/data/preprocessing.py` performs non-destructive timestamp preparation, sorting, duplicate inspection, OHLCV validation, missing-value counts, and inferred time-gap reporting. It preserves invalid rows for inspection.

### Feature layer

`src/features/technical_indicators.py` implements the approved 17 features. Rolling operations use current and prior observations only. The module sorts a copy of the data, retains warm-up `NaN` values, and does not modify source files.

### Target and modeling layer

`notebooks/06_target_creation.ipynb` constructs `future_return_15m` from the close three valid five-minute bars ahead and creates the provisional source label `target_direction_15m`. The modeling CSV contains these target columns.

`src/models/train.py` re-creates training labels from the continuous return using a threshold calibrated from the purged training partition. It prepares usable rows, performs chronological splitting, fits the candidate models, selects using validation metrics, and saves joblib artifacts and metadata.

`src/models/evaluate.py` reloads saved artifacts and evaluates all saved candidates on train, validation, and untouched test partitions. It verifies split metadata and writes tables, confusion-matrix data, metadata, and a figure under `reports/`.

### Prediction layer

`src/models/predict.py` validates the exact 17-feature schema and loads the selected classifier/regressor. It preserves optional `datetime_et`, returns fixed `DOWN`/`NEUTRAL`/`UP` probability fields, and returns the continuous 15-minute forecast.

`src/prediction/pipeline.py` combines model outputs with scenario generation and signal interpretation. `scenarios.py` keeps classifier/regressor disagreement explicit. `signals.py` returns an analytical label and evidence; it does not claim a calibrated confidence score.

### API layer

`backend/main.py` creates the FastAPI app and configures CORS. `backend/services/prediction_service.py` loads selected artifacts lazily and caches them for the process lifetime. `backend/routes/api.py` defines strict Pydantic input schemas and maps model/input failures to safe HTTP responses without exposing internal paths.

### UI layer

`frontend/src/services/api.js` is the single API client. `frontend/src/main.jsx` renders overview, probabilities, scenarios, reliability, metadata, and historical empty states. `frontend/src/styles.css` provides dark/light themes and responsive layouts. The UI uses actual API data only and labels unavailable data rather than inventing it.

## State and persistence

Model artifacts are read-only at prediction time. The backend caches loaded artifacts and stores only the latest result in memory. Restarting the API clears the latest result. A database or historical prediction store is intentionally not part of the current architecture.

## Configuration

Backend environment variables:

- `SILVERAI_MODEL_ROOT` — model artifact root, default `saved_models/`.
- `SILVERAI_SELECTION_PATH` — selection JSON, default `reports/model_comparison/evaluation_selection.json`.
- `SILVERAI_CORS_ORIGINS` — comma-separated allowed origins.

Frontend environment variable:

- `VITE_API_BASE_URL` — API origin, default `http://localhost:8000`.

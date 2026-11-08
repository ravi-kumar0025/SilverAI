# SilverAI API

The backend is FastAPI and runs from the repository root with:

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

## Feature request schema

`POST /api/predict` accepts one JSON object. All 17 feature fields are required and must be finite numeric values. `datetime_et` is optional and may be supplied as a string. Unknown fields are rejected.

```json
{
  "datetime_et": "2026-11-03T01:45:00-05:00",
  "return_1": 0.01,
  "return_3": 0.02,
  "return_6": 0.03,
  "volatility_5": 0.01,
  "volatility_20": 0.02,
  "volatility_50": 0.03,
  "sma_distance_20": 0.01,
  "sma_distance_50": 0.02,
  "candle_return": 0.01,
  "candle_range_pct": 0.02,
  "upper_wick_ratio": 0.1,
  "lower_wick_ratio": 0.1,
  "relative_volume": 1.0,
  "volume_change": 0.1,
  "hour_sin": 0.5,
  "hour_cos": 0.5,
  "day_of_week": 2
}
```

`POST /api/predict/batch` accepts:

```json
{ "observations": [ /* one or more feature objects */ ] }
```

## Endpoints

### `GET /api/health`

Loads the selected artifacts lazily and returns availability without exposing internal paths.

```json
{
  "status": "ok",
  "models_available": true,
  "classifier_name": "hist_gradient_boosting_classifier",
  "regressor_name": "linear_regression"
}
```

### `GET /api/models`

Returns selected model names, the exact 17-feature order, targets, and horizon metadata.

### `POST /api/predict`

Returns one prediction envelope containing `predicted_direction_name`, `predicted_return_15m`, `class_probabilities`, `reliability`, `scenarios`, `signal`, `model_metadata`, warnings, and a one-item `predictions` list. A real saved-model response produced during verification included:

```json
{
  "forecast_horizon_minutes": 15,
  "predicted_direction_name": "UP",
  "predicted_return_15m": 0.006886337512596169,
  "class_probabilities": {
    "down": 0.3452284854,
    "neutral": 0.2337205971,
    "up": 0.4210509174
  },
  "signal": "MIXED",
  "model_metadata": {
    "classifier_name": "hist_gradient_boosting_classifier",
    "regressor_name": "linear_regression"
  }
}
```

The displayed example omitted the full scenario and warning payload for readability. The timestamp is `null` when the request omits `datetime_et`.

### `POST /api/predict/batch`

Runs the same pipeline for multiple observations and returns `predictions`, aggregate warnings, and model metadata. Input order is preserved.

### `GET /api/predictions/latest`

Returns the latest in-memory prediction result. It returns HTTP 404 until a prediction has been generated after the current backend process started.

## Errors

- `422` — missing/extra/invalid feature values or invalid model metadata.
- `404` — no in-memory latest prediction.
- `503` — model artifacts unavailable.
- `500` — unexpected prediction failure; internal exception details are logged server-side, not returned.

## CORS and paths

Allowed origins are configured with `SILVERAI_CORS_ORIGINS`. Artifact paths are configured with `SILVERAI_MODEL_ROOT` and `SILVERAI_SELECTION_PATH`.

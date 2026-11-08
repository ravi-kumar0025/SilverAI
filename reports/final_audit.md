# SilverAI final pre-execution audit

Audit date: 2026-11-08  
Scope: read-only repository, interface, methodology, dependency, and execution-readiness audit.  
Training was not run and existing model/data artifacts were not overwritten.

## Recommendation

**NO-GO for a clean, first-time full execution until the HIGH findings are addressed.**

The current workspace is operational with its existing Python environment, datasets, saved models, reports, and frontend dependencies. Imports, tests, saved-model inference, API startup, and the frontend build all work. However, a new developer cannot reproduce the complete workflow from the declared dependency files alone, and the selected histogram-gradient-boosting models use scikit-learn's default internal early-stopping holdout. That caveat should be resolved or explicitly accepted before treating a fresh full run as methodologically final.

No CRITICAL finding was identified that invalidates the existing test metrics. The HIGH findings are execution/reproducibility and methodological-readiness blockers rather than observed runtime failures in the current environment.

## 1. Repository inventory

### Present and non-empty

- Notebooks `01_data_collection.ipynb` through `09_scenario_analysis.ipynb` all exist. Each contains executable code cells and stored outputs.
- `src/data/`: `loader.py`, `preprocessing.py`.
- `src/features/`: `technical_indicators.py`.
- `src/models/`: `train.py`, `evaluate.py`, `predict.py`.
- `src/prediction/`: `pipeline.py`, `scenarios.py`, `signals.py`.
- Backend: FastAPI application, routes, configuration, and cached prediction service.
- Frontend: Vite/React application, API service, dashboard, theme CSS, lockfile, and environment example.
- Tests: backend, evaluation, pipeline, prediction, scenarios, and signals.
- Documentation: README, architecture, methodology, API, reproducibility, demo, presentation, and viva documents.

### Empty or disconnected structure

**LOW — unused frontend directories.** `frontend/public/`, `frontend/src/components/`, `frontend/src/pages/`, and `frontend/src/services/` were created as directories, but the current UI is implemented in `frontend/src/main.jsx`, `frontend/src/styles.css`, and `frontend/src/services/api.js`. This is cleanup only and does not prevent execution.

**MEDIUM — no reusable target-creation module.** Target creation exists in `notebooks/06_target_creation.ipynb`; there is no `src/targets/` or equivalent source module. A fresh run therefore depends on notebook 06 to create `silver_modeling_data.csv`, while feature and model steps have reusable source implementations.

**MEDIUM — no confidence module.** `src/prediction/confidence.py` is absent. `src/prediction/pipeline.py` handles `confidence=None` explicitly and reports limited reliability, so prediction does not crash, but calibrated confidence is not part of the current system.

## 2. Availability checks

The following artifacts were present:

| Artifact | Observed status |
| --- | --- |
| `data/SI_1m_1098d.csv` | 1,031,365 rows loaded |
| `data/silver_5m.csv` | 189,798 rows loaded |
| `data/silver_features.csv` | 189,798 rows loaded |
| `data/silver_modeling_data.csv` | 160,612 rows loaded |
| `data/SI_daily_roll-adjusted.csv` | 7,775 rows loaded |
| `saved_models/metadata/training_metadata.json` | present and readable |
| selected classifier artifact | `hist_gradient_boosting_classifier.joblib` present |
| selected regressor artifact | `linear_regression.joblib` present |
| `reports/model_comparison/evaluation_selection.json` | present and readable |

The repository ignore rules cover dataset contents, saved models, logs, and frontend build output. `git check-ignore` confirmed the CSV, model metadata, and `logs` paths are ignored.

## 3. Cross-component consistency

### Verified consistent

- The saved metadata feature order exactly matches `src.features.technical_indicators.FEATURE_COLUMNS` and the prediction validator's 17-field order.
- `prediction_horizon_minutes` is 15 and `prediction_horizon_bars` is 3 in saved metadata and prediction responses.
- `datetime_et` is parsed by the loaders as timezone-aware `America/New_York`; a loaded five-minute sample reported that timezone.
- Training and evaluation recompute the same partition boundaries as saved metadata: 112,398 training rows, 24,083 validation rows, and 24,086 test rows.
- Model selection metadata names files that exist and can be loaded.
- FastAPI OpenAPI exposes exactly `/api/health`, `/api/models`, `/api/predict`, `/api/predict/batch`, and `/api/predictions/latest`.
- Frontend scenario access uses the actual backend keys `Bullish`, `Bearish`, and `Neutral`.
- Backend request schemas require the exact 17 feature fields and reject extra fields.

### Findings

**MEDIUM — dependency declarations are split and incomplete.** `backend/requirements.txt` contains only FastAPI and Uvicorn, while `src/` and notebooks require pandas, NumPy, scikit-learn, joblib, matplotlib, seaborn, and notebook tooling. `docs/reproducibility.md` supplies a second manual install command, but there is no root requirements file or environment lock covering the complete workflow. A fresh environment installed only from the backend requirements cannot run training, evaluation, or prediction.

**MEDIUM — frontend package ranges use `latest`.** `frontend/package.json` declares React, React DOM, Vite, and the Vite plugin as `latest`. `package-lock.json` records the currently resolved versions, but a fresh `npm install` after lockfile changes or dependency refresh can drift. The current build succeeds with the lockfile.

**LOW — no explicit root entry point.** The documented commands work from the repository root, but the project has no root `pyproject.toml`, setup configuration, or root task runner. This increases setup friction but does not affect the current imports when run from the root.

## 4. Machine-learning methodology audit

### Independently verified

The future-return target was recomputed from `data/silver_features.csv` and matched `data/silver_modeling_data.csv` for all 160,612 rows with maximum absolute difference `8.88e-16`. The check required consecutive 5-minute, 10-minute, and 15-minute timestamp differences, so session-gap rows were not treated as valid horizons.

The target construction in `notebooks/06_target_creation.ipynb` is:

```text
future_return_15m = (close[t+3] - close[t]) / close[t] * 100
```

Feature names and target columns are separate. The approved feature list contains no target column. Feature calculations use rolling and shifted values from current or prior rows, with no future row used in the 17 model inputs.

Training calibrates the neutral threshold from the purged training partition. The saved threshold is `0.0776397515528126`, while the notebook's `±0.05%` label threshold is documented as provisional. The evaluator checks that recomputed split metadata matches saved metadata and ranks models using validation rows only.

Class ordering is fixed as `DOWN (-1)`, `NEUTRAL (0)`, `UP (1)`. Probability fields are mapped by model class labels and validated before scenario use. The evaluator includes all three class positions in reports and confusion matrices.

### HIGH — internal early-stopping split is not chronological

Affected file: `src/models/train.py`, `build_classification_models()` and `build_regression_models()`.  
Evidence: both `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` are created with default `early_stopping="auto"`, which uses scikit-learn's internal validation fraction when early stopping activates. The saved selected classifier is histogram gradient boosting.

This internal holdout remains inside the external training partition, so it does not use the external validation or test rows and does not create direct future-data leakage across the saved boundaries. It does, however, introduce a non-chronological internal training split into a time-series workflow. This means the reported selected-model training procedure is not strictly chronological end to end. Before a first final run, either disable internal early stopping or document and justify a time-aware internal split, then regenerate artifacts and reports.

### No critical leakage found

No direct target leakage, future feature construction, invalid session-crossing target, or test-set selection was found in the inspected implementation. The existing metrics should still be interpreted with the internal early-stopping caveat above.

## 5. Runtime and performance audit

### Verified

- Model loading is lazy and cached by `backend/services/prediction_service.py`; the API does not reload models on every request.
- Direct single prediction and batch prediction succeeded with the saved artifacts.
- Invalid feature input returned HTTP 422.
- API JSON serialization succeeded.
- Uvicorn startup completed and `/api/health` returned `status=ok` with the selected models.
- Frontend production build completed successfully.

### MEDIUM — evaluation memory footprint

`evaluate_saved_models()` loads all saved candidate models, including `random_forest_regressor.joblib` at approximately 1.5 GB and `random_forest_classifier.joblib` at approximately 716 MB. The current environment completed evaluation, but a developer machine with limited memory may page heavily or fail. The backend only loads the selected classifier and regressor, so this does not affect normal API prediction. This is a performance/readiness risk for evaluation, not an observed correctness failure.

## 6. Security and configuration

- No hardcoded secrets were found in the inspected source or documentation.
- CORS is environment-configurable through `SILVERAI_CORS_ORIGINS`, with localhost defaults.
- API input uses strict Pydantic schemas and rejects unknown fields.
- Internal model paths and exception details are not returned by normal API errors.
- Model and dataset artifacts are ignored by Git rules.
- `VITE_API_BASE_URL`, model root, selection path, and CORS origins are documented.

No CRITICAL security finding was identified. For deployment, the permissive localhost defaults should be replaced with an explicit allowed-origin list and a production reverse proxy.

## 7. Commands checked

The following lightweight or integration commands were executed successfully:

```powershell
python -m compileall -q src backend tests
pytest -q
npm run build                  # executed from frontend/
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
python -c "from backend.services.prediction_service import PredictionService; ..."
```

Observed results:

- All Python imports succeeded.
- Test suite: `21 passed`, with one non-blocking Starlette/httpx deprecation warning.
- Frontend build: passed with Vite.
- Backend startup: passed; health returned the selected classifier and regressor.
- Direct saved-model prediction: passed for single and batch inputs.

The following were **not** executed during this audit because they can overwrite or regenerate material artifacts:

- all notebooks 01–09 as a fresh run;
- `run_training()`;
- any command that rewrites saved models or evaluation reports.

## 8. Required execution order

After resolving the HIGH findings, the controlled full workflow should be:

```powershell
# 1. Current manual install (not pinned; this is one HIGH finding)
python -m pip install pandas numpy scikit-learn joblib matplotlib seaborn jupyter pytest fastapi "uvicorn[standard]"

# 2. From the repository root, run notebooks in order
# notebooks/01_data_collection.ipynb through notebooks/09_scenario_analysis.ipynb

# 3. Or use reusable modules for the implemented stages
python -c "from src.models.train import run_training; run_training()"
python -c "from src.models.evaluate import evaluate_saved_models; evaluate_saved_models()"

# 4. Start the backend
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 5. In another terminal, start the frontend
cd frontend
npm install
npm run dev
```

Target creation still requires notebook 06 or an already-created `data/silver_modeling_data.csv`; no reusable source target-creation function currently replaces that notebook step.

## 9. Prioritized fixes

1. **HIGH:** Resolve the histogram-gradient-boosting internal early-stopping split for strict time-series methodology, or explicitly document acceptance and regenerate artifacts.
2. **HIGH:** Add one complete, pinned Python dependency/environment specification covering source modules, notebooks, tests, backend, and evaluation. Keep `backend/requirements.txt` as a backend-specific subset if desired.
3. **MEDIUM:** Move target construction into a tested reusable module or clearly define notebook 06 as a mandatory data-generation prerequisite.
4. **MEDIUM:** Reduce evaluation memory pressure by loading candidate models sequentially or documenting a minimum-memory requirement.
5. **MEDIUM:** Pin frontend dependency versions instead of relying on `latest`, even though the current lockfile builds.
6. **MEDIUM:** Implement calibrated confidence only after a validation-only calibration protocol is defined; until then keep the current explicit limited-reliability behavior.
7. **LOW:** Remove empty frontend directories or adopt the planned component/page structure.

## Final status

| Area | Status |
| --- | --- |
| Repository structure | PASS with LOW/MEDIUM cleanup findings |
| Data availability | PASS |
| Feature/target consistency | PASS |
| Timestamp/session alignment | PASS |
| Saved model loading | PASS |
| Prediction/API integration | PASS |
| Frontend build | PASS |
| Automated tests | PASS: 21 passed |
| Fresh-environment reproducibility | NOT READY |
| Strict chronological model methodology | REVIEW REQUIRED |

**Final recommendation: NO-GO for an unqualified first full end-to-end execution.** The existing workspace can execute the operational path, but dependency packaging and the selected model's internal validation behavior should be resolved or formally accepted before treating a fresh run as final evidence.

"""Environment-driven backend configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.getenv("SILVERAI_MODEL_ROOT", PROJECT_ROOT / "saved_models"))
SELECTION_PATH = Path(
    os.getenv(
        "SILVERAI_SELECTION_PATH",
        PROJECT_ROOT / "reports" / "model_comparison" / "evaluation_selection.json",
    )
)
PREDICTION_DB_PATH = Path(
    os.getenv("SILVERAI_PREDICTION_DB_PATH", PROJECT_ROOT / "data" / "silverai_predictions.sqlite3")
)
FROZEN_EVALUATION_PATH = Path(
    os.getenv("SILVERAI_FROZEN_EVALUATION_PATH", PROJECT_ROOT / "reports" / "frozen_evaluation.json")
)
_cors_value = os.getenv(
    "SILVERAI_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
)
CORS_ORIGINS = [origin.strip() for origin in _cors_value.split(",") if origin.strip()]

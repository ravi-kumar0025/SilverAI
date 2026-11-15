"""Environment-driven backend configuration."""

from __future__ import annotations

import os
from pathlib import Path

from src.spot_config import SPOT_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = SPOT_CONFIG.root_model
SELECTION_PATH = SPOT_CONFIG.root_selection
PREDICTION_DB_PATH = SPOT_CONFIG.root_prediction_db
FROZEN_EVALUATION_PATH = SPOT_CONFIG.root_frozen_evaluation
_cors_value = os.getenv(
    "SILVERAI_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
)
CORS_ORIGINS = [origin.strip() for origin in _cors_value.split(",") if origin.strip()]
DATA_MODE = os.getenv("SILVERAI_DATA_MODE", "historical").strip().lower()

"""Environment-driven backend configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.getenv("SILVERAI_MODEL_ROOT", PROJECT_ROOT / "reports" / "spot_dukascopy_20261115"))
SELECTION_PATH = Path(
    os.getenv(
        "SILVERAI_SELECTION_PATH",
        PROJECT_ROOT / "reports" / "spot_dukascopy_20261115" / "metrics.json",
    )
)
PREDICTION_DB_PATH = Path(
    os.getenv("SILVERAI_PREDICTION_DB_PATH", PROJECT_ROOT / "data" / "spot_predictions.sqlite3")
)
FROZEN_EVALUATION_PATH = Path(
    os.getenv("SILVERAI_FROZEN_EVALUATION_PATH", PROJECT_ROOT / "reports" / "frozen_evaluation.json")
)
_cors_value = os.getenv(
    "SILVERAI_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
)
CORS_ORIGINS = [origin.strip() for origin in _cors_value.split(",") if origin.strip()]
DATA_MODE = os.getenv("SILVERAI_DATA_MODE", "historical").strip().lower()

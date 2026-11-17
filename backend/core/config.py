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


def _load_project_env() -> None:
    """Load simple KEY=VALUE settings without requiring an extra dependency."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_project_env()
_cors_value = os.getenv(
    "SILVERAI_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
)
CORS_ORIGINS = [origin.strip() for origin in _cors_value.split(",") if origin.strip()]
DATA_MODE = os.getenv("SILVERAI_DATA_MODE", "historical").strip().lower()

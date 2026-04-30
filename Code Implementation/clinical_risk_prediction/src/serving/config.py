"""Configuration helpers for the Asclena risk model FastAPI service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceSettings:
    model_path: Path
    service_name: str = "asclena-clinical-risk-api"
    api_version: str = "v1"
    max_batch_size: int = 128


def _latest_model_path(models_dir: Path) -> Path:
    artifacts = list(models_dir.glob("*.joblib"))
    if not artifacts:
        raise FileNotFoundError(
            f"No model artifacts found in {models_dir}. Train the model or set ASCLENA_RISK_MODEL_PATH."
        )
    return max(artifacts, key=lambda artifact: artifact.stat().st_mtime)


def get_settings() -> ServiceSettings:
    repo_root = Path(__file__).resolve().parents[2]
    models_dir = repo_root / "models"
    configured_path = os.getenv("ASCLENA_RISK_MODEL_PATH")
    model_path = Path(configured_path).expanduser() if configured_path else _latest_model_path(models_dir)
    return ServiceSettings(model_path=model_path.resolve())

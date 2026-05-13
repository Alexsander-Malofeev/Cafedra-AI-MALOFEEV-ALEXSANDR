from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "config.yaml"


def resolve_project_path(path_value: str | Path) -> Path:
    """Return an absolute path inside project/ for a relative path."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config.

    CONFIG_PATH environment variable has priority over function argument.
    """
    path_from_env = os.getenv("CONFIG_PATH")
    path = Path(path_from_env) if path_from_env else Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.is_absolute():
        path = PROJECT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML dictionary")

    return config


def get_model_path(config: dict[str, Any]) -> Path:
    """Read model path from env or config."""
    model_path = os.getenv("MODEL_PATH") or config["paths"]["model_path"]
    return resolve_project_path(model_path)

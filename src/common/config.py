from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "configs"
LOCAL_SECRETS_PATH = CONFIG_DIR / "local.secrets.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_app_config() -> dict[str, Any]:
    config = load_yaml(CONFIG_DIR / "app.yaml")
    if LOCAL_SECRETS_PATH.exists():
        config = _deep_merge(config, load_yaml(LOCAL_SECRETS_PATH))
    return config


def load_prompt_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "prompts.yaml")


def load_training_free_grpo_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "training_free_grpo.yaml")


def load_threshold_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "thresholds.yaml")

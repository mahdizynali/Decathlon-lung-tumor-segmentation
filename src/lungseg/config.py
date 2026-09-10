"""Small, explicit YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


Config = dict[str, Any]


def load_yaml(path: Path) -> Config:
    """Load a YAML mapping from *path* and reject ambiguous root values."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def merge_configs(*configs: Mapping[str, Any]) -> Config:
    """Recursively merge mappings without mutating any input mapping."""
    merged: Config = {}
    for config in configs:
        _merge_into(merged, config)
    return merged


def _merge_into(target: Config, source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge_into(current, value)
        else:
            target[key] = deepcopy(value)

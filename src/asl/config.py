"""Load pipeline configuration from TOML override files."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib

PRESETS_DIR = Path(__file__).resolve().parent / "presets"
DEFAULT_OVERRIDE_PATH = Path("asl.toml")
CONFIG_PATH_ENV = "ASL_CONFIG"

_config_path_override: Path | None = None
_config_data_override: dict[str, Any] | None = None


def set_config_path(path: Path | None) -> None:
    """Point the user override file at path (for tests)."""
    global _config_path_override
    _config_path_override = path
    load_config.cache_clear()


def set_config_data(merged: dict[str, Any]) -> None:
    """Inject assembled configuration directly (for tests)."""
    global _config_data_override
    _config_data_override = merged
    load_config.cache_clear()


def reset_config() -> None:
    """Clear cached configuration and test overrides."""
    global _config_path_override, _config_data_override
    _config_path_override = None
    _config_data_override = None
    load_config.cache_clear()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file or return an empty mapping when missing."""
    if not path.exists():
        return {}
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _resolve_override_path() -> Path:
    if _config_path_override is not None:
        return _config_path_override
    return DEFAULT_OVERRIDE_PATH


def _load_defaults() -> dict[str, Any]:
    """Load the bundled defaults from full.toml."""
    path = PRESETS_DIR / "full.toml"
    if not path.exists():
        raise FileNotFoundError(f"Defaults file not found: {path}")
    return _load_toml(path)


def _assemble_config() -> dict[str, Any]:
    """Build merged runtime config from defaults + user overrides + ASL_CONFIG."""
    defaults = _load_defaults()
    user = _load_toml(_resolve_override_path())
    merged = _deep_merge(defaults, user)
    extra_path = os.environ.get(CONFIG_PATH_ENV)
    if extra_path:
        extra = _load_toml(Path(extra_path))
        merged = _deep_merge(merged, extra)
    return merged


class Config:
    """Runtime view of pipeline configuration."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Return the effective value for section.key."""
        sec = self._data.get(section, {})
        if not isinstance(sec, dict):
            raise TypeError(f"Config section [{section}] must be a table.")
        if key in sec:
            return sec[key]
        return default


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load and cache configuration."""
    if _config_data_override is not None:
        return Config(_config_data_override)
    return Config(_assemble_config())

"""Load pipeline configuration from preset and override TOML files."""

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
_config_data_override: tuple[dict[str, Any], dict[str, Any]] | None = None


def set_config_path(path: Path | None) -> None:
    """Point the user override file at path (for tests)."""
    global _config_path_override
    _config_path_override = path
    load_config.cache_clear()


def set_config_data(
    merged: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> None:
    """Inject assembled configuration directly (for tests)."""
    global _config_data_override
    _config_data_override = (merged, overrides if overrides is not None else {})
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


def _load_preset(smoke: bool) -> dict[str, Any]:
    """Load the bundled full or smoke preset."""
    name = "smoke" if smoke else "full"
    path = PRESETS_DIR / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")
    return _load_toml(path)


def _resolve_override_path() -> Path:
    if _config_path_override is not None:
        return _config_path_override
    return DEFAULT_OVERRIDE_PATH


def _load_override_layers() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return merged override data and the user override file contents alone."""
    user = _load_toml(_resolve_override_path())
    extra_path = os.environ.get(CONFIG_PATH_ENV)
    if extra_path:
        extra = _load_toml(Path(extra_path))
        merged_overrides = _deep_merge(user, extra)
    else:
        merged_overrides = user
    return merged_overrides, user


def _assemble_config() -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Build merged runtime config from preset plus override layers."""
    overrides, _user_only = _load_override_layers()
    smoke = bool(overrides.get("run", {}).get("smoke", False))
    preset = _load_preset(smoke)
    merged = _deep_merge(preset, overrides)
    return merged, overrides, _resolve_override_path()


class Config:
    """Runtime view of preset values with user overrides applied."""

    def __init__(
        self,
        data: dict[str, Any],
        overrides: dict[str, Any],
        path: Path,
    ) -> None:
        self._data = data
        self._overrides = overrides
        self.path = path

    @property
    def smoke(self) -> bool:
        """True when [run].smoke is enabled in the override file."""
        return bool(self._section("run").get("smoke", False))

    def has(self, section: str, key: str) -> bool:
        """True when section.key is set in a user override layer."""
        override_section = self._overrides.get(section, {})
        if not isinstance(override_section, dict):
            return False
        return key in override_section

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Return the effective value after preset and override merging."""
        sec = self._section(section)
        if key in sec:
            return sec[key]
        return default

    def _section(self, name: str) -> dict[str, Any]:
        value = self._data.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Config section [{name}] must be a table.")
        return value


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load and cache configuration from preset and override files."""
    if _config_data_override is not None:
        merged, overrides = _config_data_override
        return Config(merged, overrides, Path("test"))
    merged, overrides, path = _assemble_config()
    return Config(merged, overrides, path)

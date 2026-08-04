"""Shared fixtures for ASL pipeline unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tomllib

from asl.config import (
    PRESETS_DIR,
    _deep_merge,
    reset_config,
    set_config_data,
)
from asl.spec import Model


@pytest.fixture(autouse=True)
def reset_asl_config():
    """Reset cached configuration between tests."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def repo_root() -> Path:
    """Repository root (parent of tests/)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def config_file(repo_root):
    """Apply override fragments on top of bundled defaults."""

    def _write(content: str) -> Path:
        with open(PRESETS_DIR / "full.toml", "rb") as handle:
            defaults = tomllib.load(handle)
        if content.strip():
            overrides = tomllib.loads(content)
            merged = _deep_merge(defaults, overrides)
        else:
            merged = defaults
        set_config_data(merged)
        return repo_root / "asl.toml"

    return _write


@pytest.fixture
def toy_model() -> Model:
    """Minimal registered model for unit tests."""

    def simulate(params: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        acc = float(np.clip(0.5 + 0.1 * params[0], 0.01, 0.99))
        rt_mean = float(0.3 + 0.05 * params[1])
        rt_var = float(0.01 + 0.001 * abs(params[0]))
        return np.array([acc, rt_mean, rt_var], dtype=np.float64)

    model = Model(
        slug="toy",
        param_names=("v", "a"),
        param_bounds=((-1.0, 1.0), (0.5, 2.0)),
        prior_bounds=((-0.8, 0.8), (0.6, 1.8)),
        summary_names=("acc", "rt_mean", "rt_var"),
        summary_transforms=("identity", "log1p", "log1p"),
        simulate_summaries=simulate,
        recovery_priors={"v": "v ~ dunif(-1, 1)", "a": "a ~ dunif(0.5, 2)"},
        default_architecture="DeepWide_24x4",
    )
    return model

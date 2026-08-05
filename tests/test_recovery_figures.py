"""Unit tests for recovery figure helpers and subjects JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from asl.figures import canonical_recovery_sort_indices, plot_recovery_diagnostics
from asl.recovery import write_recovery_subjects
from asl.spec import Model


def test_canonical_recovery_sort_indices_ddm4_order():
    names = ["w", "v", "a", "t0"]
    order = canonical_recovery_sort_indices(names)
    assert [names[i] for i in order] == ["v", "a", "t0", "w"]


def test_write_recovery_subjects_payload_shape(toy_model, tmp_path):
    n = 5
    true = np.random.default_rng(0).uniform(-1, 1, size=(n, toy_model.n_params))
    est = true + 0.1
    ci_lo = est - 0.2
    ci_hi = est + 0.2
    rhats = np.full((n, toy_model.n_params), 1.01)

    path = write_recovery_subjects(
        toy_model, true, est, ci_lo, ci_hi, rhats, tmp_path
    )
    data = json.loads(path.read_text())
    assert data["param_names"] == list(toy_model.param_names)
    assert data["param_bounds"] == [list(b) for b in toy_model.prior_bounds]
    assert len(data["true"]) == n
    assert len(data["est"][0]) == toy_model.n_params
    assert set(data) == {
        "param_names",
        "param_bounds",
        "true",
        "est",
        "ci_lo",
        "ci_hi",
        "rhat",
    }


def test_plot_recovery_diagnostics_writes_pdf(toy_model, tmp_path):
    rng = np.random.default_rng(1)
    n = 12
    true = rng.uniform(-1, 1, size=(n, toy_model.n_params))
    est = true + rng.normal(0, 0.05, size=true.shape)
    ci_lo = est - 0.1
    ci_hi = est + 0.1
    out = tmp_path / "recovery.pdf"
    plot_recovery_diagnostics(toy_model, true, est, ci_lo, ci_hi, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_recovery_diagnostics_four_param_grid(tmp_path):
    def simulate(params: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
        return np.ones(4, dtype=np.float64)

    model = Model(
        slug="ddm4test",
        param_names=("v", "a", "t0", "w"),
        param_bounds=((-1, 1),) * 4,
        prior_bounds=((-1, 1),) * 4,
        summary_names=("s1", "s2", "s3", "s4"),
        summary_transforms=("identity",) * 4,
        simulate_summaries=simulate,
    )
    rng = np.random.default_rng(2)
    n = 20
    true = rng.uniform(-1, 1, size=(n, 4))
    est = true + rng.normal(0, 0.05, size=true.shape)
    ci_lo = est - 0.1
    ci_hi = est + 0.1
    out = tmp_path / "recovery.pdf"
    plot_recovery_diagnostics(model, true, est, ci_lo, ci_hi, out)
    assert out.exists()

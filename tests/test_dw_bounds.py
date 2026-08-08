"""Tests for DW training vs recovery canonical intervals."""

from __future__ import annotations

from models.social import dw_bounds
from models.social.dw import DW


def test_dw_training_bounds():
    assert DW.param_bounds == dw_bounds.DW_TRAINING_BOUNDS
    assert DW.param_bounds == (
        dw_bounds.TRAINING_EPSILON_BOUNDS,
        dw_bounds.TRAINING_MU_BOUNDS,
    )


def test_dw_prior_bounds():
    assert DW.prior_bounds == dw_bounds.DW_PRIOR_BOUNDS
    assert DW.prior_bounds == (
        dw_bounds.PRIOR_EPSILON_BOUNDS,
        dw_bounds.PRIOR_MU_BOUNDS,
    )


def test_dw_recovery_priors_match_prior_bounds():
    for name, (lo, hi) in zip(DW.param_names, DW.prior_bounds):
        assert f"dunif({lo:g}, {hi:g})" in DW.recovery_priors[name]

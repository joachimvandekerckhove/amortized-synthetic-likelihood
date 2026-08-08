"""Tests for DW training vs recovery logit intervals."""

from __future__ import annotations

from models.social import dw_bounds
from models.social.dw import DW


def test_dw_training_bounds():
    assert DW.param_bounds == dw_bounds.DW_LOGIT_TRAINING_BOUNDS
    assert DW.param_bounds == (
        dw_bounds.LOGIT_TRAINING_EPSILON_BOUNDS,
        dw_bounds.LOGIT_TRAINING_MU_BOUNDS,
    )


def test_dw_prior_bounds():
    assert DW.prior_bounds == dw_bounds.DW_LOGIT_PRIOR_BOUNDS
    assert DW.prior_bounds == (
        dw_bounds.LOGIT_PRIOR_EPSILON_BOUNDS,
        dw_bounds.LOGIT_PRIOR_MU_BOUNDS,
    )


def test_dw_recovery_priors_match_prior_bounds():
    for name, (lo, hi) in zip(DW.param_names, DW.prior_bounds):
        assert f"dunif({lo:g}, {hi:g})" in DW.recovery_priors[name]


def test_dw_reporting_bounds_are_canonical():
    assert DW.report_prior_bounds == dw_bounds.DW_PRIOR_BOUNDS
    assert DW.report_param_names == dw_bounds.CANONICAL_PARAM_NAMES

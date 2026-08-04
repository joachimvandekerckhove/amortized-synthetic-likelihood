"""Tests for DDM wide-bounds intervals and priors."""

from __future__ import annotations

from models.ddm import bounds
from models.ddm.ddm3 import DDM3
from models.ddm.ddm4 import DDM4
from models.ddm.ddmcollapsesig import DDMCOLLAPSESIG


def test_truncated_normal_prior_format():
    prior = bounds.truncated_normal_prior("v", -3.0, 3.0)
    assert "dnorm(0," in prior
    assert "T(-3, 3)" in prior


def test_ddm3_bounds():
    assert DDM3.param_bounds == bounds.DDM3_TRAINING_BOUNDS
    assert DDM3.prior_bounds == bounds.DDM3_PRIOR_BOUNDS
    assert DDM3.param_bounds == (
        bounds.V_TRAINING,
        bounds.A_TRAINING,
        bounds.T0_TRAINING,
    )
    assert DDM3.prior_bounds == (
        bounds.V_PRIOR,
        bounds.A_PRIOR,
        bounds.T0_PRIOR,
    )
    assert DDM3.summary_transforms == ("identity", "log1p", "log1p")


def test_ddm4_bounds():
    assert DDM4.param_bounds == bounds.DDM4_TRAINING_BOUNDS
    assert DDM4.prior_bounds == bounds.DDM4_PRIOR_BOUNDS
    assert DDM4.param_bounds[-1] == bounds.W_TRAINING
    assert DDM4.prior_bounds[-1] == bounds.W_PRIOR
    assert DDM4.summary_transforms == (
        "log1p",
        "log1p",
        "log1p",
        "log1p",
        "identity",
    )


def test_ddmcollapsesig_bounds():
    assert DDMCOLLAPSESIG.param_bounds == bounds.DDMCOLLAPSESIG_TRAINING_BOUNDS
    assert DDMCOLLAPSESIG.prior_bounds == bounds.DDMCOLLAPSESIG_PRIOR_BOUNDS
    assert DDMCOLLAPSESIG.param_bounds == (
        bounds.A_TRAINING,
        bounds.V_TRAINING,
        bounds.K_TRAINING,
        bounds.T0_TRAINING,
    )
    assert DDMCOLLAPSESIG.prior_bounds == (
        bounds.A_PRIOR,
        bounds.V_PRIOR,
        bounds.K_PRIOR,
        bounds.T0_PRIOR,
    )
    assert DDMCOLLAPSESIG.summary_transforms == (
        "identity",
        "log1p",
        "log1p",
        "log1p",
    )


def test_recovery_priors_are_truncated_normal():
    for model in (DDM3, DDM4, DDMCOLLAPSESIG):
        for name, line in model.recovery_priors.items():
            assert "dnorm(" in line, name
            assert " T(" in line, name

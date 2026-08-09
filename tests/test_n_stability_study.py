"""Unit tests for asl.n_stability_study."""

from __future__ import annotations

import numpy as np
import pytest

import asl.n_stability_study as n_stability_study
from asl.n_stability_study import (
    DW_PROFILE,
    SUPPORTED_SLUGS,
    build_config,
    draw_fixed_thetas,
    get_slug_profile,
)
from models.catalog import get_model
from models.social.dw import DW_STUDY_CANONICAL_THETAS


class TestNStabilityProfiles:
    def test_dw_is_supported(self):
        assert "dw" in SUPPORTED_SLUGS

    def test_dw_profile_settings(self):
        profile = get_slug_profile("dw")
        assert profile.ref_n == 150
        assert profile.null_key == "150_null"
        assert profile.n_values == (50, 100, 150, 300, 600)
        assert profile.theta_mode == "fixed_dw"

    def test_dw_draws_fixed_thetas(self):
        model = get_model("dw")
        params = draw_fixed_thetas(model, 5, seed=1, profile=DW_PROFILE)
        assert params.shape == (5, 2)
        again = draw_fixed_thetas(model, 5, seed=99, profile=DW_PROFILE)
        assert (params == again).all()

    def test_dw_profile_avoids_degenerate_low_interaction_point(self):
        assert DW_STUDY_CANONICAL_THETAS[0] == (0.25, 0.35)

    def test_dw_build_config_defaults(self, monkeypatch):
        class Args:
            slug = "dw"
            quick = False
            n_theta = None
            n_replicates = None
            seed = 1
            workers = 1
            results_dir = None

        config = build_config(Args())
        assert config.n_theta == DW_PROFILE.default_n_theta
        assert config.profile.ref_n == 150


class TestCovarianceProfileCoverage:
    def test_rejects_incomplete_covariance_profile(self):
        batches = {"150": {"ok": np.array([True, False, True])}}

        with pytest.raises(RuntimeError, match="150"):
            n_stability_study.check_complete_covariance_profile(batches)

    def test_identifies_every_profile_point_invalid_in_any_batch(self):
        batches = {
            "50": {"ok": np.array([True, False, True, True])},
            "100": {"ok": np.array([True, True, False, True])},
        }

        invalid = n_stability_study.invalid_profile_indices(batches)

        np.testing.assert_array_equal(invalid, [1, 2])

"""Unit tests for the Deffuant-Weisbuch bounded-confidence model."""

from __future__ import annotations

import numpy as np

from models.social.dw import (
    DW,
    N_SUMMARIES,
    PARAM_NAMES,
    PRIOR_EPSILON_BOUNDS,
    PRIOR_MU_BOUNDS,
    SUMMARY_NAMES,
    TRAINING_EPSILON_BOUNDS,
    TRAINING_MU_BOUNDS,
    draw_cov_parameters,
    simulate_summaries,
    to_canonical,
)


INTERIOR_PARAMS = np.array([0.22, 0.20], dtype=np.float64)


class TestSimulator:
    def test_output_shape_and_finiteness(self):
        result = simulate_summaries(INTERIOR_PARAMS, n_trials=8000, seed=11)
        assert result.shape == (N_SUMMARIES,)
        assert np.all(np.isfinite(result))

    def test_seed_determinism(self):
        a = simulate_summaries(INTERIOR_PARAMS, n_trials=8000, seed=17)
        b = simulate_summaries(INTERIOR_PARAMS, n_trials=8000, seed=17)
        assert np.allclose(a, b)

    def test_epsilon_decreases_effective_clusters(self):
        low_eps = simulate_summaries(
            np.array([TRAINING_EPSILON_BOUNDS[0] + 0.02, 0.20]),
            n_trials=12000,
            seed=23,
        )
        high_eps = simulate_summaries(
            np.array([TRAINING_EPSILON_BOUNDS[1] - 0.02, 0.20]),
            n_trials=12000,
            seed=23,
        )
        assert np.all(np.isfinite(low_eps))
        assert np.all(np.isfinite(high_eps))
        assert low_eps[0] > high_eps[0] + 0.15

    def test_mu_increases_large_move_fraction(self):
        low_mu = simulate_summaries(
            np.array([0.22, TRAINING_MU_BOUNDS[0] + 0.02]),
            n_trials=12000,
            seed=31,
        )
        high_mu = simulate_summaries(
            np.array([0.22, TRAINING_MU_BOUNDS[1] - 0.02]),
            n_trials=12000,
            seed=31,
        )
        assert np.all(np.isfinite(low_mu))
        assert np.all(np.isfinite(high_mu))
        assert high_mu[5] > low_mu[5]

    def test_mu_affects_temporal_summaries(self):
        low_mu = simulate_summaries(
            np.array([0.22, TRAINING_MU_BOUNDS[0]]),
            n_trials=12000,
            seed=37,
        )
        high_mu = simulate_summaries(
            np.array([0.22, TRAINING_MU_BOUNDS[1] - 0.02]),
            n_trials=12000,
            seed=37,
        )
        assert np.all(np.isfinite(low_mu))
        assert np.all(np.isfinite(high_mu))
        assert not np.allclose(low_mu, high_mu)

    def test_degenerate_returns_nan(self):
        result = simulate_summaries(INTERIOR_PARAMS, n_trials=50, seed=99)
        assert np.all(np.isnan(result))


class TestCanonicalTransform:
    def test_to_canonical_is_identity(self):
        params = np.array([0.22, 0.20])
        epsilon, mu = to_canonical(params)
        assert abs(epsilon - 0.22) < 1e-12
        assert abs(mu - 0.20) < 1e-12


class TestModelSpec:
    def test_parameter_layout(self):
        assert DW.param_names == PARAM_NAMES
        assert DW.summary_names == SUMMARY_NAMES
        assert DW.n_outputs == 27
        assert DW.supports_recovery()
        assert DW.slug == "dw"
        assert DW.default_architecture == "DeepWide_32x6"
        assert DW.param_bounds == (TRAINING_EPSILON_BOUNDS, TRAINING_MU_BOUNDS)
        assert DW.prior_bounds == (PRIOR_EPSILON_BOUNDS, PRIOR_MU_BOUNDS)
        assert DW.summary_transforms == (
            "log1p",
            "log1p",
            "log1p",
            "log1p",
            "log1p",
            "identity",
        )

    def test_draw_cov_parameters_within_training_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            params = draw_cov_parameters(rng)
            assert TRAINING_EPSILON_BOUNDS[0] <= params[0] <= TRAINING_EPSILON_BOUNDS[1]
            assert TRAINING_MU_BOUNDS[0] <= params[1] <= TRAINING_MU_BOUNDS[1]

    def test_recovery_priors_match_inference_bounds(self):
        assert "0.15" in DW.recovery_priors["epsilon"]
        assert "0.35" in DW.recovery_priors["epsilon"]
        assert "0.1" in DW.recovery_priors["mu"]
        assert "0.4" in DW.recovery_priors["mu"]

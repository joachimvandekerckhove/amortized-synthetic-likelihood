"""Unit tests for the Deffuant-Weisbuch bounded-confidence model."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from models.social.dw import (
    DW,
    N_SUMMARIES,
    PARAM_NAMES,
    PRIOR_EPSILON_BOUNDS,
    PRIOR_MU_BOUNDS,
    SUMMARY_NAMES,
    TRAINING_EPSILON_BOUNDS,
    TRAINING_MU_BOUNDS,
    _run_interactions,
    canonical_params_array,
    draw_cov_parameters,
    simulate_summaries,
    to_canonical,
)
from models.social.dw_bounds import (
    LOGIT_PRIOR_EPSILON_BOUNDS,
    LOGIT_PRIOR_MU_BOUNDS,
    LOGIT_TRAINING_EPSILON_BOUNDS,
    LOGIT_TRAINING_MU_BOUNDS,
)

INTERIOR_CANONICAL = np.array([0.22, 0.20], dtype=np.float64)
INTERIOR_LOGIT = np.array([logit(0.22), logit(0.20)], dtype=np.float64)


class TestSimulator:
    def test_output_shape_and_finiteness(self):
        result = simulate_summaries(INTERIOR_LOGIT, n_trials=8000, seed=11)
        assert result.shape == (N_SUMMARIES,)
        assert np.all(np.isfinite(result))

    def test_seed_determinism(self):
        a = simulate_summaries(INTERIOR_LOGIT, n_trials=8000, seed=17)
        b = simulate_summaries(INTERIOR_LOGIT, n_trials=8000, seed=17)
        assert np.allclose(a, b)

    def test_epsilon_decreases_effective_clusters(self):
        low_eps = simulate_summaries(
            np.array([logit(TRAINING_EPSILON_BOUNDS[0] + 0.02), INTERIOR_LOGIT[1]]),
            n_trials=12000,
            seed=23,
        )
        high_eps = simulate_summaries(
            np.array([logit(TRAINING_EPSILON_BOUNDS[1] - 0.02), INTERIOR_LOGIT[1]]),
            n_trials=12000,
            seed=23,
        )
        assert np.all(np.isfinite(low_eps))
        assert np.all(np.isfinite(high_eps))
        assert low_eps[0] > high_eps[0] + 0.15

    def test_mu_increases_large_move_fraction(self):
        low_mu = simulate_summaries(
            np.array([INTERIOR_LOGIT[0], logit(TRAINING_MU_BOUNDS[0] + 0.02)]),
            n_trials=12000,
            seed=31,
        )
        high_mu = simulate_summaries(
            np.array([INTERIOR_LOGIT[0], logit(TRAINING_MU_BOUNDS[1] - 0.02)]),
            n_trials=12000,
            seed=31,
        )
        assert np.all(np.isfinite(low_mu))
        assert np.all(np.isfinite(high_mu))
        assert high_mu[5] > low_mu[5]

    def test_mu_affects_temporal_summaries(self):
        low_mu = simulate_summaries(
            np.array([INTERIOR_LOGIT[0], logit(TRAINING_MU_BOUNDS[0])]),
            n_trials=12000,
            seed=37,
        )
        high_mu = simulate_summaries(
            np.array([INTERIOR_LOGIT[0], logit(TRAINING_MU_BOUNDS[1] - 0.02)]),
            n_trials=12000,
            seed=37,
        )
        assert np.all(np.isfinite(low_mu))
        assert np.all(np.isfinite(high_mu))
        assert not np.allclose(low_mu, high_mu)

    def test_pair_update_is_simultaneous(self):
        opinions = np.array([0.2, 0.8])
        mu = 0.3
        rng = np.random.default_rng(0)
        _run_interactions(opinions, epsilon=1.0, mu=mu, n_events=1, rng=rng)
        assert opinions[0] == pytest.approx(0.2 + mu * 0.6)
        assert opinions[1] == pytest.approx(0.8 - mu * 0.6)

    def test_pair_mean_conserved(self):
        opinions = np.array([0.2, 0.8])
        before = opinions.sum()
        mu = 0.3
        rng = np.random.default_rng(0)
        _run_interactions(opinions, epsilon=1.0, mu=mu, n_events=1, rng=rng)
        assert opinions.sum() == pytest.approx(before)


class TestCanonicalTransform:
    def test_to_canonical_applies_sigmoid(self):
        epsilon, mu = to_canonical(INTERIOR_LOGIT)
        assert epsilon == pytest.approx(0.22)
        assert mu == pytest.approx(0.20)

    def test_canonical_params_array_matches_expit(self):
        batch = canonical_params_array(INTERIOR_LOGIT.reshape(1, -1))
        np.testing.assert_allclose(batch[0], INTERIOR_CANONICAL)


class TestModelSpec:
    def test_parameter_layout(self):
        assert DW.param_names == PARAM_NAMES
        assert DW.summary_names == SUMMARY_NAMES
        assert DW.n_outputs == 27
        assert DW.supports_recovery()
        assert DW.slug == "dw"
        assert DW.default_architecture == "DeepWide_32x6"
        assert DW.param_bounds == (LOGIT_TRAINING_EPSILON_BOUNDS, LOGIT_TRAINING_MU_BOUNDS)
        assert DW.prior_bounds == (LOGIT_PRIOR_EPSILON_BOUNDS, LOGIT_PRIOR_MU_BOUNDS)
        assert DW.report_param_names == ("epsilon", "mu")
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
            assert LOGIT_TRAINING_EPSILON_BOUNDS[0] <= params[0] <= LOGIT_TRAINING_EPSILON_BOUNDS[1]
            assert LOGIT_TRAINING_MU_BOUNDS[0] <= params[1] <= LOGIT_TRAINING_MU_BOUNDS[1]
            epsilon, mu = to_canonical(params)
            assert TRAINING_EPSILON_BOUNDS[0] <= epsilon <= TRAINING_EPSILON_BOUNDS[1]
            assert TRAINING_MU_BOUNDS[0] <= mu <= TRAINING_MU_BOUNDS[1]

    def test_recovery_priors_match_logit_inference_bounds(self):
        assert f"{LOGIT_PRIOR_EPSILON_BOUNDS[0]:g}" in DW.recovery_priors["logit_epsilon"]
        assert f"{LOGIT_PRIOR_EPSILON_BOUNDS[1]:g}" in DW.recovery_priors["logit_epsilon"]
        assert f"{LOGIT_PRIOR_MU_BOUNDS[0]:g}" in DW.recovery_priors["logit_mu"]
        assert f"{LOGIT_PRIOR_MU_BOUNDS[1]:g}" in DW.recovery_priors["logit_mu"]

    def test_report_params_fn_is_sigmoid(self):
        mapped = DW.report_params_fn(np.array([[0.0, 0.0]]))
        np.testing.assert_allclose(mapped, [[0.5, 0.5]])

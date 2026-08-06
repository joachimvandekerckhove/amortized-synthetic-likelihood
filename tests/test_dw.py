"""Unit tests for the Deffuant-Weisbuch bounded-confidence model."""

from __future__ import annotations

import numpy as np

from models.social.dw import (
    CANONICAL_MU_MAX,
    CANONICAL_MU_MIN,
    DW,
    EPSILON_MAX,
    EPSILON_MIN,
    N_SUMMARIES,
    PARAM_NAMES,
    SUMMARY_NAMES,
    _to_epsilon_mu,
    simulate_summaries,
    to_canonical,
)


def _logit_from_unit(u: float) -> float:
    u = float(np.clip(u, 1e-6, 1.0 - 1e-6))
    return float(np.log(u / (1.0 - u)))


def _canonical_params(epsilon: float, mu: float) -> np.ndarray:
    frac_eps = (epsilon - EPSILON_MIN) / (EPSILON_MAX - EPSILON_MIN)
    frac_mu = (mu - CANONICAL_MU_MIN) / (CANONICAL_MU_MAX - CANONICAL_MU_MIN)
    return np.array(
        [_logit_from_unit(frac_eps), _logit_from_unit(frac_mu)],
        dtype=np.float64,
    )


INTERIOR_PARAMS = _canonical_params(0.22, 0.20)


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
            _canonical_params(EPSILON_MIN + 0.02, 0.20),
            n_trials=12000,
            seed=23,
        )
        high_eps = simulate_summaries(
            _canonical_params(EPSILON_MAX - 0.02, 0.20),
            n_trials=12000,
            seed=23,
        )
        assert np.all(np.isfinite(low_eps))
        assert np.all(np.isfinite(high_eps))
        assert low_eps[0] > high_eps[0] + 0.15

    def test_mu_increases_large_move_fraction(self):
        low_mu = simulate_summaries(
            _canonical_params(0.22, CANONICAL_MU_MIN + 0.02),
            n_trials=12000,
            seed=31,
        )
        high_mu = simulate_summaries(
            _canonical_params(0.22, CANONICAL_MU_MAX - 0.02),
            n_trials=12000,
            seed=31,
        )
        assert np.all(np.isfinite(low_mu))
        assert np.all(np.isfinite(high_mu))
        assert high_mu[5] > low_mu[5]

    def test_mu_affects_temporal_summaries(self):
        low_mu = simulate_summaries(
            _canonical_params(0.22, CANONICAL_MU_MIN),
            n_trials=12000,
            seed=37,
        )
        high_mu = simulate_summaries(
            _canonical_params(0.22, CANONICAL_MU_MAX - 0.02),
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
    def test_to_canonical_round_trip(self):
        params = _canonical_params(0.22, 0.20)
        epsilon, mu = to_canonical(params)
        assert abs(epsilon - 0.22) < 1e-6
        assert abs(mu - 0.20) < 1e-6

    def test_to_epsilon_mu_matches_sigmoid_bounds(self):
        epsilon, mu = _to_epsilon_mu(0.0, 0.0)
        assert EPSILON_MIN < epsilon < EPSILON_MAX
        assert CANONICAL_MU_MIN < mu < CANONICAL_MU_MAX


class TestModelSpec:
    def test_parameter_layout(self):
        assert DW.param_names == PARAM_NAMES
        assert DW.summary_names == SUMMARY_NAMES
        assert DW.n_outputs == 27
        assert DW.supports_recovery()
        assert DW.slug == "dw"
        assert DW.default_architecture == "DeepWide_128x6"

"""Unit tests for the ddmcollapsesig simulator."""

from __future__ import annotations

import numpy as np

from models.ddm.ddmcollapsesig import (
    PARAM_NAMES,
    SUMMARY_NAMES,
    collapse_bound,
    simulate_summaries,
)
from models.ddm.ddmcollapsesigmv import DDMCOLLAPSESIGMV


class TestCollapseBound:
    def test_starts_at_half_bound_and_shrinks(self):
        times = np.array([0.0, 0.2, 0.8])
        bounds = collapse_bound(times, a0=1.4, k=4.0)
        assert np.isclose(bounds[0], 0.7)
        assert bounds[0] > bounds[1] > bounds[2] > 0.0

    def test_k_zero_gives_constant_half_bound(self):
        times = np.array([0.0, 0.5, 1.5])
        bounds = collapse_bound(times, a0=1.4, k=0.0)
        assert np.allclose(bounds, 0.7)


class TestSimulator:
    def test_responds_to_collapse_rate(self):
        params_low_k = np.array([1.2, 0.2, 0.1, 0.2])
        params_high_k = np.array([1.2, 0.2, 8.0, 0.2])
        low = simulate_summaries(params_low_k, n_trials=2000, seed=23)
        high = simulate_summaries(params_high_k, n_trials=2000, seed=23)
        assert np.all(np.isfinite(low))
        assert np.all(np.isfinite(high))
        assert high[3] < low[3]

    def test_k_zero_produces_finite_summaries(self):
        params = np.array([1.2, 0.5, 0.0, 0.2])
        result = simulate_summaries(params, n_trials=400, seed=7)
        assert np.all(np.isfinite(result))
        assert len(result) == len(SUMMARY_NAMES)


class TestModelSpec:
    def test_parameter_layout(self):
        assert DDMCOLLAPSESIGMV.param_names == PARAM_NAMES
        assert DDMCOLLAPSESIGMV.summary_names == SUMMARY_NAMES
        assert DDMCOLLAPSESIGMV.n_outputs == 65
        assert DDMCOLLAPSESIGMV.supports_mv_recovery()
        assert DDMCOLLAPSESIGMV.slug == "ddmcollapsesig"

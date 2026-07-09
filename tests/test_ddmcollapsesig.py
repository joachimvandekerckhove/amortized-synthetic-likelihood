"""Unit tests for the ddmcollapsesig simulator."""

from __future__ import annotations

import numpy as np

from models.ddm.ddmcollapsesig import (
    PARAM_NAMES,
    SUMMARY_NAMES,
    collapse_bound,
    simulate_summaries_collapse,
    simulate_summaries_fixed,
)
from models.ddm.ddmcollapsesig_collapsemv import DDMCOLLAPSESIG_COLLAPSEMV
from models.ddm.ddmcollapsesig_fixedmv import DDMCOLLAPSESIG_FIXEDMV


class TestCollapseBound:
    def test_starts_at_half_bound_and_shrinks(self):
        times = np.array([0.0, 0.2, 0.8])
        bounds = collapse_bound(times, a0=1.4, k=4.0)
        assert np.isclose(bounds[0], 0.7)
        assert bounds[0] > bounds[1] > bounds[2] > 0.0


class TestSimulators:
    def test_fixed_condition_ignores_collapse_rate(self):
        params_low_k = np.array([1.2, 1.5, 0.1, 0.2])
        params_high_k = np.array([1.2, 1.5, 8.0, 0.2])
        low = simulate_summaries_fixed(params_low_k, n_trials=400, seed=11)
        high = simulate_summaries_fixed(params_high_k, n_trials=400, seed=11)
        assert np.all(np.isfinite(low))
        assert np.allclose(low, high)

    def test_collapse_condition_responds_to_collapse_rate(self):
        params_low_k = np.array([1.2, 0.2, 0.1, 0.2])
        params_high_k = np.array([1.2, 0.2, 8.0, 0.2])
        low = simulate_summaries_collapse(params_low_k, n_trials=2000, seed=23)
        high = simulate_summaries_collapse(params_high_k, n_trials=2000, seed=23)
        assert np.all(np.isfinite(low))
        assert np.all(np.isfinite(high))
        assert high[3] < low[3]


class TestModelSpecs:
    def test_shared_parameter_layout(self):
        assert DDMCOLLAPSESIG_FIXEDMV.param_names == PARAM_NAMES
        assert DDMCOLLAPSESIG_COLLAPSEMV.param_names == PARAM_NAMES
        assert DDMCOLLAPSESIG_FIXEDMV.summary_names == SUMMARY_NAMES
        assert DDMCOLLAPSESIG_COLLAPSEMV.n_outputs == 65
        assert DDMCOLLAPSESIG_FIXEDMV.supports_mv_recovery()

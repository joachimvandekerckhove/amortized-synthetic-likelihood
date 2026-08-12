"""Unit tests for the ddm4a simulator and model registration."""

from __future__ import annotations

import numpy as np

from models.ddm.ddm4a import DDM4A, PARAM_NAMES, SUMMARY_NAMES, simulate_summaries


# Interior point with moderate |v| so both choices occur often.
INTERIOR = np.array([0.4, 1.2, 0.25, 0.5], dtype=np.float64)


class TestSimulator:
    def test_summary_length_and_finite(self):
        result = simulate_summaries(INTERIOR, n_trials=800, seed=11)
        assert result.shape == (len(SUMMARY_NAMES),)
        assert np.all(np.isfinite(result))

    def test_seed_determinism(self):
        a = simulate_summaries(INTERIOR, n_trials=600, seed=17)
        b = simulate_summaries(INTERIOR, n_trials=600, seed=17)
        assert np.allclose(a, b)

    def test_q10_at_most_pooled_mean(self):
        result = simulate_summaries(INTERIOR, n_trials=1000, seed=23)
        rt_mean_corr, rt_mean_err = result[0], result[2]
        err_rate = result[4]
        rt_q10 = result[5]
        pooled_mean = (1.0 - err_rate) * rt_mean_corr + err_rate * rt_mean_err
        assert rt_q10 <= pooled_mean

    def test_too_few_trials_returns_nan(self):
        result = simulate_summaries(INTERIOR, n_trials=1, seed=3)
        assert result.shape == (len(SUMMARY_NAMES),)
        assert np.all(np.isnan(result))


class TestModelSpec:
    def test_parameter_and_summary_layout(self):
        assert DDM4A.slug == "ddm4a"
        assert DDM4A.param_names == PARAM_NAMES
        assert DDM4A.summary_names == SUMMARY_NAMES
        assert DDM4A.summary_transforms == (
            "log1p",
            "log1p",
            "log1p",
            "log1p",
            "identity",
            "log1p",
        )
        assert DDM4A.n_summaries == 6
        assert DDM4A.n_outputs == 27
        assert DDM4A.supports_recovery()
        assert DDM4A.default_architecture == "DeepWide_32x6"

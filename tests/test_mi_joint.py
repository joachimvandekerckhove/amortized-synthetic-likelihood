"""Unit tests for asl.mi_joint."""

from __future__ import annotations

import numpy as np
import pytest

from asl.mi_joint import joint_mi_ksg


class TestJointMiKsg:
    def test_independent_variables_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2000, 1))
        y = rng.normal(size=(2000, 3))
        assert joint_mi_ksg(x, y, k=5) < 0.05

    def test_dependent_variables_positive(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2000, 1))
        y = x + 0.01 * rng.normal(size=(2000, 3))
        assert joint_mi_ksg(x, y, k=5) > 0.5

    def test_is_invariant_to_summary_units(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=3000)
        summaries = np.column_stack(
            [x + 0.2 * rng.normal(size=3000), rng.normal(size=3000)]
        )
        rescaled = summaries.copy()
        rescaled[:, 1] *= 1_000_000

        baseline = joint_mi_ksg(x, summaries)
        rescaled_mi = joint_mi_ksg(x, rescaled)

        assert baseline == pytest.approx(rescaled_mi, abs=0.02)

    def test_joint_exceeds_marginal_for_spread_information(self):
        rng = np.random.default_rng(2)
        theta = rng.uniform(-1, 1, size=2000)
        s1 = rng.normal(size=2000)
        s2 = rng.normal(size=2000)
        summaries = np.column_stack([s1, s2, theta * s1 * s2])
        marginal = joint_mi_ksg(theta, summaries[:, [2]], k=5)
        joint = joint_mi_ksg(theta, summaries, k=5)
        assert joint > marginal + 0.1

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError):
            joint_mi_ksg(np.arange(3.0), np.arange(3.0), k=5)

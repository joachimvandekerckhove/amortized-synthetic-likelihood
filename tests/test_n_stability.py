"""Unit tests for asl.n_stability."""

from __future__ import annotations

import numpy as np
import pytest

from asl.n_stability import (
    correlation_differences,
    diagonal_ratios,
    estimate_c1,
    generalized_eigenvalues,
    is_positive_definite,
    omega_from_chol_upper,
    relative_frobenius_error,
    stein_discrepancy,
)


class TestEstimateC1:
    def test_gaussian_means_recover_invariant_c1(self):
        rng = np.random.default_rng(0)
        true_c1 = np.array([[0.21, 0.02], [0.02, 0.04]])
        n_trials = 100
        n_replicates = 20_000
        samp_cov = true_c1 / n_trials
        summaries = rng.multivariate_normal(
            mean=np.zeros(2), cov=samp_cov, size=n_replicates
        )
        c1_hat = estimate_c1(summaries, n_trials)
        np.testing.assert_allclose(c1_hat, true_c1, rtol=0.05, atol=0.01)

    def test_rejects_too_few_rows(self):
        with pytest.raises(ValueError):
            estimate_c1(np.ones((1, 3)), 50)


class TestMatrixComparisons:
    def test_identical_matrices(self):
        c = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 0.8]])
        assert relative_frobenius_error(c, c) == pytest.approx(0.0)
        assert stein_discrepancy(c, c) == pytest.approx(0.0, abs=1e-12)
        eigs = generalized_eigenvalues(c, c)
        np.testing.assert_allclose(eigs, np.ones(3), atol=1e-12)
        np.testing.assert_allclose(diagonal_ratios(c, c), np.ones(3))
        np.testing.assert_allclose(correlation_differences(c, c), 0.0, atol=1e-12)

    def test_known_scalar_inflation(self):
        c_ref = np.diag([1.0, 2.0, 3.0])
        scale = 1.5
        c_n = scale * c_ref
        np.testing.assert_allclose(diagonal_ratios(c_n, c_ref), np.full(3, scale))
        eigs = generalized_eigenvalues(c_n, c_ref)
        np.testing.assert_allclose(eigs, np.full(3, scale), atol=1e-12)
        expected = 3.0 * scale - 3.0 * np.log(scale) - 3.0
        assert stein_discrepancy(c_n, c_ref) == pytest.approx(expected)


class TestPositiveDefinite:
    def test_pd_true(self):
        assert is_positive_definite(np.eye(3))

    def test_psd_not_pd(self):
        matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
        assert not is_positive_definite(matrix)


class TestOmegaFromChol:
    def test_uses_exported_chol_layout(self):
        chol_upper = np.array([2.0, 1.0, 3.0])
        omega = omega_from_chol_upper(chol_upper, 2)
        expected_omega = np.array([[4.0, 2.0], [2.0, 10.0]])
        np.testing.assert_allclose(omega, expected_omega)

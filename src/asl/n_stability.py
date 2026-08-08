"""Diagnostics for single-trial covariance stability across trial count N."""

from __future__ import annotations

import numpy as np

from asl.cholesky import upper_tri_index_pairs


def estimate_c1(summaries: np.ndarray, n_trials: int) -> np.ndarray:
    """Estimate per-trial covariance C1 = N * sample_cov(S_N)."""
    summaries = np.asarray(summaries, dtype=np.float64)
    if summaries.ndim != 2 or summaries.shape[0] < 2:
        raise ValueError("Need at least two replicate summary rows.")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    return n_trials * np.cov(summaries, rowvar=False, bias=False)


def is_positive_definite(matrix: np.ndarray, eig_floor: float = 1e-12) -> bool:
    """True when matrix is finite and all eigenvalues exceed eig_floor."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        return False
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    eigvals = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigvals > eig_floor))


def diagonal_ratios(c_n: np.ndarray, c_ref: np.ndarray) -> np.ndarray:
    """Elementwise diagonal(C_N) / diagonal(C_ref)."""
    return np.diag(c_n) / np.diag(c_ref)


def correlation_matrix(c1: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    scales = np.sqrt(np.diag(c1))
    return c1 / np.outer(scales, scales)


def correlation_differences(c_n: np.ndarray, c_ref: np.ndarray) -> np.ndarray:
    """Correlation(C_N) - Correlation(C_ref)."""
    return correlation_matrix(c_n) - correlation_matrix(c_ref)


def relative_frobenius_error(c_n: np.ndarray, c_ref: np.ndarray) -> float:
    """||C_N - C_ref||_F / ||C_ref||_F."""
    denom = np.linalg.norm(c_ref, ord="fro")
    if denom <= 0.0:
        raise ValueError("Reference matrix has zero Frobenius norm.")
    return float(np.linalg.norm(c_n - c_ref, ord="fro") / denom)


def generalized_eigenvalues(c_n: np.ndarray, c_ref: np.ndarray) -> np.ndarray:
    """Eigenvalues of C_ref^{-1/2} C_N C_ref^{-1/2}, ascending."""
    eigvals, eigvecs = np.linalg.eigh(c_ref)
    if np.any(eigvals <= 0.0):
        raise ValueError("Reference matrix is not positive definite.")
    inv_sqrt = eigvecs * (1.0 / np.sqrt(eigvals))
    whitened = inv_sqrt.T @ c_n @ inv_sqrt
    return np.linalg.eigvalsh(whitened)


def stein_discrepancy(c_n: np.ndarray, c_ref: np.ndarray) -> float:
    """tr(C_ref^{-1} C_N) - logdet(C_ref^{-1} C_N) - p."""
    p = c_ref.shape[0]
    product = np.linalg.solve(c_ref, c_n)
    sign, logdet = np.linalg.slogdet(product)
    if sign <= 0.0:
        raise ValueError("C_ref^{-1} C_N is not positive definite.")
    return float(np.trace(product) - logdet - p)


def omega_from_chol_upper(chol_upper: np.ndarray, n_summaries: int) -> np.ndarray:
    """Build Omega1 = L^T L from flat upper-triangular Cholesky entries."""
    L = np.zeros((n_summaries, n_summaries), dtype=np.float64)
    for k, (i, j) in enumerate(upper_tri_index_pairs(n_summaries)):
        L[i, j] = chol_upper[k]
    return L.T @ L


def sigma_total_from_emulator(
    chol_upper: np.ndarray,
    sigma_emu: np.ndarray,
    n_trials: int,
) -> np.ndarray:
    """Assemble Sigma_total = inv(N * Omega1) + Sigma_emu."""
    n_summaries = int(np.asarray(sigma_emu).shape[0])
    omega1 = omega_from_chol_upper(chol_upper, n_summaries)
    sigma_samp = np.linalg.inv(n_trials * omega1)
    return sigma_samp + np.asarray(sigma_emu, dtype=np.float64)


def mahalanobis_d2(residual: np.ndarray, sigma_total: np.ndarray) -> float:
    """Squared Mahalanobis distance residual^T Sigma^{-1} residual."""
    return float(residual @ np.linalg.solve(sigma_total, residual))


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    """Median and central 90% interval for a 1-d array."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }

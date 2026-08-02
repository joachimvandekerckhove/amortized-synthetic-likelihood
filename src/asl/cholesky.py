"""Cholesky layout, Stein loss, and JAGS synthetic-likelihood helpers."""

import json
from pathlib import Path

import numpy as np
import torch


def n_chol(n: int) -> int:
    """Number of upper-triangular Cholesky entries for an n x n matrix."""
    return n * (n + 1) // 2


def upper_tri_index_pairs(n: int) -> list[tuple[int, int]]:
    """Return (row, col) pairs in row-major upper-triangular order."""
    return [(i, j) for i in range(n) for j in range(i, n)]


def pack_upper_tri(matrix: np.ndarray) -> np.ndarray:
    """Pack the upper triangle of a square matrix into a flat vector."""
    pairs = upper_tri_index_pairs(matrix.shape[0])
    return np.array([matrix[i, j] for i, j in pairs], dtype=matrix.dtype)


def unpack_upper_tri(flat: np.ndarray, n: int) -> np.ndarray:
    """Unpack a flat upper-triangular vector into a symmetric matrix."""
    matrix = np.zeros((n, n), dtype=flat.dtype)
    for k, (i, j) in enumerate(upper_tri_index_pairs(n)):
        matrix[i, j] = flat[k]
        if i != j:
            matrix[j, i] = flat[k]
    return matrix


def std_cov_from_logcov(C1_z: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Map per-trial covariance from log1p space to standardized summary space."""
    scale = np.asarray(scale, dtype=C1_z.dtype)
    return C1_z / np.outer(scale, scale)


def build_L_and_logdet(
    chol_raw: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble upper-triangular L and logdet(Omega) from a flat chol head."""
    if chol_raw.dim() == 1:
        chol_raw = chol_raw.unsqueeze(0)

    batch_size = chol_raw.shape[0]
    pairs = upper_tri_index_pairs(n)
    L = torch.zeros(batch_size, n, n, dtype=chol_raw.dtype, device=chol_raw.device)

    for k, (i, j) in enumerate(pairs):
        value = chol_raw[:, k]
        if i == j:
            value = torch.nn.functional.softplus(value)
        L[:, i, j] = value

    diag = torch.diagonal(L, dim1=1, dim2=2)
    logdet = 2.0 * torch.sum(torch.log(diag), dim=1)
    return L, logdet


def precision_from_chol(chol_raw: torch.Tensor, n: int) -> torch.Tensor:
    """Build precision matrix P = L^T L from a flat Cholesky head."""
    L, _ = build_L_and_logdet(chol_raw, n)
    return torch.bmm(L.transpose(1, 2), L)


def cov_stein_loss(
    chol_raw: torch.Tensor,
    C1_std: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Stein loss for per-trial precision: trace(P @ C1) - logdet(P)."""
    Omega = precision_from_chol(chol_raw, n)
    trace_term = torch.diagonal(torch.bmm(Omega, C1_std), dim1=1, dim2=2).sum(dim=1)
    _, logdet = build_L_and_logdet(chol_raw, n)
    return torch.mean(trace_term - logdet)


def build_sl_likelihood_line(
    slug: str,
    param_names: tuple[str, ...] | list[str],
    n_summaries: int,
    *,
    obs_name: str = "obs",
    n_trials_name: str = "n_trials",
) -> list[str]:
    """One-line JNNX synthetic likelihood for raw physical summaries."""
    args = ", ".join(param_names)
    dist = f"{slug}_sl"
    return [f"{obs_name}[1:{n_summaries}] ~ {dist}({args}, {n_trials_name})"]


def emulator_error_cov_path(slug: str) -> Path:
    """Path to the saved emulator mean-residual covariance sidecar."""
    return Path("results") / slug / "emulator_error_cov.json"


def debias_emulator_error_cov(
    residual_cov: np.ndarray,
    mean_C1_std: np.ndarray,
    n_rep: int,
    n_replicates: int,
) -> np.ndarray:
    """Remove MC noise in z_mean targets from empirical emulator-error covariance."""
    correction = np.asarray(mean_C1_std, dtype=np.float64) / (n_replicates * n_rep)
    sigma_emu = np.asarray(residual_cov, dtype=np.float64) - correction
    eigvals, eigvecs = np.linalg.eigh(sigma_emu)
    eigvals = np.maximum(eigvals, 1e-10)
    return (eigvecs * eigvals) @ eigvecs.T


def save_emulator_error_cov(
    slug: str,
    sigma_emu: np.ndarray,
    *,
    n_rep: int | None = None,
    n_replicates: int | None = None,
) -> None:
    """Persist debiased emulator mean-residual covariance in standardized space."""
    path = emulator_error_cov_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    sigma_emu = np.asarray(sigma_emu, dtype=np.float64)
    payload = {
        "sigma_emu": sigma_emu.tolist(),
        "summary_dim": int(sigma_emu.shape[0]),
    }
    if n_rep is not None:
        payload["n_rep"] = int(n_rep)
    if n_replicates is not None:
        payload["R"] = int(n_replicates)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_emulator_error_cov(slug: str) -> np.ndarray:
    """Load emulator mean-residual covariance; raises if missing."""
    path = emulator_error_cov_path(slug)
    if not path.exists():
        raise FileNotFoundError(
            f"Emulator error covariance not found at {path}. "
            "Run train-emulator first."
        )
    with open(path) as f:
        payload = json.load(f)
    return np.array(payload["sigma_emu"], dtype=np.float64)


def emulator_output_names_for(
    n_summaries: int, summary_names: tuple[str, ...]
) -> tuple[str, ...]:
    """Build the M ONNX output names for an emulator (mean + Cholesky)."""
    mu_names = tuple(f"mu_{name}" for name in summary_names)
    chol_names = tuple(f"chol_{k + 1}" for k in range(n_chol(n_summaries)))
    return mu_names + chol_names

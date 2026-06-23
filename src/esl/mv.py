"""
esl.mv -- Shared helpers for multivariate emulator training and JAGS wiring.

Defines the Cholesky upper-triangular layout used consistently by train_mv,
export_mv, and the JAGS likelihood generator in multivariate model files.
"""

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
    """Assemble upper-triangular L and logdet(Omega) from a flat chol head.

    Parameters
    ----------
    chol_raw : torch.Tensor
        Shape (batch, n_chol) or (n_chol,).  Diagonal entries receive softplus
        before assembly so L has a positive diagonal.
    n : int
        Summary dimension.

    Returns
    -------
    L : torch.Tensor of shape (batch, n, n)
    logdet : torch.Tensor of shape (batch,)
        log determinant of Omega = L^T L.
    """
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
    """Stein loss for per-trial precision: trace(P @ C1) - logdet(P).

    Minimized at P = inv(C1_std) when C1_std is the per-trial covariance
    in standardized summary space.
    """
    Omega = precision_from_chol(chol_raw, n)
    trace_term = torch.diagonal(torch.bmm(Omega, C1_std), dim1=1, dim2=2).sum(dim=1)
    _, logdet = build_L_and_logdet(chol_raw, n)
    return torch.mean(trace_term - logdet)


def build_mv_jags_likelihood_lines(n_summaries: int) -> list[str]:
    """Generate JAGS lines for N-agnostic multivariate normal likelihood.

    The emulator predicts per-trial precision Omega1 = L^T L.  At inference:

        Sigma_sampling = inverse(n_trials * Omega1)
        Sigma_total = Sigma_sampling + sigma_emu
        obs_std ~ dmnorm(mu_std(theta), inverse(Sigma_total))

    n_trials and sigma_emu[1:n,1:n] must be present in the JAGS data dictionary.
    """
    n = n_summaries
    pairs = upper_tri_index_pairs(n)
    lines = []

    for k, (i, j) in enumerate(pairs):
        lines.append(f"L[{i + 1},{j + 1}] <- pred[{n + k + 1}]")

    for i in range(n):
        for j in range(i):
            lines.append(f"L[{i + 1},{j + 1}] <- 0")

    lines.append(
        f"for (ii in 1:{n}) {{ for (jj in 1:{n}) {{ "
        f"Omega1[ii,jj] <- inprod(L[1:{n},ii], L[1:{n},jj]) }} }}"
    )
    lines.append(
        f"for (ii in 1:{n}) {{ for (jj in 1:{n}) {{ "
        f"Omega_sampling[ii,jj] <- n_trials * Omega1[ii,jj] }} }}"
    )
    lines.append(f"Sigma_sampling[1:{n},1:{n}] <- inverse(Omega_sampling[1:{n},1:{n}])")
    lines.append(
        f"for (ii in 1:{n}) {{ for (jj in 1:{n}) {{ "
        f"Sigma_total[ii,jj] <- Sigma_sampling[ii,jj] + sigma_emu[ii,jj] }} }}"
    )
    lines.append(f"Omega_total[1:{n},1:{n}] <- inverse(Sigma_total[1:{n},1:{n}])")
    lines.append(f"obs_std[1:{n}] ~ dmnorm(pred[1:{n}], Omega_total[1:{n},1:{n}])")
    return lines


def emulator_error_cov_path(slug: str) -> Path:
    """Path to the saved emulator mean-residual covariance sidecar."""
    return Path("results") / slug / "emulator_error_cov.json"


def debias_emulator_error_cov(
    residual_cov: np.ndarray,
    mean_C1_std: np.ndarray,
    n_rep: int,
    n_replicates: int,
) -> np.ndarray:
    """Remove MC noise in z_mean targets from empirical emulator-error covariance.

    Cov(mu_pred - z_mean_train) = Cov(b) + E[C1_std] / (R * n_rep).
    """
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
            "Run train-emulator or compute_emulator_error_cov."
        )
    with open(path) as f:
        payload = json.load(f)
    return np.array(payload["sigma_emu"], dtype=np.float64)


def raw_tau_to_std_tau(
    tau_raw: np.ndarray,
    y_raw: np.ndarray,
    rt_mask: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Convert raw sampling precision to standardized-space precision (legacy ddm4mv)."""
    tau_raw = np.maximum(tau_raw, 1e-3)
    tau_std = tau_raw * scale ** 2
    tau_std[rt_mask] *= (y_raw[rt_mask] + 1.0) ** 2
    return tau_std


def build_mv_jags_likelihood_lines_additive(n_summaries: int) -> list[str]:
    """Legacy JAGS lines: emulator cov plus diagonal sampling precision (ddm4mv).

    tau_std[1:N] must be present in the JAGS data dictionary.
    """
    n = n_summaries
    pairs = upper_tri_index_pairs(n)
    lines = []

    for k, (i, j) in enumerate(pairs):
        lines.append(f"L[{i + 1},{j + 1}] <- pred[{n + k + 1}]")

    for i in range(n):
        for j in range(i):
            lines.append(f"L[{i + 1},{j + 1}] <- 0")

    lines.append(
        f"for (ii in 1:{n}) {{ for (jj in 1:{n}) {{ "
        f"Omega_emu[ii,jj] <- inprod(L[1:{n},ii], L[1:{n},jj]) }} }}"
    )
    lines.append(f"Sigma_emu[1:{n},1:{n}] <- inverse(Omega_emu[1:{n},1:{n}])")
    lines.append(
        f"for (ii in 1:{n}) {{ for (jj in 1:{n}) {{ "
        f"Sigma_total[ii,jj] <- Sigma_emu[ii,jj] + equals(ii,jj)/tau_std[jj] }} }}"
    )
    lines.append(f"Omega_total[1:{n},1:{n}] <- inverse(Sigma_total[1:{n},1:{n}])")
    lines.append(f"obs_std[1:{n}] ~ dmnorm(pred[1:{n}], Omega_total[1:{n},1:{n}])")
    return lines


def emulator_output_names_for(n_summaries: int, summary_names: tuple[str, ...]) -> tuple[str, ...]:
    """Build the M ONNX output names for a multivariate emulator."""
    mu_names = tuple(f"mu_{name}" for name in summary_names)
    chol_names = tuple(f"chol_{k + 1}" for k in range(n_chol(n_summaries)))
    return mu_names + chol_names

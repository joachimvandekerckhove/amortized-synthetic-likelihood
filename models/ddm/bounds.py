"""Shared training and prior intervals for DDM models."""

from __future__ import annotations

V_TRAINING = (-3.5, 3.5)
V_PRIOR = (-3.0, 3.0)

A_TRAINING = (0.3, 3.0)
A_PRIOR = (0.5, 2.5)

T0_TRAINING = (0.1, 0.6)
T0_PRIOR = (0.2, 0.5)

W_TRAINING = (0.15, 0.85)
W_PRIOR = (0.2, 0.8)

K_TRAINING = (0.0, 6.0)
K_PRIOR = (0.0, 5.0)


def prior_midpoint(lo: float, hi: float) -> float:
    """Midpoint of a prior interval."""
    return (lo + hi) / 2.0


def truncated_normal_prior(
    name: str, lo: float, hi: float, *, sigma: float | None = None
) -> str:
    """JAGS truncated-normal prior centered at the interval midpoint."""
    if sigma is None:
        sigma = (hi - lo) / 4.0
    mu = prior_midpoint(lo, hi)
    tau = 1.0 / (sigma**2)
    return f"{name} ~ dnorm({mu:g}, {tau:g}) T({lo:g}, {hi:g})"


DDM3_TRAINING_BOUNDS = (V_TRAINING, A_TRAINING, T0_TRAINING)
DDM3_PRIOR_BOUNDS = (V_PRIOR, A_PRIOR, T0_PRIOR)
DDM3_RECOVERY_PRIORS = {
    "v": truncated_normal_prior("v", *V_PRIOR),
    "a": truncated_normal_prior("a", *A_PRIOR),
    "t0": truncated_normal_prior("t0", *T0_PRIOR),
}

DDM4_TRAINING_BOUNDS = (V_TRAINING, A_TRAINING, T0_TRAINING, W_TRAINING)
DDM4_PRIOR_BOUNDS = (V_PRIOR, A_PRIOR, T0_PRIOR, W_PRIOR)
DDM4_RECOVERY_PRIORS = {
    "v": truncated_normal_prior("v", *V_PRIOR),
    "a": truncated_normal_prior("a", *A_PRIOR),
    "t0": truncated_normal_prior("t0", *T0_PRIOR),
    "w": truncated_normal_prior("w", *W_PRIOR),
}

DDMCOLLAPSESIG_TRAINING_BOUNDS = (A_TRAINING, V_TRAINING, K_TRAINING, T0_TRAINING)
DDMCOLLAPSESIG_PRIOR_BOUNDS = (A_PRIOR, V_PRIOR, K_PRIOR, T0_PRIOR)
DDMCOLLAPSESIG_RECOVERY_PRIORS = {
    "a0": truncated_normal_prior("a0", *A_PRIOR),
    "v": truncated_normal_prior("v", *V_PRIOR),
    "k": truncated_normal_prior("k", *K_PRIOR),
    "t0": truncated_normal_prior("t0", *T0_PRIOR),
}

"""Training, inference, and logit-scale intervals for the DW model."""

from __future__ import annotations

from scipy.special import logit

TRAINING_EPSILON_BOUNDS = (0.125, 0.375)
TRAINING_MU_BOUNDS = (0.075, 0.425)
DW_TRAINING_BOUNDS = (TRAINING_EPSILON_BOUNDS, TRAINING_MU_BOUNDS)

PRIOR_EPSILON_BOUNDS = (0.15, 0.35)
PRIOR_MU_BOUNDS = (0.1, 0.4)
DW_PRIOR_BOUNDS = (PRIOR_EPSILON_BOUNDS, PRIOR_MU_BOUNDS)

CANONICAL_PARAM_NAMES = ("epsilon", "mu")


def logit_interval(lo: float, hi: float) -> tuple[float, float]:
    """Map an open canonical interval (lo, hi) subset of (0, 1) to logit scale."""
    return float(logit(lo)), float(logit(hi))


LOGIT_TRAINING_EPSILON_BOUNDS = logit_interval(*TRAINING_EPSILON_BOUNDS)
LOGIT_TRAINING_MU_BOUNDS = logit_interval(*TRAINING_MU_BOUNDS)
DW_LOGIT_TRAINING_BOUNDS = (LOGIT_TRAINING_EPSILON_BOUNDS, LOGIT_TRAINING_MU_BOUNDS)

LOGIT_PRIOR_EPSILON_BOUNDS = logit_interval(*PRIOR_EPSILON_BOUNDS)
LOGIT_PRIOR_MU_BOUNDS = logit_interval(*PRIOR_MU_BOUNDS)
DW_LOGIT_PRIOR_BOUNDS = (LOGIT_PRIOR_EPSILON_BOUNDS, LOGIT_PRIOR_MU_BOUNDS)


def uniform_prior(name: str, lo: float, hi: float) -> str:
    return f"{name} ~ dunif({lo:g}, {hi:g})"


DW_RECOVERY_PRIORS = {
    "logit_epsilon": uniform_prior("logit_epsilon", *LOGIT_PRIOR_EPSILON_BOUNDS),
    "logit_mu": uniform_prior("logit_mu", *LOGIT_PRIOR_MU_BOUNDS),
}

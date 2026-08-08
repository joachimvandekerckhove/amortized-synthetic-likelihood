"""Training and recovery canonical intervals for the DW model."""

from __future__ import annotations

TRAINING_EPSILON_BOUNDS = (0.125, 0.375)
TRAINING_MU_BOUNDS = (0.075, 0.425)
DW_TRAINING_BOUNDS = (TRAINING_EPSILON_BOUNDS, TRAINING_MU_BOUNDS)

PRIOR_EPSILON_BOUNDS = (0.15, 0.35)
PRIOR_MU_BOUNDS = (0.1, 0.4)
DW_PRIOR_BOUNDS = (PRIOR_EPSILON_BOUNDS, PRIOR_MU_BOUNDS)


def uniform_prior(name: str, lo: float, hi: float) -> str:
    return f"{name} ~ dunif({lo:g}, {hi:g})"


DW_RECOVERY_PRIORS = {
    "epsilon": uniform_prior("epsilon", *PRIOR_EPSILON_BOUNDS),
    "mu": uniform_prior("mu", *PRIOR_MU_BOUNDS),
}

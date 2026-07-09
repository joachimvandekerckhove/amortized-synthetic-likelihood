"""
models.ddm.ddm3 -- Three-parameter drift diffusion model.

Parameters: drift rate (v), boundary separation (a), nondecision time (t0).
Starting bias is fixed at w = 0.5 (unbiased).
Summaries: accuracy (proportion correct), mean RT, variance of RT.

Parameter ranges match the reference implementation (Chavez De la Pena &
Vandekerckhove 2025) to ensure numerically well-behaved recovery:
  v  in (-2, 2)    -- avoids near-0% / near-100% accuracy extremes
  a  in (0.5, 2.0) -- realistic boundary separations
  t0 in (0.15, 0.45) -- typical nondecision times in cognitive experiments
"""

import numpy as np

from asl.spec import Model
from models.ddm.simulator import simulate_ddm_paths_biased

FIXED_BIAS = 0.5


def simulate_summaries_ddm3(
    params: np.ndarray, n_trials: int, seed: int
) -> np.ndarray:
    """Simulate summary statistics for the 3-parameter DDM."""
    v, a, t0 = float(params[0]), float(params[1]), float(params[2])

    reaction_times, choices = simulate_ddm_paths_biased(
        drift_rate=v,
        boundary_separation=a,
        nondecision_time=t0,
        starting_bias=FIXED_BIAS,
        n_samples=n_trials,
        seed=seed,
    )

    n_absorbed = len(reaction_times)
    if n_absorbed < 2:
        return np.array([np.nan, np.nan, np.nan])

    accuracy = float(np.mean(choices == 1))
    rt_mean = float(np.mean(reaction_times))
    rt_var = float(np.var(reaction_times, ddof=1))

    return np.array([accuracy, rt_mean, rt_var])


DDM3 = Model(
    slug="ddm3",
    param_names=("v", "a", "t0"),
    param_bounds=((-2.0, 2.0), (0.5, 2.0), (0.15, 0.45)),
    summary_names=("acc", "rt_mean", "rt_var"),
    simulate_summaries=simulate_summaries_ddm3,
    recovery_priors={
        "v": "v ~ dnorm(0, 0.25)",
        "a": "a ~ dunif(0.3, 2.5)",
        "t0": "t0 ~ dunif(0.1, 0.5)",
    },
)

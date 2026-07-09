"""
models.ddm.ddm4 -- Four-parameter drift diffusion model.

Parameters: drift rate (v), boundary separation (a), nondecision time (t0),
starting-point bias (w).
Summaries: rt_mean_corr, rt_var_corr, rt_mean_err, rt_var_err, err_rate.

Parameter ranges match the reference implementation to ensure numerically
well-behaved recovery.
"""

import numpy as np

from asl.spec import Model
from models.ddm.simulator import simulate_ddm_paths_biased


def simulate_summaries_ddm4(
    params: np.ndarray, n_trials: int, seed: int
) -> np.ndarray:
    """Simulate summary statistics for the 4-parameter DDM."""
    v, a, t0, w = (
        float(params[0]),
        float(params[1]),
        float(params[2]),
        float(params[3]),
    )

    reaction_times, choices = simulate_ddm_paths_biased(
        drift_rate=v,
        boundary_separation=a,
        nondecision_time=t0,
        starting_bias=w,
        n_samples=n_trials,
        seed=seed,
    )

    rts_correct = reaction_times[choices == 1]
    rts_error = reaction_times[choices == 0]
    n_total = len(reaction_times)

    if len(rts_correct) < 2 or len(rts_error) < 2:
        return np.full(5, np.nan)

    rt_mean_corr = float(np.mean(rts_correct))
    rt_var_corr = float(np.var(rts_correct, ddof=1))
    rt_mean_err = float(np.mean(rts_error))
    rt_var_err = float(np.var(rts_error, ddof=1))
    err_rate = float(len(rts_error) / n_total)

    return np.array([rt_mean_corr, rt_var_corr, rt_mean_err, rt_var_err, err_rate])


DDM4 = Model(
    slug="ddm4",
    param_names=("v", "a", "t0", "w"),
    param_bounds=((-2.0, 2.0), (0.5, 2.0), (0.15, 0.45), (0.15, 0.85)),
    summary_names=("rt_mean_corr", "rt_var_corr", "rt_mean_err", "rt_var_err", "err_rate"),
    simulate_summaries=simulate_summaries_ddm4,
    recovery_priors={
        "v": "v ~ dnorm(0, 0.25)",
        "a": "a ~ dunif(0.25, 3.0)",
        "t0": "t0 ~ dunif(0.1, 0.6)",
        "w": "w ~ dbeta(2, 2)",
    },
)

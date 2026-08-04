"""
models.ddm.ddm3 -- Three-parameter drift diffusion model.

Parameters: drift rate (v), boundary separation (a), nondecision time (t0).
Starting bias is fixed at w = 0.5 (unbiased).
Summaries: accuracy, mean RT, variance of RT.
"""

import numpy as np

from asl.cholesky import build_sl_likelihood_line, emulator_output_names_for
from asl.spec import Model
from models.ddm.bounds import (
    DDM3_PRIOR_BOUNDS,
    DDM3_RECOVERY_PRIORS,
    DDM3_TRAINING_BOUNDS,
)
from models.ddm.simulator import simulate_ddm_paths_biased

FIXED_BIAS = 0.5

PARAM_NAMES = ("v", "a", "t0")
PARAM_BOUNDS = DDM3_TRAINING_BOUNDS
PRIOR_BOUNDS = DDM3_PRIOR_BOUNDS
SUMMARY_NAMES = ("acc", "rt_mean", "rt_var")
SUMMARY_TRANSFORMS = ("identity", "log1p", "log1p")
N_SUMMARIES = len(SUMMARY_NAMES)

RECOVERY_PRIORS = DDM3_RECOVERY_PRIORS


def simulate_summaries(params: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
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


def build_jags_likelihood(obs: dict) -> list[str]:
    del obs
    return build_sl_likelihood_line("ddm3", PARAM_NAMES, N_SUMMARIES)


DDM3 = Model(
    slug="ddm3",
    param_names=PARAM_NAMES,
    param_bounds=PARAM_BOUNDS,
    prior_bounds=PRIOR_BOUNDS,
    summary_names=SUMMARY_NAMES,
    summary_transforms=SUMMARY_TRANSFORMS,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, SUMMARY_NAMES),
    simulate_summaries=simulate_summaries,
    recovery_priors=RECOVERY_PRIORS,
    build_jags_likelihood=build_jags_likelihood,
    default_architecture="DeepWide_24x4",
)

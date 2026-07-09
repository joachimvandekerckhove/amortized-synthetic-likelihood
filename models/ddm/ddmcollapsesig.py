"""
models.ddm.ddmcollapsesig -- DDM with symmetric sigmoid collapsing bounds.

Two conditions share the parameter vector (a0, v, k, t0):

* fixed: constant boundary a0 / 2 (k is ignored)
* collapse: boundary a0 / (1 + exp(k t))
"""

from __future__ import annotations

import numpy as np

from models.ddm.simulator import SimulationConfig

FIXED_BIAS = 0.5
RT_QUANTILES = (10, 30, 50, 70, 90)
TAIL_QUANTILES = (50, 90)

SUMMARY_NAMES = (
    "acc",
    "rt_q10",
    "rt_q30",
    "rt_q50",
    "rt_q70",
    "rt_q90",
    "rt_q50_corr",
    "rt_q90_corr",
    "rt_q50_err",
    "rt_q90_err",
)
N_SUMMARIES = len(SUMMARY_NAMES)

PARAM_NAMES = ("a0", "v", "k", "t0")
PARAM_BOUNDS = ((0.5, 2.0), (-1.5, 1.5), (0.0, 8.0), (0.05, 0.45))
RECOVERY_PRIORS = {
    "a0": "a0 ~ dunif(0.5, 2.0)",
    "v": "v ~ dunif(-1.5, 1.5)",
    "k": "k ~ dunif(0, 8.0)",
    "t0": "t0 ~ dunif(0.05, 0.45)",
}


def collapse_bound(times: np.ndarray | float, a0: float, k: float) -> np.ndarray:
    """Return the positive symmetric bound for elapsed decision times."""
    times_array = np.asarray(times, dtype=np.float64)
    return a0 / (1.0 + np.exp(k * times_array))


def compute_rt_quantiles(
    reaction_times: np.ndarray, percentiles: tuple[int, ...]
) -> np.ndarray:
    """Return RT percentiles, or NaN if fewer than two RTs."""
    if len(reaction_times) < 2:
        return np.full(len(percentiles), np.nan, dtype=np.float64)
    return np.percentile(reaction_times, percentiles).astype(np.float64)


def simulate_paths(
    a0: float,
    v: float,
    k: float,
    t0: float,
    n_samples: int,
    seed: int,
    *,
    collapse: bool,
    config: SimulationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate DDM paths with either fixed or sigmoid collapsing bounds."""
    if config is None:
        config = SimulationConfig()

    dt = config.dt
    sigma = config.sigma
    n_steps = int(config.max_time / dt)
    rng = np.random.default_rng(seed)
    start_position = a0 * (FIXED_BIAS - 0.5)

    positions = np.full(n_samples, start_position, dtype=np.float64)
    alive = np.ones(n_samples, dtype=bool)
    decision_times = np.full(n_samples, np.nan, dtype=np.float64)
    choices = np.full(n_samples, -1, dtype=np.int8)
    sqrt_dt = np.sqrt(dt)

    for step in range(1, n_steps + 1):
        if not alive.any():
            break

        current_time = step * dt
        if collapse:
            bound = float(collapse_bound(current_time, a0, k))
        else:
            bound = a0 / 2.0

        noise = rng.standard_normal(alive.sum())
        positions[alive] += v * dt + sigma * sqrt_dt * noise

        crossed_upper = (positions >= bound) & alive
        crossed_lower = (positions <= -bound) & alive
        decision_times[crossed_upper] = current_time
        decision_times[crossed_lower] = current_time
        choices[crossed_upper] = 1
        choices[crossed_lower] = 0
        alive[crossed_upper | crossed_lower] = False

    absorbed = ~np.isnan(decision_times)
    if not absorbed.any():
        raise RuntimeError("No paths absorbed within simulation horizon.")
    return decision_times[absorbed] + t0, choices[absorbed]


def summaries_from_paths(reaction_times: np.ndarray, choices: np.ndarray) -> np.ndarray:
    """Compute accuracy and RT quantile summaries."""
    if len(reaction_times) < 2:
        return np.full(N_SUMMARIES, np.nan)

    rts_corr = reaction_times[choices == 1]
    rts_err = reaction_times[choices == 0]
    if len(rts_corr) < 2 or len(rts_err) < 2:
        return np.full(N_SUMMARIES, np.nan)

    overall_q = compute_rt_quantiles(reaction_times, RT_QUANTILES)
    corr_q = compute_rt_quantiles(rts_corr, TAIL_QUANTILES)
    err_q = compute_rt_quantiles(rts_err, TAIL_QUANTILES)
    return np.concatenate(
        [[float(np.mean(choices == 1))], overall_q, corr_q, err_q]
    ).astype(np.float64)


def simulate_summaries_fixed(params: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
    """Simulate summaries for the fixed-bound condition."""
    a0, v, k, t0 = (float(params[i]) for i in range(4))
    reaction_times, choices = simulate_paths(
        a0, v, k, t0, n_trials, seed, collapse=False
    )
    return summaries_from_paths(reaction_times, choices)


def simulate_summaries_collapse(
    params: np.ndarray, n_trials: int, seed: int
) -> np.ndarray:
    """Simulate summaries for the sigmoid-collapse condition."""
    a0, v, k, t0 = (float(params[i]) for i in range(4))
    reaction_times, choices = simulate_paths(
        a0, v, k, t0, n_trials, seed, collapse=True
    )
    return summaries_from_paths(reaction_times, choices)

"""
models.ddm.ddmcollapsesig -- DDM with symmetric sigmoid collapsing bounds.

Parameters: a0, v, k, t0

The boundary at time t is a0 / (1 + exp(k * t)).  When k = 0 this reduces
to the constant boundary a0 / 2, so the fixed-bound case is a special case
of this model and requires no separate treatment.
"""

from __future__ import annotations

import numpy as np

from asl.cholesky import build_sl_likelihood_line, emulator_output_names_for
from asl.spec import Model
from models.ddm.bounds import (
    DDMCOLLAPSESIG_PRIOR_BOUNDS,
    DDMCOLLAPSESIG_RECOVERY_PRIORS,
    DDMCOLLAPSESIG_TRAINING_BOUNDS,
)
from models.ddm.simulator import SimulationConfig

FIXED_BIAS = 0.5
MIN_TERTILE = 15

SUMMARY_NAMES = (
    "acc",
    "rt_q10",
    "var_t1",
    "var_t3_minus_t1",
)
N_SUMMARIES = len(SUMMARY_NAMES)

PARAM_NAMES = ("a0", "v", "k", "t0")
PARAM_BOUNDS = DDMCOLLAPSESIG_TRAINING_BOUNDS
PRIOR_BOUNDS = DDMCOLLAPSESIG_PRIOR_BOUNDS
SUMMARY_TRANSFORMS = ("identity", "log1p", "log1p", "log1p")
RECOVERY_PRIORS = DDMCOLLAPSESIG_RECOVERY_PRIORS


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


def tertile_partition(rt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split RTs into lower / middle / upper tertiles."""
    if len(rt) < 3 * MIN_TERTILE:
        return np.array([]), np.array([]), np.array([])
    q1, q2 = np.percentile(rt, [100 / 3, 200 / 3])
    low = rt[rt <= q1]
    mid = rt[(rt > q1) & (rt <= q2)]
    high = rt[rt > q2]
    if min(len(low), len(mid), len(high)) < MIN_TERTILE:
        return np.array([]), np.array([]), np.array([])
    return low, mid, high


def simulate_paths(
    a0: float,
    v: float,
    k: float,
    t0: float,
    n_samples: int,
    seed: int,
    config: SimulationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate DDM paths with a sigmoid collapsing boundary (k=0 gives fixed bounds)."""
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
        bound = float(collapse_bound(current_time, a0, k))

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
    """Compute accuracy, lower-tail RT, and tertile variance summaries."""
    if len(reaction_times) < 2:
        return np.full(N_SUMMARIES, np.nan)

    acc = float(np.mean(choices == 1))
    rt_q10 = float(compute_rt_quantiles(reaction_times, (10,))[0])

    low, _, high = tertile_partition(reaction_times.astype(np.float64))
    if low.size == 0:
        return np.full(N_SUMMARIES, np.nan)

    var_t1 = float(np.var(low, ddof=1))
    var_t3 = float(np.var(high, ddof=1))
    return np.array([acc, rt_q10, var_t1, var_t3 - var_t1], dtype=np.float64)


def simulate_summaries(params: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
    """Simulate summaries for the sigmoid-collapsing-bound DDM."""
    a0, v, k, t0 = (float(params[i]) for i in range(4))
    reaction_times, choices = simulate_paths(a0, v, k, t0, n_trials, seed)
    return summaries_from_paths(reaction_times, choices)


def build_jags_likelihood(obs: dict) -> list[str]:
    del obs
    return build_sl_likelihood_line("ddmcollapsesig", PARAM_NAMES, N_SUMMARIES)


DDMCOLLAPSESIG = Model(
    slug="ddmcollapsesig",
    param_names=PARAM_NAMES,
    param_bounds=PARAM_BOUNDS,
    prior_bounds=PRIOR_BOUNDS,
    summary_names=SUMMARY_NAMES,
    summary_transforms=SUMMARY_TRANSFORMS,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, SUMMARY_NAMES),
    simulate_summaries=simulate_summaries,
    recovery_priors=RECOVERY_PRIORS,
    build_jags_likelihood=build_jags_likelihood,
    default_architecture="DeepWide_32x6",
)

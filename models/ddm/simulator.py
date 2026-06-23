"""
models.ddm.simulator -- Biased drift-diffusion model Euler simulator.

Implements the vectorized Euler simulation of a drift-diffusion process with
symmetric absorbing bounds.  Used by both ddm3 (with w fixed to 0.5) and ddm4
(with free starting-point bias w).

The latent variable evolves as:
    x_{t+dt} = x_t + v*dt + sigma*sqrt(dt)*epsilon_t
with absorbing bounds at +a/2 and -a/2, starting at x_0 = a*(w - 0.5).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for the DDM Euler simulation.

    Attributes
    ----------
    dt : float
        Time step in seconds.
    max_time : float
        Maximum simulation duration before a path is censored.
    sigma : float
        Noise scaling (conventionally 1.0).
    """

    dt: float = 0.001
    max_time: float = 10.0
    sigma: float = 1.0


def simulate_ddm_paths_biased(
    drift_rate: float,
    boundary_separation: float,
    nondecision_time: float,
    starting_bias: float,
    n_samples: int,
    seed: int,
    config: SimulationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate DDM paths and return decision times and choices.

    Parameters
    ----------
    drift_rate : float
        Drift rate v.
    boundary_separation : float
        Threshold separation a.
    nondecision_time : float
        Non-decision time t0 added to decision time.
    starting_bias : float
        Relative starting point w in (0, 1); w=0.5 is unbiased.
    n_samples : int
        Number of independent paths to simulate.
    seed : int
        Random seed for reproducibility.
    config : SimulationConfig or None
        Euler simulation parameters.  Uses defaults if None.

    Returns
    -------
    reaction_times : np.ndarray of shape (n_absorbed,)
        Observed reaction times (decision_time + nondecision_time).
    choices : np.ndarray of shape (n_absorbed,)
        1 for upper-boundary (correct), 0 for lower-boundary (error).

    Raises
    ------
    RuntimeError
        If no paths hit a boundary within max_time.
    """
    if config is None:
        config = SimulationConfig()

    dt = config.dt
    sigma = config.sigma
    n_steps = int(config.max_time / dt)
    rng = np.random.default_rng(seed)

    upper_bound = boundary_separation / 2.0
    lower_bound = -boundary_separation / 2.0
    start_position = boundary_separation * (starting_bias - 0.5)

    positions = np.full(n_samples, start_position, dtype=np.float64)
    alive = np.ones(n_samples, dtype=bool)
    decision_times = np.full(n_samples, np.nan, dtype=np.float64)
    choices = np.full(n_samples, -1, dtype=np.int8)

    sqrt_dt = np.sqrt(dt)

    for step in range(1, n_steps + 1):
        n_alive = alive.sum()
        if n_alive == 0:
            break

        noise = rng.standard_normal(n_alive)
        positions[alive] += drift_rate * dt + sigma * sqrt_dt * noise

        crossed_upper = (positions >= upper_bound) & alive
        crossed_lower = (positions <= lower_bound) & alive

        current_time = step * dt
        decision_times[crossed_upper] = current_time
        decision_times[crossed_lower] = current_time
        choices[crossed_upper] = 1
        choices[crossed_lower] = 0
        alive[crossed_upper | crossed_lower] = False

    absorbed = ~np.isnan(decision_times)
    if not absorbed.any():
        raise RuntimeError(
            f"No paths hit a boundary within max_time={config.max_time}s. "
            f"Parameters: v={drift_rate}, a={boundary_separation}, w={starting_bias}"
        )

    reaction_times = decision_times[absorbed] + nondecision_time
    return reaction_times, choices[absorbed]

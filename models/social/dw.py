"""
Deffuant-Weisbuch bounded-confidence opinion dynamics for ASL.

Parameters are inferred on the canonical scale: epsilon (confidence bound)
and mu (compromise rate). Training draws are uniform on slightly wider
supports than the JAGS priors and recovery true-parameter distribution.
"""

from __future__ import annotations

import numpy as np

from asl.cholesky import build_sl_likelihood_line, emulator_output_names_for
from asl.spec import Model
from models.social.dw_bounds import (
    DW_PRIOR_BOUNDS,
    DW_RECOVERY_PRIORS,
    DW_TRAINING_BOUNDS,
    PRIOR_EPSILON_BOUNDS,
    PRIOR_MU_BOUNDS,
    TRAINING_EPSILON_BOUNDS,
    TRAINING_MU_BOUNDS,
)

PARAM_NAMES = ("epsilon", "mu")
PARAM_BOUNDS = DW_TRAINING_BOUNDS
PRIOR_PARAM_BOUNDS = DW_PRIOR_BOUNDS

SUMMARY_NAMES = (
    "effective_clusters_final",
    "opinion_entropy_final",
    "mean_opinion_shift",
    "late_opinion_variance",
    "abs_variance_change",
    "large_move_rate",
)
N_SUMMARIES = len(SUMMARY_NAMES)

SUMMARY_TRANSFORMS = ("log1p", "log1p", "log1p", "log1p", "log1p", "identity")

N_AGENTS = 150
N_WAVES = 5
N_BINS = 20
MOVE_THRESHOLD = 0.15


def _bin_proportions(opinions: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(opinions, bins=N_BINS, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return np.full(N_BINS, 1.0 / N_BINS)
    return counts.astype(np.float64) / total


def _effective_clusters(proportions: np.ndarray) -> float:
    return float(1.0 / np.sum(proportions**2))


def _opinion_entropy(proportions: np.ndarray) -> float:
    positive = proportions[proportions > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def _run_interactions(
    opinions: np.ndarray,
    epsilon: float,
    mu: float,
    n_events: int,
    rng: np.random.Generator,
) -> None:
    n_agents = opinions.shape[0]
    for _ in range(n_events):
        i, j = rng.choice(n_agents, size=2, replace=False)
        diff = abs(opinions[i] - opinions[j])
        if diff <= epsilon:
            opinions[i] += mu * (opinions[j] - opinions[i])
            opinions[j] += mu * (opinions[i] - opinions[j])


def _is_degenerate_run(waves: list[np.ndarray], mu: float) -> bool:
    if mu < TRAINING_MU_BOUNDS[0]:
        return False
    total_movement = 0.0
    for left, right in zip(waves[:-1], waves[1:]):
        total_movement += float(np.mean(np.abs(right - left)))
    return total_movement < 0.001


def _simulate_opinion_waves(
    epsilon: float,
    mu: float,
    n_trials: int,
    seed: int,
) -> list[np.ndarray] | None:
    n_intervals = N_WAVES - 1
    events_per_interval = max(1, int(n_trials // n_intervals))

    rng = np.random.default_rng(seed + 3_000_007)
    opinions = rng.uniform(0.0, 1.0, size=N_AGENTS).astype(np.float64)
    waves = [opinions.copy()]

    for _ in range(n_intervals):
        _run_interactions(opinions, epsilon, mu, events_per_interval, rng)
        opinions = np.clip(opinions, 0.0, 1.0)
        waves.append(opinions.copy())

    if _is_degenerate_run(waves, mu):
        return None
    return waves


def _large_move_fraction(prev_opinions: np.ndarray, next_opinions: np.ndarray) -> float:
    moves = np.abs(next_opinions - prev_opinions)
    return float(np.mean(moves > MOVE_THRESHOLD))


def _summaries_from_waves(waves: list[np.ndarray]) -> np.ndarray:
    w0, wf = waves[0], waves[-1]
    proportions_final = _bin_proportions(wf)
    variances = np.array([float(w.var()) for w in waves], dtype=np.float64)
    move_fractions = [
        _large_move_fraction(waves[i], waves[i + 1]) for i in range(len(waves) - 1)
    ]
    return np.array(
        [
            _effective_clusters(proportions_final),
            _opinion_entropy(proportions_final),
            float(np.mean(np.abs(wf - w0))),
            float(variances[-1]),
            float(abs(variances[-1] - variances[0])),
            float(np.mean(move_fractions)),
        ],
        dtype=np.float64,
    )


def draw_cov_parameters(rng: np.random.Generator) -> np.ndarray:
    """Uniform draws on the canonical training support."""
    return np.array(
        [
            rng.uniform(*TRAINING_EPSILON_BOUNDS),
            rng.uniform(*TRAINING_MU_BOUNDS),
        ],
        dtype=np.float64,
    )


def to_canonical(params: np.ndarray) -> tuple[float, float]:
    return float(params[0]), float(params[1])


def simulate_summaries(params: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
    epsilon, mu = map(float, params)
    if n_trials < N_AGENTS:
        return np.full(N_SUMMARIES, np.nan)

    waves = _simulate_opinion_waves(epsilon, mu, n_trials, seed)
    if waves is None:
        return np.full(N_SUMMARIES, np.nan)

    summaries = _summaries_from_waves(waves)
    if not np.all(np.isfinite(summaries)):
        return np.full(N_SUMMARIES, np.nan)
    return summaries


RECOVERY_PRIORS = DW_RECOVERY_PRIORS


def build_jags_likelihood(obs: dict) -> list[str]:
    del obs
    return build_sl_likelihood_line("dw", PARAM_NAMES, N_SUMMARIES)


DW = Model(
    slug="dw",
    param_names=PARAM_NAMES,
    param_bounds=PARAM_BOUNDS,
    prior_bounds=PRIOR_PARAM_BOUNDS,
    summary_names=SUMMARY_NAMES,
    summary_transforms=SUMMARY_TRANSFORMS,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, SUMMARY_NAMES),
    simulate_summaries=simulate_summaries,
    draw_cov_parameters=draw_cov_parameters,
    recovery_priors=RECOVERY_PRIORS,
    build_jags_likelihood=build_jags_likelihood,
    default_architecture="DeepWide_32x6",
)

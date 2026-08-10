"""
Deffuant-Weisbuch bounded-confidence opinion dynamics for ASL.

Inference uses logit-scale parameters mapped to the canonical (0, 1) range via
the logistic sigmoid. Training and prior support are specified as canonical
intervals in dw_bounds and converted to logit bounds; the transform itself does
not bake in any affine limits.

Sample size N is the number of agents. Pairwise interaction exposure is
controlled separately via events_per_agent_per_interval.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from asl.cholesky import build_sl_likelihood_line, emulator_output_names_for
from asl.config import load_config
from asl.spec import Model
from models.social.dw_bounds import (
    CANONICAL_PARAM_NAMES,
    DW_LOGIT_PRIOR_BOUNDS,
    DW_LOGIT_TRAINING_BOUNDS,
    DW_PRIOR_BOUNDS,
    DW_RECOVERY_PRIORS,
    DW_TRAINING_BOUNDS,
    LOGIT_TRAINING_EPSILON_BOUNDS,
    LOGIT_TRAINING_MU_BOUNDS,
    PRIOR_EPSILON_BOUNDS,
    PRIOR_MU_BOUNDS,
    TRAINING_EPSILON_BOUNDS,
    TRAINING_MU_BOUNDS,
)

PARAM_NAMES = ("logit_epsilon", "logit_mu")
PARAM_BOUNDS = DW_LOGIT_TRAINING_BOUNDS
PRIOR_PARAM_BOUNDS = DW_LOGIT_PRIOR_BOUNDS

SUMMARY_NAMES = (
    "mean_pairwise_distance_final",
    "mean_pairwise_sq_distance_final",
    "mean_opinion_shift",
    "late_opinion_variance",
    "abs_variance_change",
    "large_move_rate",
)
N_SUMMARIES = len(SUMMARY_NAMES)

SUMMARY_TRANSFORMS = ("log1p", "log1p", "log1p", "log1p", "log1p", "identity")

DEFAULT_N_AGENTS = 150
DEFAULT_EVENTS_PER_AGENT_PER_INTERVAL = 1.0
N_WAVES = 5
MOVE_THRESHOLD = 0.15

# Fixed canonical (epsilon, mu) pairs for the DW N-stability study.
DW_STUDY_CANONICAL_THETAS = (
    (0.25, 0.35),
    (0.25, 0.25),
    (0.30, 0.30),
    (0.20, 0.35),
    (0.35, 0.20),
)


def resolve_events_per_agent_per_interval() -> float:
    """Read interaction exposure from config or fall back to the paper default."""
    config = load_config()
    return float(
        config.get(
            "simulator",
            "events_per_agent_per_interval",
            DEFAULT_EVENTS_PER_AGENT_PER_INTERVAL,
        )
    )


def events_per_interval_count(
    n_agents: int,
    events_per_agent_per_interval: float | None = None,
) -> int:
    """Pair-selection attempts per inter-wave interval."""
    rate = (
        events_per_agent_per_interval
        if events_per_agent_per_interval is not None
        else resolve_events_per_agent_per_interval()
    )
    return max(1, int(round(n_agents * rate)))


def _mean_pairwise_distance(opinions: np.ndarray) -> float:
    """U-statistic for mean |x_i - x_j|; stable across agent counts."""
    n_agents = opinions.shape[0]
    if n_agents < 2:
        return 0.0
    diffs = np.abs(opinions[:, None] - opinions[None, :])
    return float(diffs.sum() / (n_agents * (n_agents - 1)))


def _mean_pairwise_sq_distance(opinions: np.ndarray) -> float:
    """U-statistic for mean (x_i - x_j)^2; complements the L1 dispersion."""
    n_agents = opinions.shape[0]
    if n_agents < 2:
        return 0.0
    diffs = (opinions[:, None] - opinions[None, :]) ** 2
    return float(diffs.sum() / (n_agents * (n_agents - 1)))


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
            xi = opinions[i]
            xj = opinions[j]
            opinions[i] = xi + mu * (xj - xi)
            opinions[j] = xj + mu * (xi - xj)


def _simulate_opinion_waves(
    epsilon: float,
    mu: float,
    n_agents: int,
    seed: int,
    *,
    events_per_agent_per_interval: float | None = None,
) -> list[np.ndarray] | None:
    n_intervals = N_WAVES - 1
    events_per_interval = events_per_interval_count(
        n_agents, events_per_agent_per_interval
    )

    rng = np.random.default_rng(seed + 3_000_007)
    opinions = rng.uniform(0.0, 1.0, size=n_agents).astype(np.float64)
    waves = [opinions.copy()]

    for _ in range(n_intervals):
        _run_interactions(opinions, epsilon, mu, events_per_interval, rng)
        opinions = np.clip(opinions, 0.0, 1.0)
        waves.append(opinions.copy())

    return waves


def _large_move_fraction(prev_opinions: np.ndarray, next_opinions: np.ndarray) -> float:
    moves = np.abs(next_opinions - prev_opinions)
    return float(np.mean(moves > MOVE_THRESHOLD))


def _summaries_from_waves(waves: list[np.ndarray]) -> np.ndarray:
    w0, wf = waves[0], waves[-1]
    variances = np.array([float(w.var()) for w in waves], dtype=np.float64)
    move_fractions = [
        _large_move_fraction(waves[i], waves[i + 1]) for i in range(len(waves) - 1)
    ]
    return np.array(
        [
            _mean_pairwise_distance(wf),
            _mean_pairwise_sq_distance(wf),
            float(np.mean(np.abs(wf - w0))),
            float(variances[-1]),
            float(abs(variances[-1] - variances[0])),
            float(np.mean(move_fractions)),
        ],
        dtype=np.float64,
    )


def to_canonical(params: np.ndarray) -> tuple[float, float]:
    """Map logit parameters to canonical epsilon and mu in (0, 1)."""
    canonical = canonical_params_array(np.asarray(params, dtype=np.float64).reshape(1, -1))
    return float(canonical[0, 0]), float(canonical[0, 1])


def canonical_params_array(params: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid map from logit inputs to canonical parameters."""
    return expit(np.asarray(params, dtype=np.float64))


def study_logit_thetas(n_theta: int) -> np.ndarray:
    """Fixed canonical (epsilon, mu) values for the DW N-stability study."""
    canonical = np.asarray(DW_STUDY_CANONICAL_THETAS, dtype=np.float64)
    logits = logit(canonical)
    if n_theta <= len(logits):
        return logits[:n_theta]
    reps = int(np.ceil(n_theta / len(logits)))
    return np.tile(logits, (reps, 1))[:n_theta]


def draw_cov_parameters(rng: np.random.Generator) -> np.ndarray:
    """Uniform draws on the logit training support."""
    return np.array(
        [
            rng.uniform(*LOGIT_TRAINING_EPSILON_BOUNDS),
            rng.uniform(*LOGIT_TRAINING_MU_BOUNDS),
        ],
        dtype=np.float64,
    )


def simulate_summaries(params: np.ndarray, n_agents: int, seed: int) -> np.ndarray:
    epsilon, mu = to_canonical(params)
    if n_agents < 2:
        return np.full(N_SUMMARIES, np.nan)

    waves = _simulate_opinion_waves(epsilon, mu, n_agents, seed)
    summaries = _summaries_from_waves(waves)
    if not np.all(np.isfinite(summaries)):
        return np.full(N_SUMMARIES, np.nan)
    return summaries


RECOVERY_PRIORS = DW_RECOVERY_PRIORS


def build_jags_likelihood(obs: dict) -> list[str]:
    del obs
    return build_sl_likelihood_line(
        "dw", PARAM_NAMES, N_SUMMARIES, n_trials_name="n_agents"
    )


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
    report_param_names=CANONICAL_PARAM_NAMES,
    report_params_fn=canonical_params_array,
    report_prior_bounds=DW_PRIOR_BOUNDS,
    sample_size_arg="n_agents",
)

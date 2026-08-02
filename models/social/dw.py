"""
Deffuant-Weisbuch bounded-confidence opinion dynamics for ASL.

Canonical parameters epsilon (confidence bound) and mu (compromise rate) are
mapped from logit inputs. Epsilon training uses raw bounds [0.15, 0.85]; the
inference prior uses [0.2, 0.8] (narrower, in raw epsilon units).
"""

from __future__ import annotations

import numpy as np

from asl.cholesky import build_sl_likelihood_line, emulator_output_names_for
from asl.spec import Model

PARAM_NAMES = ("logit_epsilon", "logit_mu")
PARAM_BOUNDS = ((-4.0, 4.0), (-4.0, 4.0))

TRAINING_EPSILON_MIN = 0.15
TRAINING_EPSILON_MAX = 0.85
PRIOR_EPSILON_MIN = 0.2
PRIOR_EPSILON_MAX = 0.8

CANONICAL_MU_MIN = 0.08
CANONICAL_MU_MAX = 0.40

SUMMARY_NAMES = (
    "effective_clusters_final",
    "opinion_entropy_final",
    "mean_opinion_shift",
    "late_opinion_variance",
    "abs_variance_change",
    "large_move_rate",
)
N_SUMMARIES = len(SUMMARY_NAMES)

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


def _is_degenerate_run(waves: list[np.ndarray], mu: float, epsilon: float) -> bool:
    if mu < CANONICAL_MU_MIN:
        return False
    total_movement = 0.0
    for left, right in zip(waves[:-1], waves[1:]):
        total_movement += float(np.mean(np.abs(right - left)))
    movement_floor = max(0.001, 0.05 * epsilon)
    return total_movement < movement_floor


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

    if _is_degenerate_run(waves, mu, epsilon):
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


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _logit_from_unit_interval(u: float) -> float:
    u = float(np.clip(u, 1e-6, 1.0 - 1e-6))
    return float(np.log(u / (1.0 - u)))


def _logit_from_epsilon(
    epsilon: float,
    eps_lo: float = TRAINING_EPSILON_MIN,
    eps_hi: float = TRAINING_EPSILON_MAX,
) -> float:
    frac = (epsilon - eps_lo) / (eps_hi - eps_lo)
    return _logit_from_unit_interval(frac)


def _epsilon_from_logit(
    logit_epsilon: float,
    eps_lo: float = TRAINING_EPSILON_MIN,
    eps_hi: float = TRAINING_EPSILON_MAX,
) -> float:
    return eps_lo + (eps_hi - eps_lo) * _sigmoid(logit_epsilon)


LOGIT_EPSILON_PRIOR_BOUNDS = (
    _logit_from_epsilon(PRIOR_EPSILON_MIN),
    _logit_from_epsilon(PRIOR_EPSILON_MAX),
)

PRIOR_PARAM_BOUNDS = (LOGIT_EPSILON_PRIOR_BOUNDS, (-4.0, 4.0))


def _to_epsilon_mu(logit_epsilon: float, logit_mu: float) -> tuple[float, float]:
    epsilon = _epsilon_from_logit(logit_epsilon)
    mu_span = CANONICAL_MU_MAX - CANONICAL_MU_MIN
    mu = CANONICAL_MU_MIN + mu_span * _sigmoid(logit_mu)
    return epsilon, mu


def to_canonical(params: np.ndarray) -> tuple[float, float]:
    return _to_epsilon_mu(float(params[0]), float(params[1]))


def draw_training_logit_parameters(rng: np.random.Generator) -> np.ndarray:
    epsilon = rng.uniform(TRAINING_EPSILON_MIN, TRAINING_EPSILON_MAX)
    logit_eps = _logit_from_epsilon(epsilon)
    lo_mu, hi_mu = PARAM_BOUNDS[1]
    logit_mu = rng.uniform(lo_mu, hi_mu)
    return np.array([logit_eps, logit_mu], dtype=np.float64)


def simulate_summaries(params: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
    logit_epsilon, logit_mu = map(float, params)
    epsilon, mu = _to_epsilon_mu(logit_epsilon, logit_mu)
    if n_trials < N_AGENTS:
        return np.full(N_SUMMARIES, np.nan)

    waves = _simulate_opinion_waves(epsilon, mu, n_trials, seed)
    if waves is None:
        return np.full(N_SUMMARIES, np.nan)

    summaries = _summaries_from_waves(waves)
    if not np.all(np.isfinite(summaries)):
        return np.full(N_SUMMARIES, np.nan)
    return summaries


def _dunif(lo: float, hi: float) -> str:
    return f"dunif({round(lo, 4)}, {round(hi, 4)})"


RECOVERY_PRIORS = {
    "logit_epsilon": f"logit_epsilon ~ {_dunif(*LOGIT_EPSILON_PRIOR_BOUNDS)}",
    "logit_mu": "logit_mu ~ dunif(-4, 4)",
}


def build_jags_likelihood(obs: dict) -> list[str]:
    del obs
    return build_sl_likelihood_line("dw", PARAM_NAMES, N_SUMMARIES)


DW = Model(
    slug="dw",
    param_names=PARAM_NAMES,
    param_bounds=PARAM_BOUNDS,
    prior_bounds=PRIOR_PARAM_BOUNDS,
    summary_names=SUMMARY_NAMES,
    draw_cov_parameters=draw_training_logit_parameters,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, SUMMARY_NAMES),
    simulate_summaries=simulate_summaries,
    recovery_priors=RECOVERY_PRIORS,
    build_jags_likelihood=build_jags_likelihood,
    default_architecture="DeepWide_32x6",
    default_n_epochs=10000,
)

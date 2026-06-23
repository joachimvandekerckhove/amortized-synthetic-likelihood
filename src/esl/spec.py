"""
esl.spec -- Model specification dataclass.

Defines the Model contract that every cognitive model in the ESL pipeline must
satisfy.  Each model provides parameter names/bounds, summary-statistic names,
a simulator, and optional recovery hooks for the JAGS likelihood.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

SimulateFn = Callable[[np.ndarray, int, int], np.ndarray]
PrecisionFn = Callable[[np.ndarray, int, int], dict]
TauFlatFn = Callable[[np.ndarray, int, int], np.ndarray]
JagsLinesFn = Callable[[dict], list[str]]
JagsDataFn = Callable[[dict], dict]
MleTauWeightsFn = Callable[[dict], np.ndarray]
MleNegLogLikFn = Callable[[dict, np.ndarray], float]
RtVarLikelihood = Literal["normal", "gamma"]


@dataclass(frozen=True)
class Model:
    """Specification for a cognitive model in the ESL pipeline.

    Attributes
    ----------
    slug : str
        Short identifier used in file paths (e.g., "ddm3", "ddm4").
    param_names : tuple[str, ...]
        Ordered names of the free parameters.
    param_bounds : tuple[tuple[float, float], ...]
        (lower, upper) bounds for each parameter, same order as param_names.
    summary_names : tuple[str, ...]
        Ordered names of the output summary statistics.
    simulate_summaries : Callable
        Function with signature (params, n_trials, seed) -> summaries array.
        Returns NaN entries when summaries are undefined for that draw.
    recovery_priors : dict
        JAGS prior specification strings keyed by parameter name.
    compute_precisions : Callable or None
        (params, n_trials, seed) -> dict of precision terms for recovery.
    compute_tau_flat : Callable or None
        (params, n_trials, seed) -> flat np.ndarray of raw tau values aligned
        with summary_names order.  Used by multivariate recovery to compute
        the per-subject sampling precision in standardized space.
    build_jags_likelihood : Callable or None
        (obs dict) -> list of JAGS model lines for the likelihood block.
    build_jags_data : Callable or None
        (obs dict) -> py2jags data dictionary for one subject.
    mle_tau_weights : Callable or None
        (obs dict) -> length-n_summaries weight vector for MLE warm-start.
        Zero for absent summaries (e.g. ddm4 corr/err blocks).
    rt_var_likelihood : str or None
        "normal" or "gamma" for models that support switchable rt_var likelihood.
    mle_neg_log_lik : Callable or None
        (obs, pred) -> negative log-likelihood for MLE warm-start when the
        observation model is not diagonal Gaussian (e.g. gamma rt_var).
    source_slug : str or None
        When set, training data is read from data/<source_slug>/ instead of
        data/<slug>/.
    emulator_output_names : tuple[str, ...] or None
        When set (multivariate models), lists all M ONNX outputs (mu + chol).
    default_architecture : str or None
        When set, skip architecture search and train this catalogue entry.
        Overridden by ESL_ARCHITECTURE.
    default_n_epochs : int or None
        When set and ESL_N_EPOCHS is unset, use this epoch count for retraining.
    """

    slug: str
    param_names: tuple[str, ...]
    param_bounds: tuple[tuple[float, float], ...]
    summary_names: tuple[str, ...]
    simulate_summaries: SimulateFn
    recovery_priors: dict = field(default_factory=dict)
    compute_precisions: PrecisionFn | None = None
    compute_tau_flat: TauFlatFn | None = None
    build_jags_likelihood: JagsLinesFn | None = None
    build_jags_data: JagsDataFn | None = None
    mle_tau_weights: MleTauWeightsFn | None = None
    rt_var_likelihood: RtVarLikelihood | None = None
    mle_neg_log_lik: MleNegLogLikFn | None = None
    source_slug: str | None = None
    emulator_output_names: tuple[str, ...] | None = None
    default_architecture: str | None = None
    default_n_epochs: int | None = None

    @property
    def n_params(self) -> int:
        """Number of free parameters."""
        return len(self.param_names)

    @property
    def n_summaries(self) -> int:
        """Number of summary statistics."""
        return len(self.summary_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        """ONNX output names (defaults to summary_names for Phase 1 models)."""
        return self.emulator_output_names or self.summary_names

    @property
    def n_outputs(self) -> int:
        """Number of ONNX outputs."""
        return len(self.output_names)

    def supports_recovery(self) -> bool:
        """True when all recovery hooks are defined."""
        return (
            self.compute_precisions is not None
            and self.build_jags_likelihood is not None
            and self.build_jags_data is not None
        )

    def supports_mv_recovery(self) -> bool:
        """True when multivariate recovery hooks are defined."""
        return (
            self.emulator_output_names is not None
            and self.build_jags_likelihood is not None
        )

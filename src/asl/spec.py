"""
asl.spec -- Model specification dataclass.

Each model provides parameter names/bounds, summary-statistic names,
a simulator, and hooks for JAGS synthetic-likelihood recovery.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

SimulateFn = Callable[[np.ndarray, int, int], np.ndarray]
DrawCovFn = Callable[[np.random.Generator], np.ndarray]
JagsLinesFn = Callable[[dict], list[str]]

SummaryTransform = Literal["log1p", "identity"]
_VALID_TRANSFORMS = frozenset({"log1p", "identity"})


@dataclass(frozen=True)
class Model:
    """Specification for a cognitive model in the ASL pipeline.

    param_bounds
        Training / emulator input domain (cov_data draws, wire ONNX limits).
    prior_bounds
        Recovery subject draws and chain initial-value support.
    summary_transforms
        Per-summary target transform before joint standardization.
    """

    slug: str
    param_names: tuple[str, ...]
    param_bounds: tuple[tuple[float, float], ...]
    prior_bounds: tuple[tuple[float, float], ...]
    summary_names: tuple[str, ...]
    summary_transforms: tuple[SummaryTransform, ...]
    simulate_summaries: SimulateFn
    draw_cov_parameters: DrawCovFn | None = None
    recovery_priors: dict = field(default_factory=dict)
    build_jags_likelihood: JagsLinesFn | None = None
    emulator_output_names: tuple[str, ...] | None = None
    default_architecture: str | None = None

    def __post_init__(self) -> None:
        if len(self.prior_bounds) != len(self.param_names):
            raise ValueError(
                f"Model '{self.slug}': prior_bounds length "
                f"{len(self.prior_bounds)} != n_params {len(self.param_names)}"
            )
        if len(self.summary_transforms) != len(self.summary_names):
            raise ValueError(
                f"Model '{self.slug}': summary_transforms length "
                f"{len(self.summary_transforms)} != n_summaries "
                f"{len(self.summary_names)}"
            )
        invalid = set(self.summary_transforms) - _VALID_TRANSFORMS
        if invalid:
            raise ValueError(
                f"Model '{self.slug}': invalid summary_transforms {invalid}"
            )

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def n_summaries(self) -> int:
        return len(self.summary_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.emulator_output_names or self.summary_names

    @property
    def n_outputs(self) -> int:
        return len(self.output_names)

    def supports_recovery(self) -> bool:
        """True when synthetic-likelihood recovery hooks are defined."""
        return (
            self.emulator_output_names is not None
            and self.build_jags_likelihood is not None
        )

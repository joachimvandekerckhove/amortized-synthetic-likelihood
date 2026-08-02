"""
asl.spec -- Model specification dataclass.

Each model provides parameter names/bounds, summary-statistic names,
a simulator, and hooks for JAGS synthetic-likelihood recovery.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

SimulateFn = Callable[[np.ndarray, int, int], np.ndarray]
JagsLinesFn = Callable[[dict], list[str]]


@dataclass(frozen=True)
class Model:
    """Specification for a cognitive model in the ASL pipeline."""

    slug: str
    param_names: tuple[str, ...]
    param_bounds: tuple[tuple[float, float], ...]
    summary_names: tuple[str, ...]
    simulate_summaries: SimulateFn
    recovery_priors: dict = field(default_factory=dict)
    build_jags_likelihood: JagsLinesFn | None = None
    emulator_output_names: tuple[str, ...] | None = None
    default_architecture: str | None = None
    default_n_epochs: int | None = None

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

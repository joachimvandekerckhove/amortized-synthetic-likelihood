"""Multivariate four-parameter DDM emulator."""

from esl.mv import build_mv_jags_likelihood_lines, emulator_output_names_for
from esl.spec import Model
from models.ddm.ddm4 import DDM4, simulate_summaries_ddm4


def build_ddm4mv_jags_likelihood(obs: dict) -> list[str]:
    """Return JAGS likelihood lines for ddm4mv recovery."""
    return build_mv_jags_likelihood_lines(DDM4.n_summaries)


N_SUMMARIES = DDM4.n_summaries

DDM4MV = Model(
    slug="ddm4mv",
    source_slug="ddm4",
    param_names=DDM4.param_names,
    param_bounds=DDM4.param_bounds,
    summary_names=DDM4.summary_names,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, DDM4.summary_names),
    simulate_summaries=simulate_summaries_ddm4,
    recovery_priors=DDM4.recovery_priors,
    build_jags_likelihood=build_ddm4mv_jags_likelihood,
    default_architecture="DeepWide_32x6",
    default_n_epochs=10000,
)

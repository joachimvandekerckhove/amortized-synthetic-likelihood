"""Multivariate DDM emulator models."""

from asl.mv import build_mv_jags_likelihood_for_model, emulator_output_names_for
from asl.spec import Model
from models.ddm.ddm3 import DDM3, simulate_summaries_ddm3


def build_ddm3mv_jags_likelihood(obs: dict) -> list[str]:
    """Return JAGS likelihood lines for ddm3mv recovery."""
    return build_mv_jags_likelihood_for_model(
        "ddm3mv", DDM3.param_names, N_SUMMARIES, obs
    )


N_SUMMARIES = DDM3.n_summaries

DDM3MV = Model(
    slug="ddm3mv",
    source_slug="ddm3",
    param_names=DDM3.param_names,
    param_bounds=DDM3.param_bounds,
    summary_names=DDM3.summary_names,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, DDM3.summary_names),
    simulate_summaries=simulate_summaries_ddm3,
    recovery_priors=DDM3.recovery_priors,
    build_jags_likelihood=build_ddm3mv_jags_likelihood,
    default_architecture="DeepWide_24x4",
    default_n_epochs=10000,
)

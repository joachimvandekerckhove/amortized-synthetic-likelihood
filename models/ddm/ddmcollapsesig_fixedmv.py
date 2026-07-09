"""Multivariate emulator for the fixed-bound ddmcollapsesig condition."""

from asl.mv import build_sl_likelihood_line, emulator_output_names_for
from asl.spec import Model
from models.ddm.ddmcollapsesig import (
    N_SUMMARIES,
    PARAM_BOUNDS,
    PARAM_NAMES,
    RECOVERY_PRIORS,
    SUMMARY_NAMES,
    simulate_summaries_fixed,
)


def build_ddmcollapsesig_fixedmv_jags_likelihood(obs: dict) -> list[str]:
    """Return JAGS likelihood lines for fixed-condition joint recovery."""
    return build_sl_likelihood_line(
        "ddmcollapsesig_fixed",
        PARAM_NAMES,
        N_SUMMARIES,
        obs_name="obs_std",
        n_trials_name="n_trials",
    )


DDMCOLLAPSESIG_FIXEDMV = Model(
    slug="ddmcollapsesig_fixed",
    param_names=PARAM_NAMES,
    param_bounds=PARAM_BOUNDS,
    summary_names=SUMMARY_NAMES,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, SUMMARY_NAMES),
    simulate_summaries=simulate_summaries_fixed,
    recovery_priors=RECOVERY_PRIORS,
    build_jags_likelihood=build_ddmcollapsesig_fixedmv_jags_likelihood,
    default_architecture="DeepWide_32x6",
    default_n_epochs=10000,
)

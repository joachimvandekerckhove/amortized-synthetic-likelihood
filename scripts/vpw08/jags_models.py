"""JAGS model strings for VPW08 hierarchical fits (JNNX v2 synthetic likelihood)."""

from __future__ import annotations

DDM3_SLUG = "ddm3"
DDM4_SLUG = "ddm4"
COLLAPSE_SLUG = "ddmcollapsesig"

DDM3_N_SUMM = 3
DDM4_N_SUMM = 5
COLLAPSE_N_SUMM = 10


def _metaregression_lines() -> list[str]:
    return [
        "    drift_mu ~ dnorm(0, 1)",
        "    drift_lambda ~ dgamma(2, 1)",
        "    for (i in 1:4) {",
        "        gamma[i] ~ dnorm(0, 1)",
        "    }",
        "    for (j in 1:5) {",
        "        drift_pred[j] <- drift_mu + A[j] * (gamma[1] * B[j] + gamma[2] * C[j] "
        "+ gamma[3] * B[j] * C[j]) + (1 - A[j]) * gamma[4]",
        "    }",
    ]


def _sl_block(
    slug: str,
    n_summ: int,
    param_args: str,
    n_trials_expr: str,
    *,
    include_ppc: bool,
) -> list[str]:
    lines = [
        f"        obs[cell,1:{n_summ}] ~ {slug}_sl({param_args}, {n_trials_expr})"
    ]
    if include_ppc:
        lines.append(
            f"        obs_rep[cell,1:{n_summ}] ~ {slug}_sl({param_args}, {n_trials_expr})"
        )
    return lines


def build_ddm3_ezmatched_model(*, include_ppc: bool = False) -> str:
    """EZ-matched VPW08 hierarchy with ddm3 synthetic likelihood."""
    lines = [
        "model {",
        *_metaregression_lines(),
        "    for (cell in 1:K) {",
        "        bound[cell] ~ dgamma(2, 1) T(0.501, 1.999)",
        "        nondt[cell] ~ dexp(1) T(0.151, 0.449)",
        "        drift[cell] ~ dnorm(drift_pred[cond[cell]], drift_lambda) T(-1.999, 1.999)",
    ]
    lines.extend(
        _sl_block(
            DDM3_SLUG,
            DDM3_N_SUMM,
            "drift[cell], bound[cell], nondt[cell]",
            "n_trials[cell]",
            include_ppc=include_ppc,
        )
    )
    lines.extend(["    }", "}"])
    return "\n".join(lines)


def build_ddm4_model(*, include_ppc: bool = False) -> str:
    """EZ metaregression with shared starting bias and ddm4 synthetic likelihood."""
    lines = [
        "model {",
        *_metaregression_lines(),
        "    w ~ dunif(0.15, 0.85)",
        "    for (cell in 1:K) {",
        "        a[cell] ~ dunif(0.5, 2.0)",
        "        t0[cell] ~ dunif(0.15, 0.45)",
        "        v[cell] ~ dnorm(drift_pred[cond[cell]], drift_lambda) T(-1.999, 1.999)",
    ]
    lines.extend(
        _sl_block(
            DDM4_SLUG,
            DDM4_N_SUMM,
            "v[cell], a[cell], t0[cell], w",
            "n_trials[cell]",
            include_ppc=include_ppc,
        )
    )
    lines.extend(["    }", "}"])
    return "\n".join(lines)


def build_collapse_delta_kappa_model(*, include_ppc: bool = False) -> str:
    """Collapsing-bound VPW08 with shared kappa in change conditions."""
    lines = [
        "model {",
        *_metaregression_lines(),
        "    kappa_change ~ dunif(0, 8.0)",
        "    kappa_nochange ~ dunif(0, 8.0)",
        "    delta_kappa <- kappa_nochange - kappa_change",
        "    for (cell in 1:K) {",
        "        bound[cell] ~ dgamma(2, 1) T(0.501, 1.999)",
        "        nondt[cell] ~ dexp(1) T(0.051, 0.449)",
        "        drift[cell] ~ dnorm(drift_pred[cond[cell]], drift_lambda) T(-1.499, 1.499)",
        "        kappa[cell] <- A[cond[cell]] * kappa_change + (1 - A[cond[cell]]) * kappa_nochange",
    ]
    lines.extend(
        _sl_block(
            COLLAPSE_SLUG,
            COLLAPSE_N_SUMM,
            "bound[cell], drift[cell], kappa[cell], nondt[cell]",
            "n_trials[cell]",
            include_ppc=include_ppc,
        )
    )
    lines.extend(["    }", "}"])
    return "\n".join(lines)


def monitor_ddm3_ezmatched(*, include_ppc: bool = False) -> list[str]:
    base = [
        "drift_mu",
        "drift_lambda",
        "gamma",
        "drift_pred",
        "drift",
        "bound",
        "nondt",
        "deviance",
    ]
    if include_ppc:
        base.append("obs_rep")
    return base


def monitor_ddm4(*, include_ppc: bool = False) -> list[str]:
    base = [
        "drift_mu",
        "drift_lambda",
        "gamma",
        "w",
        "drift_pred",
        "v",
        "a",
        "t0",
        "deviance",
    ]
    if include_ppc:
        base.append("obs_rep")
    return base


def monitor_collapse_delta_kappa(*, include_ppc: bool = False) -> list[str]:
    base = [
        "drift_mu",
        "drift_lambda",
        "gamma",
        "kappa_change",
        "kappa_nochange",
        "delta_kappa",
        "drift_pred",
        "drift",
        "bound",
        "nondt",
        "deviance",
    ]
    if include_ppc:
        base.append("obs_rep")
    return base

"""
models.ddm.ddm4 -- Four-parameter drift diffusion model.

Parameters: drift rate (v), boundary separation (a), nondecision time (t0),
starting-point bias (w).
Summaries: rt_mean_corr, rt_var_corr, rt_mean_err, rt_var_err, err_rate.

Parameter ranges match the reference implementation to ensure numerically
well-behaved recovery.  Precision terms are capped at TAU_CAP = 2000.
"""

import numpy as np

from esl.spec import Model
from models.ddm.simulator import simulate_ddm_paths_biased

TAU_CAP = 2000.0


def simulate_summaries_ddm4(
    params: np.ndarray, n_trials: int, seed: int
) -> np.ndarray:
    """Simulate summary statistics for the 4-parameter DDM."""
    v, a, t0, w = (
        float(params[0]),
        float(params[1]),
        float(params[2]),
        float(params[3]),
    )

    reaction_times, choices = simulate_ddm_paths_biased(
        drift_rate=v,
        boundary_separation=a,
        nondecision_time=t0,
        starting_bias=w,
        n_samples=n_trials,
        seed=seed,
    )

    rts_correct = reaction_times[choices == 1]
    rts_error = reaction_times[choices == 0]
    n_total = len(reaction_times)

    if len(rts_correct) < 2 or len(rts_error) < 2:
        return np.full(5, np.nan)

    rt_mean_corr = float(np.mean(rts_correct))
    rt_var_corr = float(np.var(rts_correct, ddof=1))
    rt_mean_err = float(np.mean(rts_error))
    rt_var_err = float(np.var(rts_error, ddof=1))
    err_rate = float(len(rts_error) / n_total)

    return np.array([rt_mean_corr, rt_var_corr, rt_mean_err, rt_var_err, err_rate])


def compute_ddm4_precisions(params: np.ndarray, n_trials: int, seed: int) -> dict:
    """Compute precision terms for ddm4 observed summaries."""
    v, a, t0, w = float(params[0]), float(params[1]), float(params[2]), float(params[3])
    rts, choices = simulate_ddm_paths_biased(v, a, t0, w, n_trials, seed)

    rts_corr = rts[choices == 1]
    rts_err = rts[choices == 0]
    n_corr = len(rts_corr)
    n_err = len(rts_err)
    n_total = n_corr + n_err

    has_corr = n_corr >= 2
    has_err = n_err >= 2

    # Jeffreys-style continuity correction: add 1/2 error and 1/2 correct.
    p_err = (n_err + 0.5) / (n_total + 1.0)
    tau_err_rate = min(n_total / (p_err * (1.0 - p_err)), TAU_CAP)

    tau_corr = [0.0, 0.0]
    if has_corr:
        var_corr = float(np.var(rts_corr, ddof=1))
        tau_corr = [
            min(n_corr / max(var_corr, 1e-6), TAU_CAP),
            min((n_corr - 1) ** 2 / max(2.0 * var_corr ** 2, 1e-6), TAU_CAP),
        ]

    tau_err = [0.0, 0.0]
    if has_err:
        var_err = float(np.var(rts_err, ddof=1))
        tau_err = [
            min(n_err / max(var_err, 1e-6), TAU_CAP),
            min((n_err - 1) ** 2 / max(2.0 * var_err ** 2, 1e-6), TAU_CAP),
        ]

    return {
        "tau_corr": tau_corr,
        "tau_err": tau_err,
        "tau_err_rate": float(tau_err_rate),
        "has_corr": has_corr,
        "has_err": has_err,
    }


def mle_tau_weights_ddm4(obs: dict) -> np.ndarray:
    """Length-5 MLE weight vector aligned with summary_names order."""
    tau = obs["tau"]
    weights = np.zeros(5, dtype=float)
    if tau["has_corr"]:
        weights[0], weights[1] = tau["tau_corr"]
    if tau["has_err"]:
        weights[2], weights[3] = tau["tau_err"]
    weights[4] = tau["tau_err_rate"]
    return weights


def build_ddm4_jags_likelihood(obs: dict) -> list[str]:
    """Return JAGS likelihood lines for ddm4 recovery."""
    lines = ["obs_err_rate ~ dnorm(pred[5], tau_err_rate)"]
    tau = obs["tau"]
    if tau["has_corr"]:
        lines.append("obs_rt_mean_corr ~ dnorm(pred[1], tau_corr[1])")
        lines.append("obs_rt_var_corr ~ dnorm(pred[2], tau_corr[2])")
    if tau["has_err"]:
        lines.append("obs_rt_mean_err ~ dnorm(pred[3], tau_err[1])")
        lines.append("obs_rt_var_err ~ dnorm(pred[4], tau_err[2])")
    return lines


def build_ddm4_jags_data(obs: dict) -> dict:
    """Build py2jags data dict for one ddm4 subject."""
    summaries = obs["summaries"]
    tau = obs["tau"]
    data = {
        "obs_err_rate": float(summaries[4]),
        "tau_err_rate": tau["tau_err_rate"],
    }
    if tau["has_corr"]:
        data["tau_corr"] = tau["tau_corr"]
        data["obs_rt_mean_corr"] = float(summaries[0])
        data["obs_rt_var_corr"] = float(summaries[1])
    if tau["has_err"]:
        data["tau_err"] = tau["tau_err"]
        data["obs_rt_mean_err"] = float(summaries[2])
        data["obs_rt_var_err"] = float(summaries[3])
    return data


DDM4 = Model(
    slug="ddm4",
    param_names=("v", "a", "t0", "w"),
    param_bounds=((-2.0, 2.0), (0.5, 2.0), (0.15, 0.45), (0.15, 0.85)),
    summary_names=("rt_mean_corr", "rt_var_corr", "rt_mean_err", "rt_var_err", "err_rate"),
    simulate_summaries=simulate_summaries_ddm4,
    recovery_priors={
        "v": "v ~ dnorm(0, 0.25)",
        "a": "a ~ dunif(0.25, 3.0)",
        "t0": "t0 ~ dunif(0.1, 0.6)",
        "w": "w ~ dbeta(2, 2)",
    },
    compute_precisions=compute_ddm4_precisions,
    build_jags_likelihood=build_ddm4_jags_likelihood,
    build_jags_data=build_ddm4_jags_data,
    mle_tau_weights=mle_tau_weights_ddm4,
)

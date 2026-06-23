"""
models.ddm.ddm3 -- Three-parameter drift diffusion model.

Parameters: drift rate (v), boundary separation (a), nondecision time (t0).
Starting bias is fixed at w = 0.5 (unbiased).
Summaries: accuracy (proportion correct), mean RT, variance of RT.

Parameter ranges match the reference implementation (Chávez De la Peña &
Vandekerckhove 2025) to ensure numerically well-behaved recovery:
  v  ∈ (-2, 2)    -- avoids near-0% / near-100% accuracy extremes
  a  ∈ (0.5, 2.0) -- realistic boundary separations
  t0 ∈ (0.15, 0.45) -- typical nondecision times in cognitive experiments

Recovery uses normal likelihoods for acc and rt_mean.  For rt_var, either a
normal proxy (asymptotic) or an exact gamma likelihood for the sample variance
is available via rt_var_likelihood / ESL_RT_VAR_LIKELIHOOD.
"""

import math

import numpy as np

from esl.spec import Model
from models.ddm.simulator import simulate_ddm_paths_biased

FIXED_BIAS = 0.5
RT_VAR_LIKELIHOODS = ("normal", "gamma")


def simulate_summaries_ddm3(
    params: np.ndarray, n_trials: int, seed: int
) -> np.ndarray:
    """Simulate summary statistics for the 3-parameter DDM."""
    v, a, t0 = float(params[0]), float(params[1]), float(params[2])

    reaction_times, choices = simulate_ddm_paths_biased(
        drift_rate=v,
        boundary_separation=a,
        nondecision_time=t0,
        starting_bias=FIXED_BIAS,
        n_samples=n_trials,
        seed=seed,
    )

    n_absorbed = len(reaction_times)
    if n_absorbed < 2:
        return np.array([np.nan, np.nan, np.nan])

    accuracy = float(np.mean(choices == 1))
    rt_mean = float(np.mean(reaction_times))
    rt_var = float(np.var(reaction_times, ddof=1))

    return np.array([accuracy, rt_mean, rt_var])


def compute_ddm3_precisions(params: np.ndarray, n_trials: int, seed: int) -> dict:
    """Compute precision terms for ddm3 observed summaries.

    Formulas follow the reference (Chávez De la Peña & Vandekerckhove 2025):
      tau_acc     = n / (p * (1-p))              -- binomial Fisher information
      tau_rt_mean = n / var(RT)                  -- SE of mean
      tau_rt_var  = (n-1) / (2 * var(RT)^2)     -- normal proxy for sample variance

    Also returns n_rt (absorbed trial count) for the gamma rt_var likelihood.
    """
    v, a, t0 = float(params[0]), float(params[1]), float(params[2])
    rts, choices = simulate_ddm_paths_biased(v, a, t0, FIXED_BIAS, n_trials, seed)

    n = len(rts)
    if n < 2:
        return {
            "tau_acc": 1.0,
            "tau_rt_mean": 1.0,
            "tau_rt_var": 1.0,
            "n_rt": n,
        }

    acc = float(np.mean(choices == 1))
    rt_var = float(np.var(rts, ddof=1))

    acc_denom = max(acc * (1.0 - acc), 1e-4)

    return {
        "tau_acc": float(n / acc_denom),
        "tau_rt_mean": float(n / max(rt_var, 1e-6)),
        "tau_rt_var": float((n - 1) / max(2.0 * rt_var ** 2, 1e-6)),
        "n_rt": n,
    }


def ddm3_mle_neg_log_lik(obs: dict, pred: np.ndarray) -> float:
    """Negative log-likelihood for MLE warm-start with gamma rt_var."""
    summaries = obs["summaries"]
    tau = obs["tau"]
    n_rt = int(tau["n_rt"])
    obs_rt_var = max(float(summaries[2]), 1e-12)
    sigma_sq = float(pred[2])
    if sigma_sq <= 0 or n_rt < 2:
        return 1e12

    nll = 0.5 * tau["tau_acc"] * (pred[0] - summaries[0]) ** 2
    nll += 0.5 * tau["tau_rt_mean"] * (pred[1] - summaries[1]) ** 2

    shape = (n_rt - 1) / 2.0
    rate = (n_rt - 1) / (2.0 * sigma_sq)
    log_pdf = (
        shape * math.log(rate)
        - math.lgamma(shape)
        + (shape - 1) * math.log(obs_rt_var)
        - rate * obs_rt_var
    )
    return float(nll - log_pdf)


def build_ddm3_jags_likelihood(obs: dict) -> list[str]:
    """Return JAGS likelihood lines for ddm3 recovery."""
    mode = obs.get("rt_var_likelihood", "normal")
    lines = [
        "obs_acc ~ dnorm(pred[1], tau_acc)",
        "obs_rt_mean ~ dnorm(pred[2], tau_rt_mean)",
    ]
    if mode == "gamma":
        lines.extend([
            "rt_var_shape <- (n_rt - 1) / 2",
            "rt_var_rate <- (n_rt - 1) / (2 * ifelse(pred[3] > 1e-6, pred[3], 1e-6))",
            "obs_rt_var ~ dgamma(rt_var_shape, rt_var_rate)",
        ])
    else:
        lines.append("obs_rt_var ~ dnorm(pred[3], tau_rt_var)")
    return lines


def build_ddm3_jags_data(obs: dict) -> dict:
    """Build py2jags data dict for one ddm3 subject."""
    summaries = obs["summaries"]
    tau = obs["tau"]
    data = {
        "obs_acc": float(summaries[0]),
        "obs_rt_mean": float(summaries[1]),
        "obs_rt_var": max(float(summaries[2]), 1e-12),
        "tau_acc": tau["tau_acc"],
        "tau_rt_mean": tau["tau_rt_mean"],
        "n_rt": int(tau["n_rt"]),
    }
    if obs.get("rt_var_likelihood", "normal") == "normal":
        data["tau_rt_var"] = tau["tau_rt_var"]
    return data


DDM3 = Model(
    slug="ddm3",
    param_names=("v", "a", "t0"),
    param_bounds=((-2.0, 2.0), (0.5, 2.0), (0.15, 0.45)),
    summary_names=("acc", "rt_mean", "rt_var"),
    simulate_summaries=simulate_summaries_ddm3,
    recovery_priors={
        "v": "v ~ dnorm(0, 0.25)",
        "a": "a ~ dunif(0.3, 2.5)",
        "t0": "t0 ~ dunif(0.1, 0.5)",
    },
    compute_precisions=compute_ddm3_precisions,
    build_jags_likelihood=build_ddm3_jags_likelihood,
    build_jags_data=build_ddm3_jags_data,
    rt_var_likelihood="gamma",
    mle_neg_log_lik=ddm3_mle_neg_log_lik,
)

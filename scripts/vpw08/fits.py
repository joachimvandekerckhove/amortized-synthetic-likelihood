"""MCMC fits for the three VPW08 paper illustrations."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from py2jags import run_jags
from scipy.stats import norm

from data import build_cells
from jags_models import (
    DDM3_SLUG,
    DDM4_SLUG,
    COLLAPSE_SLUG,
    build_collapse_delta_kappa_model,
    build_ddm3_ezmatched_model,
    build_ddm4_model,
    monitor_collapse_delta_kappa,
    monitor_ddm3_ezmatched,
    monitor_ddm4,
)
from mcmc import (
    BF_EPS,
    N_BURNIN,
    N_CHAINS,
    N_ITER,
    SEED,
    _r_vector,
    bayes_factor_null,
    bayes_factors_gamma,
    convergence_block,
    ensure_jags_runtime,
    get_param_samples,
    hierarchical_data,
    summarize_gamma,
    summarize_param,
)
from paths import COLLAPSE_JSON, DDM3_JSON, DDM4_JSON, RESULTS_DIR
from ppc import extract_obs_rep, summarize_ppc

from models.ddm.ddm3 import DDM3
from models.ddm.ddm4 import DDM4


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _chain_inits_ddm3(k_cells: int, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    inits = []
    for _ in range(N_CHAINS):
        inits.append(
            {
                "drift": _r_vector(np.clip(rng.normal(0, 1, size=k_cells), -1.9, 1.9)),
                "bound": _r_vector(rng.uniform(0.8, 1.5, size=k_cells)),
                "nondt": _r_vector(rng.uniform(0.18, 0.35, size=k_cells)),
            }
        )
    return inits


def _chain_inits_ddm4(k_cells: int, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    inits = []
    for chain in range(N_CHAINS):
        inits.append(
            {
                "v": _r_vector(np.clip(rng.normal(0, 0.5, size=k_cells), -1.9, 1.9)),
                "a": _r_vector(rng.uniform(0.8, 1.5, size=k_cells)),
                "t0": _r_vector(rng.uniform(0.18, 0.35, size=k_cells)),
                "w": 0.45 + 0.02 * chain,
            }
        )
    return inits


def _chain_inits_collapse(k_cells: int, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    inits = []
    for _ in range(N_CHAINS):
        inits.append(
            {
                "drift": _r_vector(np.clip(rng.normal(0, 1, size=k_cells), -1.4, 1.4)),
                "bound": _r_vector(rng.uniform(0.8, 1.5, size=k_cells)),
                "nondt": _r_vector(rng.uniform(0.10, 0.35, size=k_cells)),
                "kappa_change": float(rng.uniform(0.5, 2.5)),
                "kappa_nochange": float(rng.uniform(0.5, 2.5)),
            }
        )
    return inits


def fit_ddm3_ezmatched() -> dict:
    ensure_jags_runtime()
    cells = build_cells(summary="ez")
    summary_names = list(DDM3.summary_names)
    n_summ = len(summary_names)
    data = hierarchical_data(cells)

    print(
        f"[vpw08:ddm3] K={cells['K']} iter={N_ITER} burnin={N_BURNIN} chains={N_CHAINS}",
        flush=True,
    )
    t0 = time.monotonic()
    result = run_jags(
        model_string=build_ddm3_ezmatched_model(include_ppc=True),
        data_dict=data,
        monitorparams=monitor_ddm3_ezmatched(include_ppc=True),
        nchains=N_CHAINS,
        nsamples=N_ITER,
        nburnin=N_BURNIN,
        thin=1,
        init=_chain_inits_ddm3(cells["K"]),
        modules=["dic", f"{DDM3_SLUG}_emulator"],
        parallel=True,
        maxcores=N_CHAINS,
        verbosity=0,
    )
    runtime_s = time.monotonic() - t0

    ppc_raw = extract_obs_rep(result, cells["K"], n_summ)
    ppc_summary = summarize_ppc(cells["obs_raw"], ppc_raw, summary_names=summary_names)

    nondt_draws = np.column_stack(
        [get_param_samples(result, "nondt", i) for i in range(1, cells["K"] + 1)]
    )
    bound_draws = np.column_stack(
        [get_param_samples(result, "bound", i) for i in range(1, cells["K"] + 1)]
    )

    out = {
        "model": "ddm3_ezmatched",
        "emulator": DDM3_SLUG,
        "model_note": (
            "EZ Appendix E metaregression with ddm3 synthetic likelihood "
            "on acc/meanRT/varRT summaries"
        ),
        "dataset": "vpw08_shape",
        "reference": "Chavez & Vandekerckhove (2025) Appendix E",
        "n_subjects": int(cells["N_SUBJ"]),
        "n_cells": int(cells["K"]),
        "mcmc": {
            "n_iter": N_ITER,
            "n_burnin": N_BURNIN,
            "n_chains": N_CHAINS,
            "thin": 1,
            "seed": SEED,
            "n_ppc_draws": int(ppc_raw.shape[0]),
        },
        "runtime_s": runtime_s,
        "convergence": convergence_block(result),
        "drift_mu": summarize_param(result, "drift_mu"),
        "drift_lambda": summarize_param(result, "drift_lambda"),
        "nondt_pooled_mean": {
            "mean": float(np.mean(nondt_draws)),
            "sd": float(np.std(nondt_draws)),
            "lo95": float(np.percentile(nondt_draws, 2.5)),
            "hi95": float(np.percentile(nondt_draws, 97.5)),
        },
        "bound_pooled_mean": {
            "mean": float(np.mean(bound_draws)),
            "sd": float(np.std(bound_draws)),
            "lo95": float(np.percentile(bound_draws, 2.5)),
            "hi95": float(np.percentile(bound_draws, 97.5)),
        },
        "gamma": summarize_gamma(result),
        "bayes_factors_gamma1_3": bayes_factors_gamma(result),
        "drift_pred": {
            str(j): summarize_param(result, "drift_pred", j) for j in range(1, 6)
        },
        "ppc_summary": ppc_summary,
    }

    _write_json(DDM3_JSON, out)
    np.savez_compressed(
        DDM3_JSON.with_suffix(".npz"),
        obs_raw=cells["obs_raw"],
        ppc_raw=ppc_raw,
        subj=cells["subj"],
        cond=cells["cond"],
        n_trials=cells["n_trials"],
        summary_names=np.array(summary_names),
    )
    print(
        f"[vpw08:ddm3] done runtime={runtime_s:.0f}s "
        f"max_rhat={out['convergence']['max_rhat']:.3f}",
        flush=True,
    )
    return out


def fit_ddm4() -> dict:
    ensure_jags_runtime()
    cells = build_cells(summary="ddm4")
    summary_names = list(DDM4.summary_names)
    n_summ = len(summary_names)
    data = hierarchical_data(cells)

    print(
        f"[vpw08:ddm4] K={cells['K']} iter={N_ITER} burnin={N_BURNIN} chains={N_CHAINS}",
        flush=True,
    )
    t0 = time.monotonic()
    result = run_jags(
        model_string=build_ddm4_model(include_ppc=True),
        data_dict=data,
        monitorparams=monitor_ddm4(include_ppc=True),
        nchains=N_CHAINS,
        nsamples=N_ITER,
        nburnin=N_BURNIN,
        thin=1,
        init=_chain_inits_ddm4(cells["K"]),
        modules=["dic", f"{DDM4_SLUG}_emulator"],
        parallel=True,
        maxcores=N_CHAINS,
        verbosity=0,
    )
    runtime_s = time.monotonic() - t0

    ppc_raw = extract_obs_rep(result, cells["K"], n_summ)
    ppc_summary = summarize_ppc(cells["obs_raw"], ppc_raw, summary_names=summary_names)

    out = {
        "model": "ddm4_hier",
        "emulator": DDM4_SLUG,
        "model_note": (
            "EZ metaregression with shared starting bias w and ddm4 synthetic likelihood"
        ),
        "dataset": "vpw08_shape",
        "reference": "Vandekerckhove, Panis, & Wagemans (2007); EZ Appendix E",
        "n_subjects": int(cells["N_SUBJ"]),
        "n_cells": int(cells["K"]),
        "mcmc": {
            "n_iter": N_ITER,
            "n_burnin": N_BURNIN,
            "n_chains": N_CHAINS,
            "thin": 1,
            "seed": SEED,
            "n_ppc_draws": int(ppc_raw.shape[0]),
        },
        "runtime_s": runtime_s,
        "convergence": convergence_block(result),
        "drift_mu": summarize_param(result, "drift_mu"),
        "drift_lambda": summarize_param(result, "drift_lambda"),
        "w": summarize_param(result, "w"),
        "gamma": summarize_gamma(result),
        "bayes_factors_gamma1_3": bayes_factors_gamma(result),
        "drift_pred": {
            str(j): summarize_param(result, "drift_pred", j) for j in range(1, 6)
        },
        "ppc_summary": ppc_summary,
    }

    _write_json(DDM4_JSON, out)
    np.savez_compressed(
        DDM4_JSON.with_suffix(".npz"),
        obs_raw=cells["obs_raw"],
        ppc_raw=ppc_raw,
        subj=cells["subj"],
        cond=cells["cond"],
        n_trials=cells["n_trials"],
        summary_names=np.array(summary_names),
    )
    print(
        f"[vpw08:ddm4] done runtime={runtime_s:.0f}s "
        f"max_rhat={out['convergence']['max_rhat']:.3f} "
        f"w={out['w']['mean']:.3f}",
        flush=True,
    )
    return out


def _bayes_factor_delta_zero(delta_draws: np.ndarray, epsilon: float = BF_EPS) -> float:
    prior_null_mass = norm.cdf(epsilon) - norm.cdf(-epsilon)
    post_null_mass = float(np.mean((delta_draws > -epsilon) & (delta_draws < epsilon)))
    if post_null_mass <= 0:
        return float("inf")
    return prior_null_mass / post_null_mass


def fit_collapse_delta_kappa() -> dict:
    ensure_jags_runtime()
    cells = build_cells(summary="collapse")
    summary_names = cells["summary_names"]
    n_summ = len(summary_names)
    data = hierarchical_data(cells)

    print(
        f"[vpw08:collapse] K={cells['K']} iter={N_ITER} burnin={N_BURNIN} "
        f"chains={N_CHAINS}",
        flush=True,
    )
    t0 = time.monotonic()
    result = run_jags(
        model_string=build_collapse_delta_kappa_model(include_ppc=True),
        data_dict=data,
        monitorparams=monitor_collapse_delta_kappa(include_ppc=True),
        nchains=N_CHAINS,
        nsamples=N_ITER,
        nburnin=N_BURNIN,
        thin=1,
        init=_chain_inits_collapse(cells["K"]),
        modules=["dic", f"{COLLAPSE_SLUG}_emulator"],
        parallel=True,
        maxcores=N_CHAINS,
        verbosity=0,
    )
    runtime_s = time.monotonic() - t0

    ppc_raw = extract_obs_rep(result, cells["K"], n_summ)
    ppc_summary = summarize_ppc(cells["obs_raw"], ppc_raw, summary_names=summary_names)

    delta_draws = get_param_samples(result, "delta_kappa")
    nondt_draws = np.column_stack(
        [get_param_samples(result, "nondt", i) for i in range(1, cells["K"] + 1)]
    )
    bound_draws = np.column_stack(
        [get_param_samples(result, "bound", i) for i in range(1, cells["K"] + 1)]
    )

    out = {
        "model": "collapse_delta_kappa",
        "emulator": COLLAPSE_SLUG,
        "model_note": (
            "Collapsing-bound emulator with shared kappa_change across change-present "
            "conditions and separate kappa_nochange; inference targets delta_kappa"
        ),
        "dataset": "vpw08_shape",
        "reference": "Vandekerckhove, Panis, & Wagemans (2007); EZ Appendix E",
        "n_subjects": int(cells["N_SUBJ"]),
        "n_cells": int(cells["K"]),
        "mcmc": {
            "n_iter": N_ITER,
            "n_burnin": N_BURNIN,
            "n_chains": N_CHAINS,
            "thin": 1,
            "seed": SEED,
            "n_ppc_draws": int(ppc_raw.shape[0]),
        },
        "runtime_s": runtime_s,
        "convergence": convergence_block(result),
        "drift_mu": summarize_param(result, "drift_mu"),
        "drift_lambda": summarize_param(result, "drift_lambda"),
        "kappa_change": summarize_param(result, "kappa_change"),
        "kappa_nochange": summarize_param(result, "kappa_nochange"),
        "delta_kappa": summarize_param(result, "delta_kappa"),
        "bayes_factor_delta_kappa_zero": float(_bayes_factor_delta_zero(delta_draws)),
        "nondt_pooled_mean": {
            "mean": float(np.mean(nondt_draws)),
            "sd": float(np.std(nondt_draws)),
            "lo95": float(np.percentile(nondt_draws, 2.5)),
            "hi95": float(np.percentile(nondt_draws, 97.5)),
        },
        "bound_pooled_mean": {
            "mean": float(np.mean(bound_draws)),
            "sd": float(np.std(bound_draws)),
            "lo95": float(np.percentile(bound_draws, 2.5)),
            "hi95": float(np.percentile(bound_draws, 97.5)),
        },
        "gamma": summarize_gamma(result),
        "bayes_factors_gamma1_3": bayes_factors_gamma(result),
        "drift_pred": {
            str(j): summarize_param(result, "drift_pred", j) for j in range(1, 6)
        },
        "ppc_summary": ppc_summary,
    }

    _write_json(COLLAPSE_JSON, out)
    np.savez_compressed(
        COLLAPSE_JSON.with_suffix(".npz"),
        obs_raw=cells["obs_raw"],
        ppc_raw=ppc_raw,
        subj=cells["subj"],
        cond=cells["cond"],
        n_trials=cells["n_trials"],
        summary_names=np.array(summary_names),
    )
    print(
        f"[vpw08:collapse] done runtime={runtime_s:.0f}s "
        f"max_rhat={out['convergence']['max_rhat']:.3f} "
        f"delta_kappa={out['delta_kappa']['mean']:.3f}",
        flush=True,
    )
    return out

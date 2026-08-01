"""Multivariate simulate-and-recover study via py2jags."""

import json
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.optimize import minimize

from asl.config import load_config
from asl.data import load_target_transform
from asl.figures import plot_recovery_diagnostics
from asl.recovery import (
    check_coverage_gate,
    format_recovery_progress,
    recovery_report_interval,
    resolve_recovery_settings,
    resolve_recovery_workers,
)
from asl.registry import get_model
from asl.spec import Model

N_CHAINS = 4


def simulate_subject_observations_mv(
    model: Model,
    params: np.ndarray,
    n_trials: int,
    seed: int,
) -> dict:
    """Simulate one subject and return raw observed summaries for JAGS."""
    summaries = model.simulate_summaries(params, n_trials, seed)
    valid = bool(np.all(np.isfinite(summaries)))
    if not valid:
        return {"valid": False}

    return {"obs": summaries.reshape(-1), "valid": True}


def build_jags_model_string_mv(model: Model, obs: dict) -> str:
    """Build a JAGS model string for multivariate recovery."""
    if not model.supports_mv_recovery():
        raise ValueError(f"Model '{model.slug}' does not define mv recovery hooks.")

    priors = "\n    ".join(model.recovery_priors.values())
    lines = [
        "model {",
        f"    {priors}",
    ]
    lines.extend(f"    {line}" for line in model.build_jags_likelihood(obs))
    lines.append("}")
    return "\n".join(lines)


def compute_mle_initial_values_mv(
    model: Model, obs_std: np.ndarray, onnx_path: Path
) -> list[dict]:
    """Compute MLE-based initial values using the mean head in standardized space."""
    session = ort.InferenceSession(str(onnx_path))
    n = model.n_summaries

    def neg_log_lik(params: np.ndarray) -> float:
        for i, (lo, hi) in enumerate(model.param_bounds):
            if params[i] <= lo or params[i] >= hi:
                return 1e12
        x = params.astype(np.float32).reshape(1, -1)
        pred = session.run(None, {"input": x})[0][0]
        mu_std = pred[:n]
        return float(0.5 * np.sum((mu_std - obs_std) ** 2))

    if model.n_params <= 3:
        grid_vals = [
            np.linspace(lo + (hi - lo) * 0.2, hi - (hi - lo) * 0.2, 3)
            for lo, hi in model.param_bounds
        ]
        grid_starts = [np.array(pt) for pt in product(*grid_vals)]
    else:
        v_lo, v_hi = model.param_bounds[0]
        t0_lo, t0_hi = model.param_bounds[2]
        v_vals = np.linspace(v_lo + 0.2 * (v_hi - v_lo), v_hi - 0.2 * (v_hi - v_lo), 3)
        a_vals = np.array([0.8, 1.5])
        t0_vals = np.array([(t0_lo + t0_hi) / 2.0])
        w_vals = np.array([0.3, 0.5, 0.7])
        grid_starts = [
            np.array([v, a, t0, w])
            for v, a, t0, w in product(v_vals, a_vals, t0_vals, w_vals)
        ]

    best_result = None
    for x0 in grid_starts:
        res = minimize(
            neg_log_lik,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6},
        )
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    mle = best_result.x  # type: ignore[union-attr]
    jitter_scale = np.array([(hi - lo) * 0.02 for lo, hi in model.param_bounds])

    rng = np.random.default_rng(99)
    inits = []
    for _ in range(N_CHAINS):
        jittered = {}
        for i, name in enumerate(model.param_names):
            lo, hi = model.param_bounds[i]
            val = float(mle[i]) + rng.normal(0, jitter_scale[i])
            jittered[name] = float(np.clip(val, lo + 1e-4, hi - 1e-4))
        inits.append(jittered)
    return inits


def _recover_one_subject_mv(args: tuple) -> dict | None:
    """Worker: run multivariate MCMC for one subject."""
    slug, true_params, subj_seed, settings, onnx_path_str = args
    from py2jags import run_jags

    model = get_model(slug)
    target_transform = load_target_transform(slug)
    onnx_path = Path(onnx_path_str)
    module_name = f"{model.slug}_emulator"

    obs = simulate_subject_observations_mv(
        model, true_params, settings["n_trials"], subj_seed
    )
    if not obs["valid"]:
        return None

    model_string = build_jags_model_string_mv(model, obs)
    obs_raw = np.asarray(obs["obs"], dtype=np.float64)
    obs_std = target_transform.transform(obs_raw.reshape(1, -1))[0]
    data = {
        "obs": obs_raw.tolist(),
        "n_trials": settings["n_trials"],
    }
    inits = compute_mle_initial_values_mv(model, obs_std, onnx_path)

    try:
        result = run_jags(
            model_string=model_string,
            data_dict=data,
            monitorparams=list(model.param_names),
            nchains=settings["n_chains"],
            nsamples=settings["n_iter"],
            nburnin=settings["n_burnin"],
            thin=2,
            init=inits,
            modules=[module_name],
            parallel=True,
            maxcores=settings["n_chains"],
        )
    except Exception:
        return None

    est = np.empty(model.n_params)
    ci_lo = np.empty(model.n_params)
    ci_hi = np.empty(model.n_params)
    rhats = np.empty(model.n_params)

    for i, name in enumerate(model.param_names):
        samples = result.get_samples(name)
        est[i] = np.mean(samples)
        ci_lo[i] = np.percentile(samples, 2.5)
        ci_hi[i] = np.percentile(samples, 97.5)
        rhats[i] = result.rhat(name)

    if np.any(rhats > 1.1):
        return None

    return {
        "true_params": true_params,
        "est": est,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "rhats": rhats,
    }


def run_recovery_study_mv(slug: str) -> None:
    """Run a multivariate simulate-and-recover study."""
    model = get_model(slug)
    settings = resolve_recovery_settings()

    if not model.supports_mv_recovery():
        print(f"[recovery_mv] FAIL: Model '{slug}' has no mv recovery hooks.", file=sys.stderr)
        sys.exit(1)

    onnx_path = Path("results") / slug / "model.onnx"
    transform_path = Path("results") / slug / "target_transform.pkl"
    if not onnx_path.exists():
        print(f"[recovery_mv] FAIL: {onnx_path} not found.", file=sys.stderr)
        sys.exit(1)
    if not transform_path.exists():
        print(f"[recovery_mv] FAIL: {transform_path} not found.", file=sys.stderr)
        sys.exit(1)

    load_target_transform(slug)
    print(f"[recovery_mv] Using JNNX synthetic likelihood ({slug}_sl)")

    n_chains = settings["n_chains"]
    max_workers = resolve_recovery_workers(n_chains)

    print(f"[recovery_mv] Model: {slug}")
    print(f"[recovery_mv] Settings: {settings}")
    print(
        f"[recovery_mv] Parallel workers: {max_workers} "
        f"(each uses {n_chains} cores for chains)"
    )

    rng = np.random.default_rng(42)
    work_items = []
    for subj in range(settings["n_subjects"]):
        true_params = np.empty(model.n_params)
        for i, (lo, hi) in enumerate(model.param_bounds):
            true_params[i] = rng.uniform(lo, hi)
        subj_seed = 1000 + subj
        work_items.append((slug, true_params, subj_seed, settings, str(onnx_path)))

    true_params_list = []
    estimated_params_list = []
    ci_lower_list = []
    ci_upper_list = []
    rhat_list = []
    n_failed = 0
    report_interval = recovery_report_interval(settings["n_subjects"])
    t0 = time.monotonic()

    with Pool(processes=max_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_recover_one_subject_mv, work_items, chunksize=1)
        ):
            if result is not None:
                true_params_list.append(result["true_params"])
                estimated_params_list.append(result["est"])
                ci_lower_list.append(result["ci_lo"])
                ci_upper_list.append(result["ci_hi"])
                rhat_list.append(result["rhats"])
            else:
                n_failed += 1

            if (i + 1) % report_interval == 0 or (i + 1) == settings["n_subjects"]:
                line = format_recovery_progress(
                    i + 1,
                    settings["n_subjects"],
                    len(true_params_list),
                    n_failed,
                    t0,
                )
                print(line.replace("[recovery]", "[recovery_mv]", 1))

    print(
        f"[recovery_mv] Finished: {len(true_params_list)} converged, {n_failed} failed"
    )

    if len(true_params_list) < 3:
        print("[recovery_mv] FAIL: Too few converged subjects.", file=sys.stderr)
        sys.exit(1)

    true_params_arr = np.array(true_params_list)
    est_params_arr = np.array(estimated_params_list)
    ci_lower_arr = np.array(ci_lower_list)
    ci_upper_arr = np.array(ci_upper_list)

    figures_dir = Path("figures") / slug
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_recovery_diagnostics(
        model=model,
        true_params=true_params_arr,
        estimated_params=est_params_arr,
        ci_lower=ci_lower_arr,
        ci_upper=ci_upper_arr,
        output_path=figures_dir / "recovery.pdf",
    )
    print(f"[recovery_mv] Recovery plot: {figures_dir / 'recovery.pdf'}")

    correlations = [
        float(np.corrcoef(true_params_arr[:, i], est_params_arr[:, i])[0, 1])
        for i in range(model.n_params)
    ]
    coverages = [
        float(
            np.mean(
                (true_params_arr[:, i] >= ci_lower_arr[:, i])
                & (true_params_arr[:, i] <= ci_upper_arr[:, i])
            )
        )
        for i in range(model.n_params)
    ]

    summary = {
        "n_converged": len(true_params_list),
        "n_attempted": settings["n_subjects"],
        "correlations": dict(zip(model.param_names, correlations)),
        "coverages_95ci": dict(zip(model.param_names, coverages)),
        "mean_rhat": dict(zip(model.param_names, np.mean(rhat_list, axis=0).tolist())),
    }

    results_dir = Path("results") / slug
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[recovery_mv] Summary: {summary}")

    check_coverage_gate(coverages, model.param_names)

    print(
        f"[recovery_mv] PASS: {len(true_params_list)} subjects recovered successfully"
    )

"""Simulate-and-recover study via py2jags."""

import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from asl.config import load_config
from asl.data import load_target_transform
from asl.figures import plot_recovery_diagnostics
from asl.onnxruntime_sdk import ensure_onnxruntime_lib_on_path
from asl.spec import Model
from models.catalog import get_model

N_CHAINS = 4
N_MCMC_ITER = 5000
N_BURNIN = 2000

COVERAGE_TARGET = 0.95
COVERAGE_LO = 0.90
COVERAGE_HI = 0.99


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def format_recovery_progress(
    done: int,
    total: int,
    n_converged: int,
    n_failed: int,
    t0: float,
) -> str:
    elapsed = time.monotonic() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else None
    line = (
        f"[recovery] {done}/{total} done, {n_converged} converged, {n_failed} failed"
        f" | elapsed {_format_duration(elapsed)}"
        f" | {rate:.2f} subj/s"
    )
    if remaining is not None:
        line += f" | ETA ~{_format_duration(remaining)}"
    return line


def recovery_report_interval(n_subjects: int) -> int:
    config = load_config()
    progress_every = int(config.get("recovery", "progress_log_interval", 0))
    if progress_every > 0:
        return max(1, progress_every)
    return max(1, min(10, n_subjects // 50))


def check_coverage_gate(coverages: list[float], param_names: tuple[str, ...]) -> None:
    for i, name in enumerate(param_names):
        cov = coverages[i]
        if cov <= COVERAGE_LO or cov >= COVERAGE_HI:
            print(
                f"[recovery] FAIL: {name} 95% CI coverage {cov:.3f} "
                f"outside ({COVERAGE_LO:.0%}, {COVERAGE_HI:.0%}) "
                f"(target ~{COVERAGE_TARGET:.0%})",
                file=sys.stderr,
            )
            sys.exit(1)


def resolve_recovery_workers(n_chains: int) -> int:
    config = load_config()
    workers = int(config.get("recovery", "parallel_workers", 0))
    if workers > 0:
        return max(1, workers)
    return max(1, int(cpu_count() * 0.9) // n_chains)


def resolve_recovery_settings() -> dict:
    config = load_config()
    return {
        "n_subjects": int(config.get("recovery", "synthetic_subjects", 500)),
        "n_trials": int(config.get("recovery", "trials_per_subject", 500)),
        "n_iter": N_MCMC_ITER,
        "n_burnin": N_BURNIN,
        "n_chains": N_CHAINS,
    }


def iqr_interval(lo: float, hi: float) -> tuple[float, float]:
    """Interquartile range for a uniform distribution on (lo, hi)."""
    span = hi - lo
    return lo + 0.25 * span, lo + 0.75 * span


def compute_chain_initial_values(model: Model, rng_seed: int) -> list[dict]:
    """Draw one uniform start per chain over each parameter's IQR."""
    bounds = model.prior_bounds
    rng = np.random.default_rng(rng_seed)
    inits = []
    for _ in range(N_CHAINS):
        inits.append(
            {
                name: float(rng.uniform(*iqr_interval(lo, hi)))
                for name, (lo, hi) in zip(model.param_names, bounds)
            }
        )
    return inits


def simulate_subject_observations(
    model: Model,
    params: np.ndarray,
    n_trials: int,
    seed: int,
) -> dict:
    summaries = model.simulate_summaries(params, n_trials, seed)
    if not np.all(np.isfinite(summaries)):
        return {"valid": False}
    return {"obs": summaries.reshape(-1), "valid": True}


def build_jags_model_string(model: Model, obs: dict) -> str:
    if not model.supports_recovery():
        raise ValueError(f"Model '{model.slug}' does not define recovery hooks.")

    priors = "\n    ".join(model.recovery_priors.values())
    lines = ["model {", f"    {priors}"]
    lines.extend(f"    {line}" for line in model.build_jags_likelihood(obs))
    lines.append("}")
    return "\n".join(lines)


def _recover_one_subject(args: tuple) -> dict | None:
    slug, true_params, subj_seed, settings = args
    from py2jags import run_jags

    model = get_model(slug)
    load_target_transform(slug)
    module_name = f"{model.slug}_emulator"

    obs = simulate_subject_observations(
        model, true_params, settings["n_trials"], subj_seed
    )
    if not obs["valid"]:
        return None

    model_string = build_jags_model_string(model, obs)
    obs_raw = np.asarray(obs["obs"], dtype=np.float64)
    data = {
        "obs": obs_raw.tolist(),
        "n_trials": settings["n_trials"],
    }
    inits = compute_chain_initial_values(model, subj_seed)

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


def run_recovery_study(model: Model) -> None:
    """Run a simulate-and-recover study."""
    ensure_onnxruntime_lib_on_path()
    slug = model.slug
    settings = resolve_recovery_settings()

    if not model.supports_recovery():
        print(f"[recovery] FAIL: Model '{slug}' has no recovery hooks.", file=sys.stderr)
        sys.exit(1)

    onnx_path = Path("results") / slug / "model.onnx"
    transform_path = Path("results") / slug / "target_transform.pkl"
    if not onnx_path.exists():
        print(f"[recovery] FAIL: {onnx_path} not found.", file=sys.stderr)
        sys.exit(1)
    if not transform_path.exists():
        print(f"[recovery] FAIL: {transform_path} not found.", file=sys.stderr)
        sys.exit(1)

    load_target_transform(slug)
    print(f"[recovery] Model: {slug}")
    print(f"[recovery] Settings: {settings}")
    print("[recovery] Chain inits: uniform on per-parameter IQR of prior bounds")

    n_chains = settings["n_chains"]
    max_workers = resolve_recovery_workers(n_chains)
    print(
        f"[recovery] Parallel workers: {max_workers} "
        f"(each uses {n_chains} cores for chains)"
    )

    rng = np.random.default_rng(42)
    work_items = []
    recovery_bounds = model.prior_bounds
    for subj in range(settings["n_subjects"]):
        true_params = np.empty(model.n_params)
        for i, (lo, hi) in enumerate(recovery_bounds):
            true_params[i] = rng.uniform(lo, hi)
        subj_seed = 1000 + subj
        work_items.append((slug, true_params, subj_seed, settings))

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
            pool.imap_unordered(_recover_one_subject, work_items, chunksize=1)
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
                print(
                    format_recovery_progress(
                        i + 1,
                        settings["n_subjects"],
                        len(true_params_list),
                        n_failed,
                        t0,
                    )
                )

    print(
        f"[recovery] Finished: {len(true_params_list)} converged, {n_failed} failed"
    )

    if len(true_params_list) < 3:
        print("[recovery] FAIL: Too few converged subjects.", file=sys.stderr)
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
    print(f"[recovery] Recovery plot: {figures_dir / 'recovery.pdf'}")

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

    print(f"[recovery] Summary: {summary}")

    check_coverage_gate(coverages, model.param_names)

    print(
        f"[recovery] PASS: {len(true_params_list)} subjects recovered successfully"
    )

"""Simulate-and-recover study via py2jags."""

import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy import stats

from asl.config import load_config
from asl.cov_data import SEED_DEFAULT
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
        "min_success_rate": float(config.get("recovery", "min_success_rate", 0.98)),
        "max_retries_per_subject": int(
            config.get("recovery", "max_retries_per_subject", 5)
        ),
    }


def resolve_true_param_bounds(model: Model) -> tuple[tuple[float, float], ...]:
    """Bounds for drawing synthetic true parameters (defaults to model.prior_bounds)."""
    config = load_config()
    custom = config.get("recovery", "true_param_bounds", None)
    if custom is None:
        return model.prior_bounds
    if len(custom) != model.n_params:
        raise ValueError(
            f"recovery.true_param_bounds length {len(custom)} "
            f"!= n_params {model.n_params}"
        )
    bounds: list[tuple[float, float]] = []
    for item in custom:
        lo, hi = float(item[0]), float(item[1])
        bounds.append((lo, hi))
    return tuple(bounds)


def draw_from_prior(
    model: Model,
    rng: np.random.Generator,
    bounds: tuple[tuple[float, float], ...] | None = None,
) -> np.ndarray:
    """Draw one parameter vector within bounds using the model's prior family."""
    draw_bounds = bounds or model.prior_bounds
    params = np.empty(model.n_params)
    for i, (name, (lo, hi)) in enumerate(zip(model.param_names, draw_bounds)):
        prior_line = model.recovery_priors.get(name, "")
        if "dnorm" in prior_line:
            sigma = (hi - lo) / 4.0
            mu = (lo + hi) / 2.0
            a, b = (lo - mu) / sigma, (hi - mu) / sigma
            params[i] = float(stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=rng))
        else:
            params[i] = float(rng.uniform(lo, hi))
    return params


def check_success_rate_gate(n_converged: int, n_attempted: int, min_rate: float) -> None:
    rate = n_converged / n_attempted if n_attempted else 0.0
    if rate < min_rate:
        print(
            f"[recovery] FAIL: success rate {rate:.3f} < {min_rate:.3f} "
            f"({n_converged}/{n_attempted} converged)",
            file=sys.stderr,
        )
        sys.exit(1)


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


def report_param_bounds(
    model: Model, bounds: tuple[tuple[float, float], ...]
) -> tuple[tuple[float, float], ...]:
    """Map parameter bounds to the scale used in recovery reports."""
    if model.report_params_fn is None:
        return bounds

    reported = model.report_params_fn(np.asarray(bounds, dtype=np.float64))
    return tuple((float(lo), float(hi)) for lo, hi in reported)


def recovery_reporting_arrays(
    model: Model,
    true_params: np.ndarray,
    estimated_params: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    true_draw_bounds: tuple[tuple[float, float], ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[tuple[float, float], ...]]:
    """Optionally map latent inference parameters to reported canonical scale."""
    if model.report_params_fn is None:
        bounds = true_draw_bounds or model.prior_bounds
        return true_params, estimated_params, ci_lower, ci_upper, model.param_names, bounds

    true_report = model.report_params_fn(true_params)
    est_report = model.report_params_fn(estimated_params)
    lo_report = model.report_params_fn(ci_lower)
    hi_report = model.report_params_fn(ci_upper)
    names = model.report_param_names or model.param_names
    bounds = report_param_bounds(model, true_draw_bounds or model.prior_bounds)
    return true_report, est_report, lo_report, hi_report, names, bounds


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


def _recover_one_subject_attempt(
    slug: str,
    true_params: np.ndarray,
    subj_seed: int,
    settings: dict,
) -> dict:
    from py2jags import run_jags

    model = get_model(slug)
    load_target_transform(slug)
    module_name = f"{model.slug}_emulator"

    obs = simulate_subject_observations(
        model, true_params, settings["n_trials"], subj_seed
    )
    if not obs["valid"]:
        return {"status": "failed", "reason": "invalid_simulation"}

    model_string = build_jags_model_string(model, obs)
    obs_raw = np.asarray(obs["obs"], dtype=np.float64)
    data = {
        "obs": obs_raw.tolist(),
        model.sample_size_arg: settings["n_trials"],
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
    except Exception as exc:
        return {"status": "failed", "reason": "jags_exception", "exception": str(exc)}

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

    rhat_max = float(np.max(rhats))
    if rhat_max > 1.1:
        return {"status": "failed", "reason": "rhat", "rhat_max": rhat_max}

    return {
        "status": "converged",
        "true_params": true_params,
        "est": est,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "rhats": rhats,
    }


def _recover_one_subject(args: tuple) -> dict:
    slug, subj_idx, rng_seed, settings = args
    model = get_model(slug)
    rng = np.random.default_rng(rng_seed)
    max_retries = int(settings["max_retries_per_subject"])
    last_result: dict = {"status": "failed", "reason": "exhausted_retries"}
    true_draw_bounds = settings.get("true_draw_bounds", model.prior_bounds)

    for retry in range(max_retries):
        true_params = draw_from_prior(model, rng, true_draw_bounds)
        subj_seed = 1000 + subj_idx + retry * 100_000
        result = _recover_one_subject_attempt(slug, true_params, subj_seed, settings)
        if result["status"] == "converged":
            return result
        last_result = result

    return last_result


def write_recovery_subjects(
    model: Model,
    true_params: np.ndarray,
    estimated_params: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    rhats: np.ndarray,
    results_dir: Path,
    true_draw_bounds: tuple[tuple[float, float], ...] | None = None,
    *,
    param_names: tuple[str, ...] | None = None,
    inference_bounds: tuple[tuple[float, float], ...] | None = None,
) -> Path:
    """Write legacy per-subject recovery arrays for paper figure scripts."""
    names = param_names or model.param_names
    inference_bounds = inference_bounds or model.prior_bounds
    draw_bounds = true_draw_bounds or inference_bounds
    payload = {
        "param_names": list(names),
        "param_bounds": [list(bounds) for bounds in inference_bounds],
        "true_draw_bounds": [list(bounds) for bounds in draw_bounds],
        "true": true_params.tolist(),
        "est": estimated_params.tolist(),
        "ci_lo": ci_lower.tolist(),
        "ci_hi": ci_upper.tolist(),
        "rhat": rhats.tolist(),
    }
    path = results_dir / "recovery_subjects.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


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
    true_draw_bounds = resolve_true_param_bounds(model)
    settings["true_draw_bounds"] = true_draw_bounds
    print(f"[recovery] True-parameter draw bounds: {true_draw_bounds}")
    print("[recovery] True-parameter distribution: model prior family")
    print("[recovery] Chain inits: uniform on per-parameter IQR of prior bounds")

    n_chains = settings["n_chains"]
    max_workers = resolve_recovery_workers(n_chains)
    print(
        f"[recovery] Parallel workers: {max_workers} "
        f"(each uses {n_chains} cores for chains)"
    )

    work_items = [
        (slug, subj, SEED_DEFAULT + subj, settings)
        for subj in range(settings["n_subjects"])
    ]

    true_params_list = []
    estimated_params_list = []
    ci_lower_list = []
    ci_upper_list = []
    rhat_list = []
    failure_counts: dict[str, int] = {}
    n_failed = 0
    report_interval = recovery_report_interval(settings["n_subjects"])
    t0 = time.monotonic()

    with Pool(processes=max_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_recover_one_subject, work_items, chunksize=1)
        ):
            if result.get("status") == "converged":
                true_params_list.append(result["true_params"])
                estimated_params_list.append(result["est"])
                ci_lower_list.append(result["ci_lo"])
                ci_upper_list.append(result["ci_hi"])
                rhat_list.append(result["rhats"])
            else:
                n_failed += 1
                reason = result.get("reason", "unknown")
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

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

    recovery_time_seconds = time.monotonic() - t0
    print(
        f"[recovery] Finished: {len(true_params_list)} converged, {n_failed} failed"
    )
    if failure_counts:
        print(f"[recovery] Failure counts: {failure_counts}")

    check_success_rate_gate(
        len(true_params_list),
        settings["n_subjects"],
        settings["min_success_rate"],
    )

    true_params_arr = np.array(true_params_list)
    est_params_arr = np.array(estimated_params_list)
    ci_lower_arr = np.array(ci_lower_list)
    ci_upper_arr = np.array(ci_upper_list)
    rhat_arr = np.array(rhat_list)

    (
        true_report,
        est_report,
        ci_lo_report,
        ci_hi_report,
        report_names,
        report_draw_bounds,
    ) = recovery_reporting_arrays(
        model, true_params_arr, est_params_arr, ci_lower_arr, ci_upper_arr, true_draw_bounds
    )

    results_dir = Path("results") / slug
    results_dir.mkdir(parents=True, exist_ok=True)
    subjects_path = write_recovery_subjects(
        model,
        true_report,
        est_report,
        ci_lo_report,
        ci_hi_report,
        rhat_arr,
        results_dir,
        true_draw_bounds=report_draw_bounds,
        param_names=report_names,
        inference_bounds=model.report_prior_bounds
        or report_param_bounds(model, model.prior_bounds),
    )
    print(f"[recovery] Per-subject arrays: {subjects_path}")

    figures_dir = Path("figures") / slug
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_recovery_diagnostics(
        model=model,
        true_params=true_report,
        estimated_params=est_report,
        ci_lower=ci_lo_report,
        ci_upper=ci_hi_report,
        output_path=figures_dir / "recovery.pdf",
        param_names=report_names,
    )
    print(f"[recovery] Recovery plot: {figures_dir / 'recovery.pdf'}")

    n_report = len(report_names)
    correlations = [
        float(np.corrcoef(true_report[:, i], est_report[:, i])[0, 1])
        for i in range(n_report)
    ]
    coverages = [
        float(
            np.mean(
                (true_report[:, i] >= ci_lo_report[:, i])
                & (true_report[:, i] <= ci_hi_report[:, i])
            )
        )
        for i in range(n_report)
    ]

    summary = {
        "n_converged": len(true_params_list),
        "n_attempted": settings["n_subjects"],
        "n_failed": n_failed,
        "failure_counts": failure_counts,
        "recovery_time_seconds": recovery_time_seconds,
        "correlations": dict(zip(report_names, correlations)),
        "coverages_95ci": dict(zip(report_names, coverages)),
        "mean_rhat": dict(zip(model.param_names, np.mean(rhat_list, axis=0).tolist())),
    }

    results_dir = Path("results") / slug
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[recovery] Summary: {summary}")

    check_coverage_gate(coverages, report_names)

    print(
        f"[recovery] PASS: {len(true_params_list)} subjects recovered successfully"
    )

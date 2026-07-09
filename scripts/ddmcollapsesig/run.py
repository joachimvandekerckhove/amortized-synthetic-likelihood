#!/usr/bin/env python3
"""Application entry point for the ddmcollapsesig ASL pipeline."""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from asl.cov_data import generate_cov_dataset
from asl.data import load_target_transform
from asl.mv import build_sl_likelihood_line
from asl.recovery import (
    check_coverage_gate,
    format_recovery_progress,
    recovery_report_interval,
    resolve_recovery_settings,
    resolve_recovery_workers,
)
from asl.registry import register
from asl.train_mv import train_emulator_mv
from asl.wire import wire_to_jags
from models.ddm.ddmcollapsesig import PARAM_BOUNDS, PARAM_NAMES
from models.ddm.ddmcollapsesig_collapsemv import DDMCOLLAPSESIG_COLLAPSEMV
from models.ddm.ddmcollapsesig_fixedmv import DDMCOLLAPSESIG_FIXEDMV

FIXED_SLUG = "ddmcollapsesig_fixed"
COLLAPSE_SLUG = "ddmcollapsesig_collapse"
N_SUMMARIES = DDMCOLLAPSESIG_FIXEDMV.n_summaries


def register_models() -> None:
    """Register both condition-specific MV models."""
    register(DDMCOLLAPSESIG_FIXEDMV)
    register(DDMCOLLAPSESIG_COLLAPSEMV)


def build_joint_jags_model() -> str:
    """Build the JAGS model that combines both condition emulators."""
    lines = [
        "model {",
        "    a0 ~ dunif(0.5, 2.0)",
        "    v ~ dunif(-1.5, 1.5)",
        "    k ~ dunif(0, 8.0)",
        "    t0 ~ dunif(0.05, 0.45)",
    ]
    lines.extend(
        f"    {line}"
        for line in build_sl_likelihood_line(
            FIXED_SLUG,
            PARAM_NAMES,
            N_SUMMARIES,
            obs_name="obs_fixed_std",
            n_trials_name="n_trials_fixed",
        )
    )
    lines.extend(
        f"    {line}"
        for line in build_sl_likelihood_line(
            COLLAPSE_SLUG,
            PARAM_NAMES,
            N_SUMMARIES,
            obs_name="obs_collapse_std",
            n_trials_name="n_trials_collapse",
        )
    )
    lines.append("}")
    return "\n".join(lines)


def _draw_joint_params(rng: np.random.Generator) -> np.ndarray:
    """Draw one parameter vector uniformly within model bounds."""
    return np.array(
        [rng.uniform(lo, hi) for lo, hi in PARAM_BOUNDS], dtype=np.float64
    )


def _standardized_obs(slug: str, summaries: np.ndarray) -> np.ndarray:
    """Transform raw summaries to standardized emulator target space."""
    transform = load_target_transform(slug)
    return transform.transform(summaries.reshape(1, -1))[0]


def _joint_inits(true_params: np.ndarray) -> list[dict[str, float]]:
    """Create dispersed initial values near the truth for MCMC."""
    rng = np.random.default_rng(123)
    inits = []
    bounds = dict(zip(PARAM_NAMES, PARAM_BOUNDS))
    for _ in range(4):
        init = {}
        for name, value in zip(PARAM_NAMES, true_params):
            lo, hi = bounds[name]
            jitter = rng.normal(0.0, 0.02 * (hi - lo))
            init[name] = float(np.clip(value + jitter, lo + 1e-4, hi - 1e-4))
        inits.append(init)
    return inits


def _recover_one_subject(args: tuple) -> dict | None:
    """Run joint JAGS recovery for one simulated subject."""
    index, true_params, settings = args
    from py2jags import run_jags

    fixed_obs = DDMCOLLAPSESIG_FIXEDMV.simulate_summaries(
        true_params, settings["n_trials"], 1000 + index * 10
    )
    collapse_obs = DDMCOLLAPSESIG_COLLAPSEMV.simulate_summaries(
        true_params, settings["n_trials"], 1001 + index * 10
    )
    if not np.all(np.isfinite(fixed_obs)) or not np.all(np.isfinite(collapse_obs)):
        return None

    data = {
        "obs_fixed_std": _standardized_obs(FIXED_SLUG, fixed_obs).tolist(),
        "obs_collapse_std": _standardized_obs(COLLAPSE_SLUG, collapse_obs).tolist(),
        "n_trials_fixed": settings["n_trials"],
        "n_trials_collapse": settings["n_trials"],
    }
    try:
        result = run_jags(
            model_string=build_joint_jags_model(),
            data_dict=data,
            monitorparams=list(PARAM_NAMES),
            nchains=settings["n_chains"],
            nsamples=settings["n_iter"],
            nburnin=settings["n_burnin"],
            thin=2,
            init=_joint_inits(true_params),
            modules=[
                "ddmcollapsesig_fixed_emulator",
                "ddmcollapsesig_collapse_emulator",
            ],
            parallel=True,
            maxcores=settings["n_chains"],
        )
    except Exception:
        return None

    est = np.empty(len(PARAM_NAMES))
    ci_lo = np.empty(len(PARAM_NAMES))
    ci_hi = np.empty(len(PARAM_NAMES))
    rhats = np.empty(len(PARAM_NAMES))
    for i, name in enumerate(PARAM_NAMES):
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


def run_joint_recovery() -> None:
    """Run a parallel joint recovery study using both condition emulators."""
    settings = resolve_recovery_settings()
    print("[joint-recovery] Using JNNX synthetic likelihood (_sl nodes)")

    n_chains = settings["n_chains"]
    max_workers = resolve_recovery_workers(n_chains)

    print(f"[joint-recovery] Settings: {settings}")
    print(
        f"[joint-recovery] Parallel workers: {max_workers} "
        f"(each uses {n_chains} cores for chains)"
    )

    rng = np.random.default_rng(42)
    work_items = []
    for subj in range(settings["n_subjects"]):
        true_params = _draw_joint_params(rng)
        work_items.append((subj + 1, true_params, settings))

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
                line = format_recovery_progress(
                    i + 1,
                    settings["n_subjects"],
                    len(true_params_list),
                    n_failed,
                    t0,
                )
                print(line.replace("[recovery]", "[joint-recovery]", 1))

    if len(true_params_list) < 3:
        print("[joint-recovery] FAIL: Too few converged subjects.", file=sys.stderr)
        sys.exit(1)

    true_params_arr = np.array(true_params_list)
    est_params_arr = np.array(estimated_params_list)
    ci_lower_arr = np.array(ci_lower_list)
    ci_upper_arr = np.array(ci_upper_list)

    coverages = [
        float(
            np.mean(
                (true_params_arr[:, j] >= ci_lower_arr[:, j])
                & (true_params_arr[:, j] <= ci_upper_arr[:, j])
            )
        )
        for j in range(len(PARAM_NAMES))
    ]
    correlations = [
        float(np.corrcoef(true_params_arr[:, j], est_params_arr[:, j])[0, 1])
        for j in range(len(PARAM_NAMES))
    ]

    summary = {
        "n_converged": len(true_params_list),
        "n_attempted": settings["n_subjects"],
        "correlations": dict(zip(PARAM_NAMES, correlations)),
        "coverages_95ci": dict(zip(PARAM_NAMES, coverages)),
        "mean_rhat": dict(zip(PARAM_NAMES, np.mean(rhat_list, axis=0).tolist())),
    }

    results_dir = Path("results") / "ddmcollapsesig_joint"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[joint-recovery] Summary: {summary}")
    check_coverage_gate(coverages, PARAM_NAMES)
    print(
        f"[joint-recovery] PASS: {len(true_params_list)} subjects recovered successfully"
    )


def main() -> None:
    steps = (
        "generate-fixed",
        "generate-collapse",
        "train-fixed",
        "train-collapse",
        "wire-fixed",
        "wire-collapse",
        "joint-recovery",
    )
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        print(f"Usage: {sys.argv[0]} <{'|'.join(steps)}>", file=sys.stderr)
        sys.exit(1)

    register_models()
    step = sys.argv[1]

    if step == "generate-fixed":
        generate_cov_dataset(FIXED_SLUG)
    elif step == "generate-collapse":
        generate_cov_dataset(COLLAPSE_SLUG)
    elif step == "train-fixed":
        train_emulator_mv(FIXED_SLUG)
    elif step == "train-collapse":
        train_emulator_mv(COLLAPSE_SLUG)
    elif step == "wire-fixed":
        wire_to_jags(FIXED_SLUG)
    elif step == "wire-collapse":
        wire_to_jags(COLLAPSE_SLUG)
    elif step == "joint-recovery":
        run_joint_recovery()


if __name__ == "__main__":
    main()

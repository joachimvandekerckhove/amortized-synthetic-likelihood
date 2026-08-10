"""
Multi-N C1 stability evaluation for DDM and DW emulators.

Draws fixed parameter vectors, simulates replicate datasets at several N,
estimates C1(theta, N) = N * Cov(S_N | theta) in frozen target_transform
space, and compares each N against a reference batch plus an independent
null batch at the reference N.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from asl.cov_data import SEED_DEFAULT
from asl.data import load_target_transform
from asl.n_stability import (
    correlation_differences,
    diagonal_ratios,
    estimate_c1,
    generalized_eigenvalues,
    is_positive_definite,
    percentile_summary,
    relative_frobenius_error,
    stein_discrepancy,
)
from asl.spec import Model
from models.catalog import get_model

SUPPORTED_SLUGS = ("ddm3", "ddm4", "ddmcollapsesig", "dw")
MAX_PROFILE_REPLACEMENT_ATTEMPTS = 20


@dataclass(frozen=True)
class SlugProfile:
    """Per-model N-stability settings."""

    n_values: tuple[int, ...]
    ref_n: int
    null_key: str
    table_order: tuple[str, ...]
    n_size_label: str
    default_n_theta: int
    theta_mode: str  # "random" or "fixed_dw"


DDM_PROFILE = SlugProfile(
    n_values=(50, 100, 300, 600, 1000),
    ref_n=600,
    null_key="600_null",
    table_order=("50", "100", "300", "600", "600_null", "1000"),
    n_size_label="N",
    default_n_theta=200,
    theta_mode="random",
)

DW_PROFILE = SlugProfile(
    n_values=(50, 100, 150, 300, 600),
    ref_n=150,
    null_key="150_null",
    table_order=("50", "100", "150", "300", "600", "150_null"),
    n_size_label=r"n_{\mathrm{agents}}",
    default_n_theta=5,
    theta_mode="fixed_dw",
)

SLUG_PROFILES: dict[str, SlugProfile] = {
    "ddm3": DDM_PROFILE,
    "ddm4": DDM_PROFILE,
    "ddmcollapsesig": DDM_PROFILE,
    "dw": DW_PROFILE,
}


def get_slug_profile(slug: str) -> SlugProfile:
    return SLUG_PROFILES[slug]


@dataclass(frozen=True)
class StudyConfig:
    """Fixed settings for one stability evaluation run."""

    slug: str
    n_theta: int
    n_replicates: int
    profile: SlugProfile
    seed: int
    workers: int
    results_dir: Path

    @property
    def model(self) -> Model:
        return get_model(self.slug)

    @property
    def n_summaries(self) -> int:
        return self.model.n_summaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-unit covariance stability across N."
    )
    parser.add_argument(
        "--slug",
        required=True,
        choices=SUPPORTED_SLUGS,
        help="Model slug (DDM or dw).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke run (8 thetas, 40 replicates).",
    )
    parser.add_argument("--n-theta", type=int, default=None)
    parser.add_argument("--n-replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> StudyConfig:
    profile = get_slug_profile(args.slug)
    if args.quick:
        n_theta, n_replicates = 8, 40
    else:
        n_theta, n_replicates = profile.default_n_theta, 500
    if args.n_theta is not None:
        n_theta = args.n_theta
    if args.n_replicates is not None:
        n_replicates = args.n_replicates
    workers = args.workers if args.workers > 0 else max(1, int(cpu_count() * 0.9))
    results_dir = args.results_dir or (Path("results") / args.slug)
    return StudyConfig(
        slug=args.slug,
        n_theta=n_theta,
        n_replicates=n_replicates,
        profile=profile,
        seed=args.seed,
        workers=workers,
        results_dir=results_dir,
    )


def draw_random_thetas(model: Model, n_theta: int, seed: int) -> np.ndarray:
    """Draw deterministic parameter vectors from the training support."""
    rng = np.random.default_rng(seed)
    params = np.empty((n_theta, model.n_params), dtype=np.float64)
    for i, (lo, hi) in enumerate(model.param_bounds):
        params[:, i] = rng.uniform(lo, hi, size=n_theta)
    return params


def draw_fixed_thetas(
    model: Model, n_theta: int, seed: int, profile: SlugProfile
) -> np.ndarray:
    if profile.theta_mode == "fixed_dw":
        from models.social.dw import study_logit_thetas

        return study_logit_thetas(n_theta)

    return draw_random_thetas(model, n_theta, seed)


def stable_tag_code(batch_tag: str) -> int:
    code = 0
    for character in batch_tag:
        code = (code * 131 + ord(character)) % 1_000_003
    return code


def batch_seed(master_seed: int, batch_tag: str, theta_index: int) -> int:
    return int(master_seed + 1_000_003 * stable_tag_code(batch_tag) + 10_007 * theta_index)


def raw_checkpoint_path(results_dir: Path, batch_key: str) -> Path:
    return results_dir / f"n_stability_raw_N{batch_key}.npz"


def file_sha256(path: Path) -> str:
    """Return a stable content fingerprint for one input artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulation_context_sha256() -> str:
    """Fingerprint simulator source and configuration inputs for raw batches."""
    repo_root = Path(__file__).resolve().parents[2]
    paths = sorted((repo_root / "models").rglob("*.py"))
    paths.extend(
        [
            repo_root / "src" / "asl" / "config.py",
            repo_root / "src" / "asl" / "n_stability_study.py",
            repo_root / "src" / "asl" / "presets" / "full.toml",
            repo_root / "asl.toml",
        ]
    )
    config_override = os.environ.get("ASL_CONFIG")
    if config_override:
        paths.append(Path(config_override))

    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths if path.is_file()}):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_matches_study(
    payload: dict,
    config: StudyConfig,
    params: np.ndarray,
    batch_key: str,
    n_size: int,
    target_transform_sha256: str,
    simulation_context_sha256: str,
) -> bool:
    """Return whether a cached batch belongs to this exact study request."""
    required = {
        "slug",
        "seed",
        "n_replicates",
        "n_size",
        "batch_key",
        "target_transform_sha256",
        "simulation_context_sha256",
        "params",
    }
    if not required.issubset(payload):
        return False
    return (
        str(payload["slug"].item()) == config.slug
        and int(payload["seed"].item()) == config.seed
        and int(payload["n_replicates"].item()) == config.n_replicates
        and int(payload["n_size"].item()) == n_size
        and str(payload["batch_key"].item()) == batch_key
        and str(payload["target_transform_sha256"].item()) == target_transform_sha256
        and str(payload["simulation_context_sha256"].item())
        == simulation_context_sha256
        and payload["params"].shape == params.shape
        and np.array_equal(payload["params"], params)
    )


def _simulate_theta_batch(args: tuple) -> dict:
    slug, theta_index, params, n_size, n_replicates, base_seed, transform_path = args
    model = get_model(slug)
    with open(transform_path, "rb") as handle:
        target_transform = pickle.load(handle)

    summaries_std = np.full((n_replicates, model.n_summaries), np.nan, dtype=np.float64)
    n_failed = 0
    for replicate_index in range(n_replicates):
        summaries = model.simulate_summaries(
            params, n_size, base_seed + replicate_index
        )
        if not np.all(np.isfinite(summaries)):
            n_failed += 1
            continue
        summaries_std[replicate_index] = target_transform.transform(
            summaries.reshape(1, -1)
        )[0]

    valid = np.all(np.isfinite(summaries_std), axis=1)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return {
            "theta_index": theta_index,
            "params": params.astype(np.float64),
            "n_valid": n_valid,
            "n_failed": n_failed,
            "c1": np.full((model.n_summaries, model.n_summaries), np.nan),
            "mean_std": np.full(model.n_summaries, np.nan),
            "summaries_std": summaries_std,
            "ok": False,
        }

    valid_rows = summaries_std[valid]
    c1 = estimate_c1(valid_rows, n_size)
    return {
        "theta_index": theta_index,
        "params": params.astype(np.float64),
        "n_valid": n_valid,
        "n_failed": n_failed,
        "c1": c1,
        "mean_std": valid_rows.mean(axis=0),
        "summaries_std": summaries_std,
        "ok": bool(is_positive_definite(c1)),
    }


def run_batch(
    config: StudyConfig,
    params: np.ndarray,
    batch_key: str,
    n_size: int,
) -> dict:
    path = raw_checkpoint_path(config.results_dir, batch_key)
    transform_path = config.results_dir / "target_transform.pkl"
    transform_sha256 = file_sha256(transform_path)
    simulation_sha256 = simulation_context_sha256()
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        payload = {key: loaded[key] for key in loaded.files}
        if checkpoint_matches_study(
            payload,
            config,
            params,
            batch_key,
            n_size,
            transform_sha256,
            simulation_sha256,
        ):
            print(f"[n_stability] Loading checkpoint {path.name}")
            payload["batch_key"] = batch_key
            payload["n_size"] = n_size
            return payload
        print(f"[n_stability] Discarding stale checkpoint {path.name}")

    n_summaries = config.n_summaries
    work_items = [
        (
            config.slug,
            theta_index,
            params[theta_index],
            n_size,
            config.n_replicates,
            batch_seed(config.seed, batch_key, theta_index),
            str(transform_path),
        )
        for theta_index in range(config.n_theta)
    ]

    print(
        f"[n_stability] {config.slug} batch {batch_key} "
        f"(N={n_size}, thetas={config.n_theta}, R={config.n_replicates}, "
        f"workers={config.workers})"
    )

    c1 = np.full((config.n_theta, n_summaries, n_summaries), np.nan)
    mean_std = np.full((config.n_theta, n_summaries), np.nan)
    summaries_std = np.full(
        (config.n_theta, config.n_replicates, n_summaries), np.nan
    )
    n_valid = np.zeros(config.n_theta, dtype=np.int32)
    n_failed = np.zeros(config.n_theta, dtype=np.int32)
    ok = np.zeros(config.n_theta, dtype=bool)

    done = 0
    report_every = max(1, config.n_theta // 10)
    with Pool(processes=config.workers) as pool:
        for result in pool.imap_unordered(_simulate_theta_batch, work_items, chunksize=1):
            idx = int(result["theta_index"])
            c1[idx] = result["c1"]
            mean_std[idx] = result["mean_std"]
            summaries_std[idx] = result["summaries_std"]
            n_valid[idx] = result["n_valid"]
            n_failed[idx] = result["n_failed"]
            ok[idx] = result["ok"]
            done += 1
            if done % report_every == 0:
                print(
                    f"[n_stability]   {batch_key}: {done}/{config.n_theta} "
                    f"({100 * done / config.n_theta:.0f}%), "
                    f"pd={int(ok.sum())}",
                    flush=True,
                )

    payload = {
        "slug": np.array(config.slug),
        "params": params,
        "c1": c1,
        "mean_std": mean_std,
        "summaries_std": summaries_std.astype(np.float32),
        "n_valid": n_valid,
        "n_failed": n_failed,
        "ok": ok,
        "n_size": np.array(n_size),
        "batch_key": np.array(batch_key),
        "target_transform_sha256": np.array(transform_sha256),
        "simulation_context_sha256": np.array(simulation_sha256),
        "seed": np.array(config.seed),
        "n_replicates": np.array(config.n_replicates),
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[n_stability] Wrote {path}")
    payload["batch_key"] = batch_key
    payload["n_size"] = n_size
    return payload


def rerun_batch_points(
    config: StudyConfig,
    batch: dict,
    params: np.ndarray,
    batch_key: str,
    n_size: int,
    theta_indices: np.ndarray,
    replacement_attempt: int,
) -> None:
    """Replace selected profile points in one cached batch."""
    transform_path = config.results_dir / "target_transform.pkl"
    work_items = [
        (
            config.slug,
            int(theta_index),
            params[theta_index],
            n_size,
            config.n_replicates,
            batch_seed(
                config.seed + replacement_attempt * 1_000_000_000,
                batch_key,
                int(theta_index),
            ),
            str(transform_path),
        )
        for theta_index in theta_indices
    ]

    with Pool(processes=config.workers) as pool:
        for result in pool.imap_unordered(_simulate_theta_batch, work_items, chunksize=1):
            idx = int(result["theta_index"])
            batch["params"][idx] = params[idx]
            batch["c1"][idx] = result["c1"]
            batch["mean_std"][idx] = result["mean_std"]
            batch["summaries_std"][idx] = result["summaries_std"]
            batch["n_valid"][idx] = result["n_valid"]
            batch["n_failed"][idx] = result["n_failed"]
            batch["ok"][idx] = result["ok"]

    batch["slug"] = np.array(config.slug)
    batch["target_transform_sha256"] = np.array(file_sha256(transform_path))
    batch["simulation_context_sha256"] = np.array(simulation_context_sha256())
    np.savez_compressed(raw_checkpoint_path(config.results_dir, batch_key), **batch)


def invalid_profile_indices(batches: dict[str, dict]) -> np.ndarray:
    """Return profile indices invalid in at least one sample-size batch."""
    valid = np.ones(len(next(iter(batches.values()))["ok"]), dtype=bool)
    for batch in batches.values():
        valid &= batch["ok"]
    return np.flatnonzero(~valid)


def replace_invalid_random_profile_points(
    config: StudyConfig,
    params: np.ndarray,
    batches: dict[str, dict],
) -> None:
    """Resample random profile points until every batch has positive-definite C1."""
    if config.profile.theta_mode != "random":
        return

    for attempt in range(1, MAX_PROFILE_REPLACEMENT_ATTEMPTS + 1):
        invalid = invalid_profile_indices(batches)
        if not len(invalid):
            return

        print(
            f"[n_stability] Replacing {len(invalid)} invalid profile points "
            f"(attempt {attempt}/{MAX_PROFILE_REPLACEMENT_ATTEMPTS})"
        )
        params[invalid] = draw_random_thetas(
            config.model,
            len(invalid),
            config.seed + attempt * 10_007,
        )
        for n_size in config.profile.n_values:
            rerun_batch_points(
                config,
                batches[str(n_size)],
                params,
                str(n_size),
                n_size,
                invalid,
                attempt,
            )
        rerun_batch_points(
            config,
            batches[config.profile.null_key],
            params,
            config.profile.null_key,
            config.profile.ref_n,
            invalid,
            attempt,
        )

    invalid = invalid_profile_indices(batches)
    raise RuntimeError(
        "Unable to replace all invalid N-stability profile points after "
        f"{MAX_PROFILE_REPLACEMENT_ATTEMPTS} attempts: {invalid.tolist()}"
    )


def compare_to_reference(
    c_n: np.ndarray,
    c_ref: np.ndarray,
    ok_n: bool,
    ok_ref: bool,
    n_summaries: int,
) -> dict:
    if not (ok_n and ok_ref):
        return {
            "valid": False,
            "diag_ratios": [float("nan")] * n_summaries,
            "max_abs_corr_diff": float("nan"),
            "rel_frobenius": float("nan"),
            "gen_eigs": [float("nan")] * n_summaries,
            "stein": float("nan"),
        }
    ratios = diagonal_ratios(c_n, c_ref)
    corr_diff = correlation_differences(c_n, c_ref)
    eigs = generalized_eigenvalues(c_n, c_ref)
    return {
        "valid": True,
        "diag_ratios": [float(x) for x in ratios],
        "max_abs_corr_diff": float(np.max(np.abs(corr_diff))),
        "rel_frobenius": relative_frobenius_error(c_n, c_ref),
        "gen_eigs": [float(x) for x in eigs],
        "stein": stein_discrepancy(c_n, c_ref),
    }


def summarize_comparisons(rows: list[dict], n_summaries: int) -> dict:
    valid_rows = [row for row in rows if row["valid"]]
    if not valid_rows:
        empty = percentile_summary(np.array([]))
        return {
            "n_valid": 0,
            "diag_ratios": [empty] * n_summaries,
            "max_abs_corr_diff": empty,
            "rel_frobenius": empty,
            "gen_eig_min": empty,
            "gen_eig_max": empty,
            "stein": empty,
        }

    diag = np.array([row["diag_ratios"] for row in valid_rows], dtype=np.float64)
    return {
        "n_valid": len(valid_rows),
        "diag_ratios": [
            percentile_summary(diag[:, summary_index])
            for summary_index in range(n_summaries)
        ],
        "max_abs_corr_diff": percentile_summary(
            np.array([row["max_abs_corr_diff"] for row in valid_rows])
        ),
        "rel_frobenius": percentile_summary(
            np.array([row["rel_frobenius"] for row in valid_rows])
        ),
        "gen_eig_min": percentile_summary(
            np.array([min(row["gen_eigs"]) for row in valid_rows])
        ),
        "gen_eig_max": percentile_summary(
            np.array([max(row["gen_eigs"]) for row in valid_rows])
        ),
        "stein": percentile_summary(np.array([row["stein"] for row in valid_rows])),
    }


def mean_shift_summary(
    mean_n: np.ndarray,
    mean_ref: np.ndarray,
    ok: np.ndarray,
    n_summaries: int,
) -> dict:
    if not np.any(ok):
        empty = percentile_summary(np.array([]))
        return {"n_valid": 0, "l2": empty, "per_summary": [empty] * n_summaries}
    diffs = mean_n[ok] - mean_ref[ok]
    return {
        "n_valid": int(ok.sum()),
        "l2": percentile_summary(np.linalg.norm(diffs, axis=1)),
        "per_summary": [
            percentile_summary(diffs[:, summary_index])
            for summary_index in range(n_summaries)
        ],
    }


def format_interval(summary: dict) -> str:
    rounded = []
    for value in (summary["median"], summary["p05"], summary["p95"]):
        text = f"{value:.3f}"
        if text in {"-0.000", "0.000"}:
            text = "0.000"
        rounded.append(text)
    return f"{rounded[0]} [{rounded[1]}, {rounded[2]}]"


def write_table_tex(summary: dict, path: Path, profile: SlugProfile) -> None:
    n_summaries = len(summary["summary_names"])
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        (
            rf"${profile.n_size_label}$ & diag.\ ratio & rel.\ Frob. "
            r"& Stein \\"
        ),
        r"\midrule",
    ]
    for batch_key in profile.table_order:
        if batch_key not in summary["batches"]:
            continue
        block = summary["batches"][batch_key]
        ref_n = summary["reference_n"]
        if batch_key == profile.null_key:
            label = f"{ref_n}$^{{\\mathrm{{null}}}}$"
        else:
            label = batch_key
        cov = block["covariance_vs_ref"]
        diag_medians = [item["median"] for item in cov["diag_ratios"]]
        diag_cell = float(np.mean(diag_medians))
        lines.append(
            f"{label} & {diag_cell:.3f} & "
            f"{format_interval(cov['rel_frobenius'])} & "
            f"{format_interval(cov['stein'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="ascii")


def check_complete_covariance_profile(batches: dict[str, dict]) -> None:
    """Require positive-definite covariance estimates at every profile point."""
    invalid = invalid_profile_indices(batches)
    if len(invalid):
        incomplete = [
            f"{batch_key}: {np.flatnonzero(~batch['ok']).tolist()}"
            for batch_key, batch in batches.items()
            if not np.all(batch["ok"])
        ]
        raise RuntimeError(
            "N-stability covariance profile is incomplete at indices "
            f"{invalid.tolist()} ({'; '.join(incomplete)})"
        )


def aggregate_all(config: StudyConfig, batches: dict[str, dict]) -> dict:
    n_summaries = config.n_summaries
    profile = config.profile

    plot_payload: dict[str, np.ndarray] = {}
    summary_batches: dict[str, dict] = {}
    ref = batches[str(profile.ref_n)]

    for batch_key, batch in batches.items():
        rows = []
        stein_vals = np.full(config.n_theta, np.nan)
        fro_vals = np.full(config.n_theta, np.nan)
        diag_vals = np.full((config.n_theta, n_summaries), np.nan)
        for theta_index in range(config.n_theta):
            comparison = compare_to_reference(
                batch["c1"][theta_index],
                ref["c1"][theta_index],
                bool(batch["ok"][theta_index]),
                bool(ref["ok"][theta_index]),
                n_summaries,
            )
            rows.append(comparison)
            if comparison["valid"]:
                stein_vals[theta_index] = comparison["stein"]
                fro_vals[theta_index] = comparison["rel_frobenius"]
                diag_vals[theta_index] = comparison["diag_ratios"]

        ok_both = batch["ok"] & ref["ok"]
        summary_batches[batch_key] = {
            "n_size": int(batch["n_size"]),
            "n_ok": int(batch["ok"].sum()),
            "n_failed_replicates": int(batch["n_failed"].sum()),
            "covariance_vs_ref": summarize_comparisons(rows, n_summaries),
            "mean_shift_vs_ref": mean_shift_summary(
                batch["mean_std"], ref["mean_std"], ok_both, n_summaries
            ),
        }
        plot_payload[f"stein_{batch_key}"] = stein_vals
        plot_payload[f"frobenius_{batch_key}"] = fro_vals
        plot_payload[f"diag_ratios_{batch_key}"] = diag_vals

    summary = {
        "slug": config.slug,
        "seed": config.seed,
        "n_theta": config.n_theta,
        "n_replicates": config.n_replicates,
        "reference_n": profile.ref_n,
        "null_key": profile.null_key,
        "sample_size_label": profile.n_size_label,
        "summary_names": list(config.model.summary_names),
        "batches": summary_batches,
    }

    summary_path = config.results_dir / "n_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(f"[n_stability] Wrote {summary_path}")

    table_path = config.results_dir / "n_stability_table.tex"
    write_table_tex(summary, table_path, profile)
    print(f"[n_stability] Wrote {table_path}")

    plot_path = config.results_dir / "n_stability_plot_data.npz"
    np.savez_compressed(plot_path, **plot_payload, params=ref["params"])
    print(f"[n_stability] Wrote {plot_path}")
    return summary


def run_study(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    config = build_config(args)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    transform_path = config.results_dir / "target_transform.pkl"
    onnx_path = config.results_dir / "model.onnx"
    if not transform_path.exists() or not onnx_path.exists():
        print(
            f"[n_stability] FAIL: need {transform_path} and {onnx_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    load_target_transform(config.slug)

    params = draw_fixed_thetas(
        config.model, config.n_theta, config.seed, config.profile
    )
    batches: dict[str, dict] = {}
    profile = config.profile
    for n_size in profile.n_values:
        batches[str(n_size)] = run_batch(config, params, str(n_size), n_size)
    batches[profile.null_key] = run_batch(
        config, params, profile.null_key, profile.ref_n
    )

    replace_invalid_random_profile_points(config, params, batches)
    check_complete_covariance_profile(batches)
    summary = aggregate_all(config, batches)
    print(f"[n_stability] Done ({config.slug}).")
    for key, block in summary["batches"].items():
        cov = block["covariance_vs_ref"]
        mean_shift = block["mean_shift_vs_ref"]
        print(
            f"  N={key}: mean_l2 med={mean_shift['l2']['median']:.4f}, "
            f"stein med={cov['stein']['median']:.4f}, "
            f"fro med={cov['rel_frobenius']['median']:.4f}"
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    run_study(argv)


if __name__ == "__main__":
    main()

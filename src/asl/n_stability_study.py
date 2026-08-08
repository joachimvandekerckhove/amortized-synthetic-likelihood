"""
Multi-N C1 stability evaluation for DDM emulators.

Draws fixed parameter vectors, simulates replicate datasets at several N,
estimates C1(theta, N) = N * Cov(S_N | theta) in frozen target_transform
space, and compares each N against an N=600 reference plus an independent
N=600 null batch. Also reports frozen-emulator Mahalanobis calibration.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy import stats

from asl.cholesky import load_emulator_error_cov
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
    sigma_total_from_emulator,
    stein_discrepancy,
)
from asl.ort_env import cpu_inference_session
from asl.spec import Model
from models.catalog import get_model

SUPPORTED_SLUGS = ("ddm3", "ddm4", "ddmcollapsesig")
DEFAULT_N_VALUES = (50, 100, 300, 600, 1000)
REF_N = 600
NULL_KEY = "600_null"


@dataclass(frozen=True)
class StudyConfig:
    """Fixed settings for one stability evaluation run."""

    slug: str
    n_theta: int
    n_replicates: int
    n_values: tuple[int, ...]
    seed: int
    workers: int
    results_dir: Path

    @property
    def model(self) -> Model:
        return get_model(self.slug)

    @property
    def n_summaries(self) -> int:
        return self.model.n_summaries

    @property
    def chi2_p95(self) -> float:
        return float(stats.chi2.ppf(0.95, df=self.n_summaries))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-trial covariance stability across N."
    )
    parser.add_argument(
        "--slug",
        required=True,
        choices=SUPPORTED_SLUGS,
        help="DDM emulator slug (ddm3, ddm4, or ddmcollapsesig).",
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
    if args.quick:
        n_theta, n_replicates = 8, 40
    else:
        n_theta, n_replicates = 200, 500
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
        n_values=DEFAULT_N_VALUES,
        seed=args.seed,
        workers=workers,
        results_dir=results_dir,
    )


def draw_fixed_thetas(model: Model, n_theta: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = np.empty((n_theta, model.n_params), dtype=np.float64)
    for i, (lo, hi) in enumerate(model.param_bounds):
        params[:, i] = rng.uniform(lo, hi, size=n_theta)
    return params


def stable_tag_code(batch_tag: str) -> int:
    code = 0
    for character in batch_tag:
        code = (code * 131 + ord(character)) % 1_000_003
    return code


def batch_seed(master_seed: int, batch_tag: str, theta_index: int) -> int:
    return int(master_seed + 1_000_003 * stable_tag_code(batch_tag) + 10_007 * theta_index)


def raw_checkpoint_path(results_dir: Path, batch_key: str) -> Path:
    return results_dir / f"n_stability_raw_N{batch_key}.npz"


def _simulate_theta_batch(args: tuple) -> dict:
    slug, theta_index, params, n_trials, n_replicates, base_seed, transform_path = args
    model = get_model(slug)
    with open(transform_path, "rb") as handle:
        target_transform = pickle.load(handle)

    summaries_std = np.full((n_replicates, model.n_summaries), np.nan, dtype=np.float64)
    n_failed = 0
    for replicate_index in range(n_replicates):
        summaries = model.simulate_summaries(
            params, n_trials, base_seed + replicate_index
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
    c1 = estimate_c1(valid_rows, n_trials)
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
    n_trials: int,
) -> dict:
    path = raw_checkpoint_path(config.results_dir, batch_key)
    if path.exists():
        print(f"[n_stability] Loading checkpoint {path.name}")
        loaded = np.load(path, allow_pickle=False)
        payload = {key: loaded[key] for key in loaded.files}
        payload["batch_key"] = batch_key
        payload["n_trials"] = n_trials
        return payload

    n_summaries = config.n_summaries
    transform_path = str(config.results_dir / "target_transform.pkl")
    work_items = [
        (
            config.slug,
            theta_index,
            params[theta_index],
            n_trials,
            config.n_replicates,
            batch_seed(config.seed, batch_key, theta_index),
            transform_path,
        )
        for theta_index in range(config.n_theta)
    ]

    print(
        f"[n_stability] {config.slug} batch {batch_key} "
        f"(N={n_trials}, thetas={config.n_theta}, R={config.n_replicates}, "
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
        "params": params,
        "c1": c1,
        "mean_std": mean_std,
        "summaries_std": summaries_std.astype(np.float32),
        "n_valid": n_valid,
        "n_failed": n_failed,
        "ok": ok,
        "n_trials": np.array(n_trials),
        "batch_key": np.array(batch_key),
        "seed": np.array(config.seed),
        "n_replicates": np.array(config.n_replicates),
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[n_stability] Wrote {path}")
    payload["batch_key"] = batch_key
    payload["n_trials"] = n_trials
    return payload


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


def mahalanobis_for_batch(
    batch: dict,
    session,
    sigma_emu: np.ndarray,
    n_summaries: int,
) -> dict[str, np.ndarray]:
    params = batch["params"]
    summaries_std = np.asarray(batch["summaries_std"], dtype=np.float64)
    n_trials = int(batch["n_trials"])
    n_theta, n_replicates, _ = summaries_std.shape

    d2 = np.full((n_theta, n_replicates), np.nan, dtype=np.float64)
    for theta_index in range(n_theta):
        pred = session.run(
            None,
            {"input": params[theta_index].astype(np.float32).reshape(1, -1)},
        )[0][0]
        mu_std = pred[:n_summaries]
        chol_upper = pred[n_summaries:]
        try:
            sigma_total = sigma_total_from_emulator(chol_upper, sigma_emu, n_trials)
            precision = np.linalg.inv(sigma_total)
        except np.linalg.LinAlgError:
            continue
        rows = summaries_std[theta_index]
        valid = np.all(np.isfinite(rows), axis=1)
        if not np.any(valid):
            continue
        residuals = rows[valid] - mu_std
        d2[theta_index, valid] = np.einsum(
            "ij,jk,ik->i", residuals, precision, residuals
        )
    return {"d2": d2}


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


def summarize_mahalanobis(d2: np.ndarray, n_summaries: int, chi2_p95: float) -> dict:
    flat = d2[np.isfinite(d2)]
    if flat.size == 0:
        return {
            "n_obs": 0,
            "mean_d2_over_p": float("nan"),
            "median_d2_over_p": float("nan"),
            "coverage_95": float("nan"),
        }
    return {
        "n_obs": int(flat.size),
        "mean_d2_over_p": float(flat.mean() / n_summaries),
        "median_d2_over_p": float(np.median(flat) / n_summaries),
        "coverage_95": float(np.mean(flat <= chi2_p95)),
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


def write_table_tex(summary: dict, path: Path) -> None:
    n_summaries = len(summary["summary_names"])
    ordered_keys = ["50", "100", "300", "600", NULL_KEY, "1000"]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        (
            rf"$N$ & diag.\ ratio & rel.\ Frob. "
            rf"& Stein & $E[D^2]/{n_summaries}$/cov$_{{95}}$ \\"
        ),
        r"\midrule",
    ]
    for batch_key in ordered_keys:
        if batch_key not in summary["batches"]:
            continue
        block = summary["batches"][batch_key]
        label = "600$^{\\mathrm{null}}$" if batch_key == NULL_KEY else batch_key
        cov = block["covariance_vs_ref"]
        maha = block["mahalanobis"]
        diag_medians = [item["median"] for item in cov["diag_ratios"]]
        diag_cell = float(np.mean(diag_medians))
        lines.append(
            f"{label} & {diag_cell:.3f} & "
            f"{format_interval(cov['rel_frobenius'])} & "
            f"{format_interval(cov['stein'])} & "
            f"{maha['mean_d2_over_p']:.3f} / {maha['coverage_95']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="ascii")


def aggregate_all(config: StudyConfig, batches: dict[str, dict]) -> dict:
    session = cpu_inference_session(config.results_dir / "model.onnx")
    sigma_emu = load_emulator_error_cov(config.slug)
    n_summaries = config.n_summaries
    chi2_p95 = config.chi2_p95

    maha_by_key = {
        key: mahalanobis_for_batch(batch, session, sigma_emu, n_summaries)
        for key, batch in batches.items()
    }

    plot_payload: dict[str, np.ndarray] = {}
    summary_batches: dict[str, dict] = {}
    ref = batches[str(REF_N)]

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
            "n_trials": int(batch["n_trials"]),
            "n_ok": int(batch["ok"].sum()),
            "n_failed_replicates": int(batch["n_failed"].sum()),
            "covariance_vs_ref": summarize_comparisons(rows, n_summaries),
            "mahalanobis": summarize_mahalanobis(
                maha_by_key[batch_key]["d2"], n_summaries, chi2_p95
            ),
            "mean_shift_vs_ref": mean_shift_summary(
                batch["mean_std"], ref["mean_std"], ok_both, n_summaries
            ),
        }
        plot_payload[f"stein_{batch_key}"] = stein_vals
        plot_payload[f"frobenius_{batch_key}"] = fro_vals
        plot_payload[f"diag_ratios_{batch_key}"] = diag_vals
        plot_payload[f"d2_{batch_key}"] = maha_by_key[batch_key]["d2"]

    summary = {
        "slug": config.slug,
        "seed": config.seed,
        "n_theta": config.n_theta,
        "n_replicates": config.n_replicates,
        "reference_n": REF_N,
        "null_key": NULL_KEY,
        "summary_names": list(config.model.summary_names),
        "chi2_p95": chi2_p95,
        "batches": summary_batches,
    }

    summary_path = config.results_dir / "n_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(f"[n_stability] Wrote {summary_path}")

    table_path = config.results_dir / "n_stability_table.tex"
    write_table_tex(summary, table_path)
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

    params = draw_fixed_thetas(config.model, config.n_theta, config.seed)
    batches: dict[str, dict] = {}
    for n_trials in config.n_values:
        batches[str(n_trials)] = run_batch(config, params, str(n_trials), n_trials)
    batches[NULL_KEY] = run_batch(config, params, NULL_KEY, REF_N)

    summary = aggregate_all(config, batches)
    print(f"[n_stability] Done ({config.slug}).")
    for key, block in summary["batches"].items():
        cov = block["covariance_vs_ref"]
        maha = block["mahalanobis"]
        print(
            f"  N={key}: stein med={cov['stein']['median']:.4f}, "
            f"fro med={cov['rel_frobenius']['median']:.4f}, "
            f"E[D2]/p={maha['mean_d2_over_p']:.3f}, "
            f"cov95={maha['coverage_95']:.3f}"
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    run_study(argv)


if __name__ == "__main__":
    main()

"""
asl.cov_data -- Replicate-based per-trial covariance training data.

For each parameter draw theta, simulates R replicate datasets at n_rep trials,
computes log1p-space summary means and per-trial covariances, and streams rows
to data/<slug>/cov_train.csv.

Usage:
    Called from scripts/<model>/run.py generate-data

Configuration:
    [cov_data] parameter_draws, trials_per_replicate, replicates_per_parameter,
               random_seed, parallel_workers, parameter_mi_gate  (in asl.toml)
"""

import json
import sys
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from asl.config import load_config
from asl.data import summary_column_masks
from asl.mi_joint import joint_mi_ksg
from asl.cholesky import pack_upper_tri, upper_tri_index_pairs
from asl.spec import Model
from models.catalog import get_model

SEED_DEFAULT = 1
N_THETA = 20_000
N_REP = 600
R = 120
CHUNK_SIZE = 200
MIN_SUMMARY_VARIANCE_DEFAULT = 1.0e-12
SUMMARY_MI_PERMUTATIONS_DEFAULT = 30
SUMMARY_MI_SUBSAMPLE_DEFAULT = 5000
SUMMARY_MI_QUANTILE_DEFAULT = 0.95
SUMMARY_MI_NEIGHBORS_DEFAULT = 5


def cov_settings_path(slug: str) -> Path:
    """Path to metadata describing how cov_train.csv was generated."""
    return Path("data") / slug / "cov_settings.json"


def save_cov_settings(slug: str, n_rep: int, n_replicates: int, seed: int) -> None:
    """Persist replicate counts used to build cov_train.csv."""
    path = cov_settings_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {"n_rep": n_rep, "R": n_replicates, "seed": seed},
            f,
            indent=2,
        )


def load_cov_settings(slug: str) -> tuple[int, int]:
    """Return (n_rep, R) used for cov_train.csv."""
    path = cov_settings_path(slug)
    if path.exists():
        with open(path) as f:
            payload = json.load(f)
        return int(payload["n_rep"]), int(payload["R"])
    return N_REP, R


def resolve_cov_settings() -> tuple[int, int, int, int]:
    """Return n_theta, n_rep, R, seed from TOML configuration."""
    config = load_config()
    n_theta = int(config.get("cov_data", "parameter_draws", N_THETA))
    n_rep = int(config.get("cov_data", "trials_per_replicate", N_REP))
    n_r = int(config.get("cov_data", "replicates_per_parameter", R))
    seed = int(config.get("cov_data", "random_seed", SEED_DEFAULT))
    return n_theta, n_rep, n_r, seed


def resolve_cov_workers() -> int:
    """Return parallel worker count from TOML or CPU default."""
    config = load_config()
    workers = int(config.get("cov_data", "parallel_workers", 0))
    if workers > 0:
        return workers
    return max(1, int(cpu_count() * 0.9))


def resolve_cov_qa_settings() -> tuple[float, int, int, float, int]:
    """Return summary QA thresholds from TOML."""
    config = load_config()
    min_var = float(config.get("cov_data", "min_summary_variance", MIN_SUMMARY_VARIANCE_DEFAULT))
    n_perm = int(config.get("cov_data", "summary_mi_permutations", SUMMARY_MI_PERMUTATIONS_DEFAULT))
    subsample = int(config.get("cov_data", "summary_mi_subsample", SUMMARY_MI_SUBSAMPLE_DEFAULT))
    quantile = float(config.get("cov_data", "summary_mi_quantile", SUMMARY_MI_QUANTILE_DEFAULT))
    neighbors = int(config.get("cov_data", "summary_mi_neighbors", SUMMARY_MI_NEIGHBORS_DEFAULT))
    return min_var, n_perm, subsample, quantile, neighbors


def parameter_mi_gate_enabled() -> bool:
    """Return whether the joint parameter MI feasibility gate is active."""
    config = load_config()
    return bool(config.get("cov_data", "parameter_mi_gate", True))


def check_summary_variance_gate(
    y_raw: np.ndarray, model: Model, min_var: float = MIN_SUMMARY_VARIANCE_DEFAULT
) -> None:
    """Require every summary mean to vary across training rows."""
    failures: list[str] = []
    for j, name in enumerate(model.summary_names):
        var = float(np.var(y_raw[:, j]))
        if var < min_var:
            failures.append(f"{name} (var={var:.2e})")
    if failures:
        msg = "Summary variance gate failed for: " + ", ".join(failures)
        print(f"[cov_data] FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def _summary_parameter_mi_vector(
    summary_col: np.ndarray,
    param_cols: np.ndarray,
    *,
    neighbors: int,
    random_state: int,
) -> np.ndarray:
    """MI between one summary and each parameter column."""
    mi = np.empty(param_cols.shape[1], dtype=np.float64)
    for j in range(param_cols.shape[1]):
        mi[j] = mutual_info_regression(
            param_cols[:, [j]],
            summary_col,
            random_state=random_state,
            n_neighbors=neighbors,
        )[0]
    return mi


def _max_summary_parameter_mi(
    summary_col: np.ndarray,
    param_cols: np.ndarray,
    *,
    neighbors: int,
    random_state: int,
) -> float:
    """Maximum MI between one summary and any parameter."""
    best = 0.0
    for j in range(param_cols.shape[1]):
        mi = mutual_info_regression(
            param_cols[:, [j]],
            summary_col,
            random_state=random_state,
            n_neighbors=neighbors,
        )[0]
        best = max(best, float(mi))
    return best


def _summary_mi_threshold(
    summary_col: np.ndarray,
    param_cols: np.ndarray,
    *,
    n_perm: int,
    neighbors: int,
    quantile: float,
    random_state: int,
) -> float:
    """Permutation null for summary->parameter MI (shuffle summary labels)."""
    rng = np.random.default_rng(random_state)
    nulls = []
    for _ in range(n_perm):
        perm = summary_col.copy()
        rng.shuffle(perm)
        nulls.append(
            _max_summary_parameter_mi(
                perm, param_cols, neighbors=neighbors, random_state=random_state
            )
        )
    return float(np.quantile(nulls, quantile))


def _parameter_joint_mi(
    param_col: np.ndarray,
    summary_cols: np.ndarray,
    *,
    neighbors: int,
) -> float:
    """MI between one parameter and the full summary vector."""
    return joint_mi_ksg(param_col, summary_cols, k=neighbors)


def _parameter_joint_mi_threshold(
    param_col: np.ndarray,
    summary_cols: np.ndarray,
    *,
    n_perm: int,
    neighbors: int,
    quantile: float,
    random_state: int,
) -> float:
    """Permutation null for I(theta; S) (shuffle parameter labels)."""
    rng = np.random.default_rng(random_state)
    nulls = []
    for _ in range(n_perm):
        perm = param_col.copy()
        rng.shuffle(perm)
        nulls.append(_parameter_joint_mi(perm, summary_cols, neighbors=neighbors))
    return float(np.quantile(nulls, quantile))


def report_parameter_mi_gate(
    y_raw: np.ndarray,
    X: np.ndarray,
    model: Model,
    *,
    n_perm: int = SUMMARY_MI_PERMUTATIONS_DEFAULT,
    subsample: int = SUMMARY_MI_SUBSAMPLE_DEFAULT,
    quantile: float = SUMMARY_MI_QUANTILE_DEFAULT,
    neighbors: int = SUMMARY_MI_NEIGHBORS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Return per-parameter joint MI diagnostics for the hard gate."""
    n_rows = len(y_raw)
    if subsample is not None and n_rows > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_rows, size=subsample, replace=False)
        y_sub = y_raw[idx]
        x_sub = X[idx]
        n_used = subsample
    else:
        y_sub = y_raw
        x_sub = X
        n_used = n_rows

    parameter_reports: list[dict] = []
    failures: list[str] = []
    for j, name in enumerate(model.param_names):
        param_col = x_sub[:, j]
        mi_joint = _parameter_joint_mi(param_col, y_sub, neighbors=neighbors)
        threshold = _parameter_joint_mi_threshold(
            param_col,
            y_sub,
            n_perm=n_perm,
            neighbors=neighbors,
            quantile=quantile,
            random_state=seed + j + 1,
        )
        passes = mi_joint > threshold
        if not passes:
            failures.append(f"{name} (mi_joint={mi_joint:.4f}, thr={threshold:.4f})")
        parameter_reports.append(
            {
                "name": name,
                "mi_joint": mi_joint,
                "threshold": threshold,
                "passes": passes,
            }
        )

    return {
        "model": model.slug,
        "n_rows_total": n_rows,
        "n_rows_used": n_used,
        "settings": {
            "n_perm": n_perm,
            "subsample": subsample,
            "quantile": quantile,
            "neighbors": neighbors,
            "seed": seed,
        },
        "parameters": parameter_reports,
        "gate_passes": not failures,
        "failures": failures,
    }


def check_parameter_mi_gate(
    y_raw: np.ndarray,
    X: np.ndarray,
    model: Model,
    *,
    n_perm: int = SUMMARY_MI_PERMUTATIONS_DEFAULT,
    subsample: int = SUMMARY_MI_SUBSAMPLE_DEFAULT,
    quantile: float = SUMMARY_MI_QUANTILE_DEFAULT,
    neighbors: int = SUMMARY_MI_NEIGHBORS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> None:
    """Require each parameter to carry detectable MI with the summary vector jointly."""
    report = report_parameter_mi_gate(
        y_raw,
        X,
        model,
        n_perm=n_perm,
        subsample=subsample,
        quantile=quantile,
        neighbors=neighbors,
        seed=seed,
    )
    if not report["gate_passes"]:
        msg = "Parameter MI gate failed for: " + "; ".join(report["failures"])
        print(f"[cov_data] FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def report_summary_mi_diagnostic(
    y_raw: np.ndarray,
    X: np.ndarray,
    model: Model,
    *,
    n_perm: int = SUMMARY_MI_PERMUTATIONS_DEFAULT,
    subsample: int = SUMMARY_MI_SUBSAMPLE_DEFAULT,
    quantile: float = SUMMARY_MI_QUANTILE_DEFAULT,
    neighbors: int = SUMMARY_MI_NEIGHBORS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Return per-summary MI diagnostics (warning-only, not a hard gate)."""
    n_rows = len(y_raw)
    if subsample is not None and n_rows > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_rows, size=subsample, replace=False)
        y_sub = y_raw[idx]
        x_sub = X[idx]
        n_used = subsample
    else:
        y_sub = y_raw
        x_sub = X
        n_used = n_rows

    summary_reports: list[dict] = []
    failures: list[str] = []
    for j, name in enumerate(model.summary_names):
        summary_col = y_sub[:, j]
        mi_by_param = _summary_parameter_mi_vector(
            summary_col, x_sub, neighbors=neighbors, random_state=seed
        )
        mi_max = float(np.max(mi_by_param))
        best_idx = int(np.argmax(mi_by_param))
        threshold = _summary_mi_threshold(
            summary_col,
            x_sub,
            n_perm=n_perm,
            neighbors=neighbors,
            quantile=quantile,
            random_state=seed + j + 1,
        )
        passes = mi_max > threshold
        if not passes:
            failures.append(f"{name} (mi_max={mi_max:.4f}, thr={threshold:.4f})")
        summary_reports.append(
            {
                "name": name,
                "variance": float(np.var(y_sub[:, j])),
                "mi_max": mi_max,
                "best_parameter": model.param_names[best_idx],
                "threshold": threshold,
                "passes": passes,
                "mi_by_parameter": {
                    param_name: float(mi_by_param[k])
                    for k, param_name in enumerate(model.param_names)
                },
            }
        )

    return {
        "model": model.slug,
        "n_rows_total": n_rows,
        "n_rows_used": n_used,
        "settings": {
            "n_perm": n_perm,
            "subsample": subsample,
            "quantile": quantile,
            "neighbors": neighbors,
            "seed": seed,
        },
        "summaries": summary_reports,
        "gate_passes": not failures,
        "failures": failures,
    }


def warn_summary_mi_diagnostic(
    y_raw: np.ndarray,
    X: np.ndarray,
    model: Model,
    *,
    n_perm: int = SUMMARY_MI_PERMUTATIONS_DEFAULT,
    subsample: int = SUMMARY_MI_SUBSAMPLE_DEFAULT,
    quantile: float = SUMMARY_MI_QUANTILE_DEFAULT,
    neighbors: int = SUMMARY_MI_NEIGHBORS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> list[str]:
    """Warn when a summary has no detectable MI with any parameter."""
    report = report_summary_mi_diagnostic(
        y_raw,
        X,
        model,
        n_perm=n_perm,
        subsample=subsample,
        quantile=quantile,
        neighbors=neighbors,
        seed=seed,
    )
    warnings = report["failures"]
    for item in warnings:
        print(f"[cov_data] WARN: summary MI diagnostic: {item}")
    return warnings


def validate_cov_training_data(
    X: np.ndarray,
    y_raw: np.ndarray,
    model: Model,
    *,
    seed: int = SEED_DEFAULT,
    data_path: Path | None = None,
    write_reports: bool = True,
) -> list[str]:
    """Run post-generation QA gates on covariance training data."""
    min_var, n_perm, subsample, quantile, neighbors = resolve_cov_qa_settings()
    check_summary_variance_gate(y_raw, model, min_var=min_var)

    param_report = report_parameter_mi_gate(
        y_raw,
        X,
        model,
        n_perm=n_perm,
        subsample=subsample,
        quantile=quantile,
        neighbors=neighbors,
        seed=seed,
    )
    summary_report = report_summary_mi_diagnostic(
        y_raw,
        X,
        model,
        n_perm=n_perm,
        subsample=subsample,
        quantile=quantile,
        neighbors=neighbors,
        seed=seed,
    )

    if parameter_mi_gate_enabled():
        if not param_report["gate_passes"]:
            msg = "Parameter MI gate failed for: " + "; ".join(param_report["failures"])
            print(f"[cov_data] FAIL: {msg}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[cov_data] Parameter MI gate disabled.")

    warnings = summary_report["failures"]
    for item in warnings:
        print(f"[cov_data] WARN: summary MI diagnostic: {item}")

    if write_reports:
        resolved_data_path = data_path or cov_train_path(model.slug)
        param_path, summary_path = save_cov_mi_reports(
            model.slug,
            param_report,
            summary_report,
            data_path=resolved_data_path,
            y_raw=y_raw,
            model=model,
            min_var=min_var,
        )
        print(f"[cov_data] Wrote parameter MI report: {param_path}")
        print(f"[cov_data] Wrote summary MI report: {summary_path}")

    return warnings


def c1_column_names(n_summaries: int) -> list[str]:
    """Column names for upper-triangular per-trial covariance entries."""
    return [f"c1_{i}_{j}" for i, j in upper_tri_index_pairs(n_summaries)]


def z_mean_column_names(summary_names: tuple[str, ...]) -> list[str]:
    """Column names for log1p-space summary means."""
    return [f"z_mean_{name}" for name in summary_names]


def draw_parameters(model: Model, rng: np.random.Generator) -> np.ndarray:
    """Draw one parameter vector for covariance training."""
    if model.draw_cov_parameters is not None:
        return model.draw_cov_parameters(rng)
    params = np.empty(model.n_params)
    for i, (lo, hi) in enumerate(model.param_bounds):
        params[i] = rng.uniform(lo, hi)
    return params


def summaries_to_logspace(summaries: np.ndarray, rt_mask: np.ndarray) -> np.ndarray:
    """Apply log1p to RT summary columns."""
    z = summaries.copy()
    z[rt_mask] = np.log1p(z[rt_mask])
    return z


def logspace_to_raw(z_mean: np.ndarray, rt_mask: np.ndarray) -> np.ndarray:
    """Invert log1p on RT columns for R^2 evaluation in physical units."""
    y_raw = z_mean.copy()
    y_raw[rt_mask] = np.expm1(y_raw[rt_mask])
    return y_raw


def cov_train_path(slug: str) -> Path:
    """Path to replicate-based covariance training data."""
    return Path("data") / slug / "cov_train.csv"


def parameter_mi_report_path(slug: str) -> Path:
    """Path to the hard joint parameter MI gate report."""
    return Path("results") / slug / "parameter_mi_report.json"


def summary_mi_report_path(slug: str) -> Path:
    """Path to the per-summary MI diagnostic report."""
    return Path("results") / slug / "summary_mi.json"


def generation_summary_path(slug: str) -> Path:
    """Path to metadata describing training-data generation statistics."""
    return Path("data") / slug / "generation_summary.json"


def save_generation_summary(slug: str, payload: dict) -> None:
    """Persist generation statistics for cov_train.csv."""
    path = generation_summary_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_parameter_mi_report(
    slug: str,
    report: dict,
    *,
    data_path: Path,
) -> Path:
    """Persist the joint parameter MI gate report."""
    path = parameter_mi_report_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **report,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path.resolve()),
        "estimator": "ksg_joint_mi",
        "statistic": "I(theta_j; S) for full summary vector S",
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def save_summary_mi_report(
    slug: str,
    report: dict,
    *,
    data_path: Path,
    y_raw: np.ndarray,
    model: Model,
    min_var: float,
) -> Path:
    """Persist the per-summary MI diagnostic report."""
    path = summary_mi_report_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **report,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path.resolve()),
        "variance_gate": {
            name: float(np.var(y_raw[:, j]))
            for j, name in enumerate(model.summary_names)
        },
        "min_summary_variance": min_var,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def save_cov_mi_reports(
    slug: str,
    param_report: dict,
    summary_report: dict,
    *,
    data_path: Path,
    y_raw: np.ndarray,
    model: Model,
    min_var: float,
) -> tuple[Path, Path]:
    """Write both MI QA reports for a cov_train.csv evaluation."""
    param_path = save_parameter_mi_report(slug, param_report, data_path=data_path)
    summary_path = save_summary_mi_report(
        slug,
        summary_report,
        data_path=data_path,
        y_raw=y_raw,
        model=model,
        min_var=min_var,
    )
    return param_path, summary_path


def _simulate_one_theta(args: tuple) -> tuple[np.ndarray | None, bool]:
    """Worker: simulate R replicates and return (row, replicate_rejected)."""
    slug, params, n_rep, n_r, base_seed = args
    model = get_model(slug)
    rt_mask, _ = summary_column_masks(model)
    n_summaries = model.n_summaries

    if n_r < 2:
        return None, True

    replicates = np.empty((n_r, n_summaries), dtype=np.float64)
    for r in range(n_r):
        summaries = model.simulate_summaries(params, n_rep, base_seed + r)
        if not np.all(np.isfinite(summaries)):
            return None, True
        replicates[r] = summaries_to_logspace(summaries, rt_mask)

    z_mean = replicates.mean(axis=0)
    C1_z = n_rep * np.cov(replicates, rowvar=False, bias=False)
    if not np.all(np.isfinite(C1_z)):
        return None, True

    return np.concatenate([params, z_mean, pack_upper_tri(C1_z)]), False


def expected_columns(model: Model) -> list[str]:
    """Return the ordered column names for cov_train.csv."""
    return (
        list(model.param_names)
        + z_mean_column_names(model.summary_names)
        + c1_column_names(model.n_summaries)
    )


def generate_cov_dataset(model: Model) -> None:
    """Generate replicate-based covariance training data."""
    slug = model.slug
    n_theta, n_rep, n_r, seed = resolve_cov_settings()
    n_workers = resolve_cov_workers()

    output_dir = Path("data") / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cov_train_path(slug)

    print(f"[cov_data] Model: {slug}")
    print(
        f"[cov_data] Target theta: {n_theta}, n_rep: {n_rep}, R: {n_r}, seed: {seed}"
    )
    print(f"[cov_data] Workers: {n_workers}")
    print(f"[cov_data] Output: {output_path}")

    param_rng = np.random.default_rng(seed)
    all_params = [draw_parameters(model, param_rng) for _ in range(n_theta)]
    work_items = [
        (slug, all_params[i], n_rep, n_r, seed + 10_000 + i * n_r)
        for i in range(n_theta)
    ]

    columns = expected_columns(model)
    valid_rows: list[np.ndarray] = []
    replicates_rejected = 0
    report_interval = max(1, n_theta // 20)
    processed = 0

    with Pool(processes=n_workers) as pool:
        for result, rejected in pool.imap_unordered(
            _simulate_one_theta, work_items, chunksize=CHUNK_SIZE
        ):
            processed += 1
            if rejected:
                replicates_rejected += 1
            if result is not None:
                valid_rows.append(result)

            if processed % report_interval == 0:
                pct = 100 * processed / n_theta
                print(
                    f"[cov_data] {pct:.0f}% processed, {len(valid_rows)} valid rows"
                )

    print(f"[cov_data] Writing {len(valid_rows)} valid rows to {output_path}")
    df = pd.DataFrame(valid_rows, columns=columns)
    df.to_csv(output_path, index=False)
    save_cov_settings(slug, n_rep, n_r, seed)
    print(f"[cov_data] Done. {len(valid_rows)} valid rows written.")

    if len(valid_rows) == 0:
        print("[cov_data] FAIL: No valid rows produced.", file=sys.stderr)
        sys.exit(1)

    rt_mask, _ = summary_column_masks(model)
    z_cols = z_mean_column_names(model.summary_names)
    qa_df = df[list(model.param_names) + z_cols]
    X_qa = qa_df[list(model.param_names)].values.astype(np.float64)
    z_mean_qa = qa_df[z_cols].values.astype(np.float64)
    y_raw_qa = np.array(
        [logspace_to_raw(row, rt_mask) for row in z_mean_qa], dtype=np.float64
    )
    print("[cov_data] Running training-data QA gates ...")
    summary_mi_warnings = validate_cov_training_data(
        X_qa,
        y_raw_qa,
        model,
        seed=seed,
        data_path=output_path,
    )
    print("[cov_data] Training-data QA gates passed.")

    save_generation_summary(
        slug,
        {
            "parameter_draws_attempted": n_theta,
            "parameter_rows_retained": len(valid_rows),
            "parameter_rows_rejected": n_theta - len(valid_rows),
            "replicates_attempted": n_theta * n_r,
            "replicates_rejected": replicates_rejected,
            "summary_mi_warnings": summary_mi_warnings,
        },
    )


def load_cov_dataset(
    model: Model, subsample: int | None = None, seed: int = SEED_DEFAULT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Model]:
    """Load cov_train.csv and return arrays for training."""
    slug = model.slug
    data_path = cov_train_path(slug)
    if not data_path.exists():
        raise FileNotFoundError(f"Covariance training data not found: {data_path}")

    df = pd.read_csv(data_path)
    expected_cols = expected_columns(model)
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Unexpected columns in {data_path}. "
            f"Expected {expected_cols}, got {list(df.columns)}"
        )

    mask = np.isfinite(df.values).all(axis=1)
    df = df.loc[mask].reset_index(drop=True)

    if subsample is not None and len(df) > subsample:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(df), size=subsample, replace=False)
        df = df.iloc[indices].reset_index(drop=True)

    rt_mask, _ = summary_column_masks(model)
    z_cols = z_mean_column_names(model.summary_names)
    c1_cols = c1_column_names(model.n_summaries)

    X = df[list(model.param_names)].values.astype(np.float32)
    z_mean = df[z_cols].values.astype(np.float32)
    C1_z = df[c1_cols].values.astype(np.float32)
    y_raw = np.array([logspace_to_raw(row, rt_mask) for row in z_mean], dtype=np.float32)
    validate_cov_training_data(
        X.astype(np.float64),
        y_raw.astype(np.float64),
        model,
        seed=seed,
        data_path=data_path,
    )
    return X, z_mean, C1_z, y_raw, model

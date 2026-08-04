"""
asl.cov_data -- Replicate-based per-trial covariance training data.

For each parameter draw theta, simulates R replicate datasets at n_rep trials,
computes log1p-space summary means and per-trial covariances, and streams rows
to data/<slug>/cov_train.csv.

Usage:
    Called from scripts/<model>/run.py generate-data

Configuration:
    [cov_data] parameter_draws, trials_per_replicate, replicates_per_parameter,
               random_seed, parallel_workers  (in asl.toml)
"""

import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from asl.config import load_config
from asl.data import summary_column_masks
from asl.cholesky import pack_upper_tri, upper_tri_index_pairs
from asl.spec import Model
from models.catalog import get_model

SEED_DEFAULT = 42
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


def check_summary_mi_gate(
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
    """Require each summary to carry detectable MI with at least one parameter."""
    n_rows = len(y_raw)
    if subsample is not None and n_rows > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_rows, size=subsample, replace=False)
        y_sub = y_raw[idx]
        x_sub = X[idx]
    else:
        y_sub = y_raw
        x_sub = X

    failures: list[str] = []
    for j, name in enumerate(model.summary_names):
        summary_col = y_sub[:, j]
        mi_max = _max_summary_parameter_mi(
            summary_col, x_sub, neighbors=neighbors, random_state=seed
        )
        threshold = _summary_mi_threshold(
            summary_col,
            x_sub,
            n_perm=n_perm,
            neighbors=neighbors,
            quantile=quantile,
            random_state=seed + j + 1,
        )
        if mi_max <= threshold:
            failures.append(f"{name} (mi_max={mi_max:.4f}, thr={threshold:.4f})")

    if failures:
        msg = "Summary MI gate failed for: " + "; ".join(failures)
        print(f"[cov_data] FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def validate_cov_training_data(
    X: np.ndarray,
    y_raw: np.ndarray,
    model: Model,
    *,
    seed: int = SEED_DEFAULT,
) -> None:
    """Run post-generation QA gates on covariance training data."""
    min_var, n_perm, subsample, quantile, neighbors = resolve_cov_qa_settings()
    check_summary_variance_gate(y_raw, model, min_var=min_var)
    check_summary_mi_gate(
        y_raw,
        X,
        model,
        n_perm=n_perm,
        subsample=subsample,
        quantile=quantile,
        neighbors=neighbors,
        seed=seed,
    )


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


def _simulate_one_theta(args: tuple) -> np.ndarray | None:
    """Worker: simulate R replicates and return one cov_train row or None."""
    slug, params, n_rep, n_r, base_seed = args
    model = get_model(slug)
    rt_mask, _ = summary_column_masks(model)
    n_summaries = model.n_summaries

    if n_r < 2:
        return None

    replicates = np.empty((n_r, n_summaries), dtype=np.float64)
    for r in range(n_r):
        summaries = model.simulate_summaries(params, n_rep, base_seed + r)
        if not np.all(np.isfinite(summaries)):
            return None
        replicates[r] = summaries_to_logspace(summaries, rt_mask)

    z_mean = replicates.mean(axis=0)
    C1_z = n_rep * np.cov(replicates, rowvar=False, bias=False)
    if not np.all(np.isfinite(C1_z)):
        return None

    return np.concatenate([params, z_mean, pack_upper_tri(C1_z)])


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
    output_path = output_dir / "cov_train.csv"

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
    report_interval = max(1, n_theta // 20)
    processed = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(
            _simulate_one_theta, work_items, chunksize=CHUNK_SIZE
        ):
            processed += 1
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
    validate_cov_training_data(X_qa, y_raw_qa, model, seed=seed)
    print("[cov_data] Training-data QA gates passed.")


def load_cov_dataset(
    model: Model, subsample: int | None = None, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Model]:
    """Load cov_train.csv and return arrays for training."""
    slug = model.slug
    data_path = Path("data") / slug / "cov_train.csv"
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
        X.astype(np.float64), y_raw.astype(np.float64), model, seed=seed
    )
    return X, z_mean, C1_z, y_raw, model

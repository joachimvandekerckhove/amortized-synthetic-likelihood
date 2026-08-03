"""Shared MCMC helpers for VPW08 hierarchical fits."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm

RHAT_GATE = float(os.environ.get("ASL_VPW08_RHAT_GATE", "1.05"))
N_ITER = int(os.environ.get("ASL_VPW08_N_ITER", "2500"))
N_BURNIN = int(os.environ.get("ASL_VPW08_N_BURNIN", "250"))
N_CHAINS = 4
SEED = 15
BF_EPS = 0.1


def force_rerun() -> bool:
    return os.environ.get("ASL_VPW08_FORCE", "0") == "1"


def converged_json(path: Path) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return bool(data.get("convergence", {}).get("converged"))


def _r_vector(vals: np.ndarray | list[float]) -> str:
    return "c(" + ", ".join(str(float(x)) for x in vals) + ")"


def get_param_samples(result, base: str, index: int | None = None) -> np.ndarray:
    candidates = [base]
    if index is not None:
        candidates = [f"{base}_{index}", f"{base}[{index}]", base]
    for name in candidates:
        try:
            return result.get_samples(name)
        except Exception:
            continue
    raise KeyError(f"Could not extract parameter {base} (index={index})")


def summarize_param(result, name: str, index: int | None = None) -> dict:
    samples = get_param_samples(result, name, index)
    rhat = float("nan")
    candidates = [name]
    if index is not None:
        candidates = [f"{name}_{index}", f"{name}[{index}]"]
    for candidate in candidates:
        try:
            rhat = float(result.rhat(candidate))
            break
        except Exception:
            continue
    return {
        "mean": float(np.mean(samples)),
        "sd": float(np.std(samples)),
        "lo95": float(np.percentile(samples, 2.5)),
        "hi95": float(np.percentile(samples, 97.5)),
        "rhat": rhat,
    }


def gamma_samples(result) -> np.ndarray:
    return np.column_stack([get_param_samples(result, "gamma", i) for i in range(1, 5)])


def summarize_gamma(result) -> dict[str, dict]:
    draws = gamma_samples(result)
    out: dict[str, dict] = {}
    for i in range(1, 5):
        samples = draws[:, i - 1]
        rhat = float("nan")
        for name in (f"gamma_{i}", f"gamma[{i}]"):
            try:
                rhat = float(result.rhat(name))
                break
            except Exception:
                continue
        out[f"gamma_{i}"] = {
            "mean": float(np.mean(samples)),
            "sd": float(np.std(samples)),
            "lo95": float(np.percentile(samples, 2.5)),
            "hi95": float(np.percentile(samples, 97.5)),
            "rhat": rhat,
        }
    return out


def bayes_factor_null(gamma_draws: np.ndarray, epsilon: float = BF_EPS) -> float:
    prior_null_mass = norm.cdf(epsilon) - norm.cdf(-epsilon)
    post_null_mass = float(np.mean((gamma_draws > -epsilon) & (gamma_draws < epsilon)))
    if post_null_mass <= 0:
        return float("inf")
    return prior_null_mass / post_null_mass


def bayes_factors_gamma(result) -> dict[str, float]:
    draws = gamma_samples(result)
    return {
        f"gamma_{i}": float(bayes_factor_null(draws[:, i - 1]))
        for i in range(1, 4)
    }


def convergence_block(result) -> dict:
    max_rhat = float(result.max_rhat())
    return {
        "max_rhat": max_rhat,
        "converged": bool(max_rhat < RHAT_GATE),
        "rhat_gate": RHAT_GATE,
    }


def hierarchical_data(cells: dict) -> dict:
    return {
        "K": cells["K"],
        "cond": cells["cond"].tolist(),
        "n_trials": cells["n_trials"].tolist(),
        "obs": cells["obs_raw"].tolist(),
        "A": cells["A"].tolist(),
        "B": cells["B"].tolist(),
        "C": cells["C"].tolist(),
    }


def ensure_jags_runtime() -> None:
    import os
    from pathlib import Path

    from asl.onnxruntime_sdk import (
        _sdk_is_valid,
        ensure_onnxruntime_lib_on_path,
        find_repo_root,
    )

    root = find_repo_root()
    env_dir = os.environ.get("ONNXRUNTIME_DIR", "").strip()
    if env_dir and not _sdk_is_valid(Path(env_dir)):
        os.environ.pop("ONNXRUNTIME_DIR", None)
    ensure_onnxruntime_lib_on_path(root)

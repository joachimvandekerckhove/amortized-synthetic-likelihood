"""Posterior predictive helpers for VPW08 fits (raw summary scale)."""

from __future__ import annotations

import numpy as np


def extract_obs_rep(result, k_cells: int, n_summ: int) -> np.ndarray:
    """Extract obs_rep MCMC samples with shape (n_draws, K, n_summ)."""
    sample0 = result.get_samples("obs_rep_1_1")
    n_draws = len(sample0)
    arr = np.empty((n_draws, k_cells, n_summ), dtype=np.float64)
    for cell in range(k_cells):
        for j in range(n_summ):
            arr[:, cell, j] = result.get_samples(f"obs_rep_{cell + 1}_{j + 1}")
    return arr


def summarize_ppc(obs_raw: np.ndarray, ppc_raw: np.ndarray, *, summary_names: list[str]) -> dict:
    """Summarize posterior predictive checks per summary statistic."""
    summary: dict = {"summary_names": list(summary_names), "by_stat": {}}
    for j, name in enumerate(summary_names):
        obs_j = obs_raw[:, j]
        ppc_j = ppc_raw[:, :, j]
        outside95 = np.array(
            [
                obs_j[c] < np.percentile(ppc_j[:, c], 2.5)
                or obs_j[c] > np.percentile(ppc_j[:, c], 97.5)
                for c in range(obs_raw.shape[0])
            ]
        )
        summary["by_stat"][name] = {
            "n_cells": int(obs_raw.shape[0]),
            "frac_outside_95": float(np.mean(outside95)),
        }
    return summary

"""KSG k-nearest-neighbor estimate of I(X; Y) for continuous variables."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import rankdata


def rank_normalize_columns(values: np.ndarray) -> np.ndarray:
    """Map every column to average ranks on the closed interval [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite values.")
    if len(values) < 2:
        raise ValueError("Need at least two rows to normalize ranks.")

    ranks = rankdata(values, method="average", axis=0)
    return (ranks - 1.0) / (len(values) - 1.0)


def joint_mi_ksg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 5,
) -> float:
    """Estimate mutual information I(X; Y) with the Kraskov-Stogbauer-Grassberger estimator.

    Parameters
    ----------
    x
        Shape (n,) or (n, d_x).
    y
        Shape (n,) or (n, d_y).
    k
        Number of nearest neighbors in the joint space.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows.")
    n = x.shape[0]
    if n <= k:
        raise ValueError(f"Need more than k={k} samples (got n={n}).")

    x = rank_normalize_columns(x)
    y = rank_normalize_columns(y)
    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = np.nextafter(dists[:, -1], 0.0)

    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)
    for i in range(n):
        nx[i] = len(tree_x.query_ball_point(x[i], eps[i], p=np.inf)) - 1
        ny[i] = len(tree_y.query_ball_point(y[i], eps[i], p=np.inf)) - 1

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(max(mi, 0.0))

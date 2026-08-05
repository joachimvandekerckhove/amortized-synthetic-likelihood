"""Recovery diagnostic figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from asl.spec import Model

COLUMN_WIDTH_IN = 3.45
ERRORBAR_MS = 2.5
ERRORBAR_CAPSIZE = 1.5
RECOVERY_MARKER_ALPHA = 0.3
RECOVERY_WHISKER_ALPHA = 0.1
IDENTITY_LINE_KW = {"color": "#222222", "linewidth": 0.7, "alpha": 0.55}

SERIES_COLORS = (
    "#4c72b0",
    "#55a868",
    "#dd8452",
    "#8172b3",
    "#8c8c8c",
    "#c44e52",
)
SERIES_MARKERS = ("o", "s", "^", "D", "v", "x")

_RECOVERY_STYLE_INDEX: dict[str, int] = {
    "v": 0,
    "a": 1,
    "a0": 1,
    "t0": 2,
    "w": 3,
    "k": 4,
}
RECOVERY_PARAM_LABELS: dict[str, str] = {
    "v": r"$v$",
    "a": r"$a$",
    "a0": r"$a_0$",
    "t0": r"$t_0$",
    "w": r"$w$",
    "k": r"$k$",
}
RECOVERY_PARAM_SORT_ORDER: tuple[str, ...] = ("v", "a", "t0", "w", "k")


def _recovery_sort_key(param_key: str) -> int:
    key = "a" if param_key == "a0" else param_key
    try:
        return RECOVERY_PARAM_SORT_ORDER.index(key)
    except ValueError:
        return len(RECOVERY_PARAM_SORT_ORDER)


def canonical_recovery_sort_indices(
    param_names: list[str] | tuple[str, ...],
) -> list[int]:
    """Column indices for drift -> bound -> t0 -> other, top-left to bottom-right."""
    return sorted(range(len(param_names)), key=lambda i: _recovery_sort_key(param_names[i]))


def recovery_param_label(param_key: str) -> str:
    return RECOVERY_PARAM_LABELS.get(param_key, param_key)


def _series_style(index: int) -> dict[str, Any]:
    i = index % len(SERIES_COLORS)
    return {
        "color": SERIES_COLORS[i],
        "marker": SERIES_MARKERS[i % len(SERIES_MARKERS)],
    }


def recovery_param_style(param_key: str, panel_index: int = 0) -> dict[str, Any]:
    idx = _RECOVERY_STYLE_INDEX.get(param_key)
    if idx is None:
        idx = len(_RECOVERY_STYLE_INDEX) + panel_index
    return _series_style(idx)


def panel_grid(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    ncols = 2
    nrows = math.ceil(n / 2)
    return ncols, nrows


def column_figsize(ncols: int, nrows: int, *, aspect: float = 1.0, pad_in: float = 0.08) -> tuple[float, float]:
    panel_w = COLUMN_WIDTH_IN / ncols
    panel_h = panel_w * aspect
    return (COLUMN_WIDTH_IN, nrows * panel_h + pad_in)


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=2.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _identity_line(ax: plt.Axes, lo: float, hi: float) -> None:
    ax.plot([lo, hi], [lo, hi], **IDENTITY_LINE_KW)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_recovery_diagnostics(
    model: Model,
    true_params: np.ndarray,
    estimated_params: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    output_path: Path,
) -> None:
    """Create a 2-column paper-style recovery scatter PDF."""
    order = canonical_recovery_sort_indices(model.param_names)
    keys = tuple(model.param_names[i] for i in order)
    true = true_params[:, order]
    est = estimated_params[:, order]
    ci_lo = ci_lower[:, order]
    ci_hi = ci_upper[:, order]
    labels = tuple(recovery_param_label(k) for k in keys)

    n_params = len(keys)
    ncols, nrows = panel_grid(n_params)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=column_figsize(ncols, nrows),
        squeeze=False,
    )

    for i in range(n_params):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        _style_axes(ax)
        sty = recovery_param_style(keys[i], panel_index=i)
        x = true[:, i]
        y = est[:, i]
        err_lo = y - ci_lo[:, i]
        err_hi = ci_hi[:, i] - y
        ax.errorbar(
            x,
            y,
            yerr=[err_lo, err_hi],
            fmt="none",
            color=sty["color"],
            ecolor=sty["color"],
            elinewidth=0.6,
            capsize=ERRORBAR_CAPSIZE,
            alpha=RECOVERY_WHISKER_ALPHA,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=sty["marker"],
            color=sty["color"],
            markersize=ERRORBAR_MS,
            alpha=RECOVERY_MARKER_ALPHA,
            mew=0.5,
            zorder=2,
        )
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        lo -= pad
        hi += pad
        _identity_line(ax, lo, hi)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"True {labels[i]}")
        ax.set_ylabel(f"Posterior mean {labels[i]}")
        r = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.05, 0.95, rf"$r={r:.3f}$", transform=ax.transAxes, va="top", fontsize=7)

    for j in range(n_params, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)

    fig.subplots_adjust(hspace=0.38, wspace=0.42)
    _save_figure(fig, output_path)

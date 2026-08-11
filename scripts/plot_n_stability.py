#!/usr/bin/env python3
"""Plot N-stability diagnostics from evaluate_n_stability artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from asl.figures import (
    ERRORBAR_CAPSIZE,
    IDENTITY_LINE_KW,
    MPLSTYLE_PATH,
    SERIES_COLORS,
    column_figsize,
)

DDM_KEYS = ("50", "100", "300", "600_null", "1000")
DW_KEYS = ("50", "100", "300", "600", "150_null")
PANEL_WSPACE = 0.62


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--figure-path", type=Path, default=None)
    return parser.parse_args(argv)


def plot_keys(summary: dict) -> list[str]:
    ref_key = str(summary["reference_n"])
    null_key = summary["null_key"]
    template = DDM_KEYS if null_key.endswith("_null") and summary["reference_n"] == 600 else DW_KEYS
    if null_key not in template:
        template = tuple(
            key for key in summary["batches"] if key != ref_key or key == null_key
        )
    return [key for key in template if key in summary["batches"] and key != ref_key]


def x_positions(keys: list[str], null_key: str) -> np.ndarray:
    positions = []
    cursor = 0.0
    for key in keys:
        if key == null_key:
            cursor += 0.35
        positions.append(cursor)
        cursor += 1.0
    return np.asarray(positions, dtype=np.float64)


def tick_labels(keys: list[str], null_key: str) -> list[str]:
    return ["ref" if key == null_key else key for key in keys]


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=2.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _errorbar_panel(
    ax: plt.Axes,
    xs: np.ndarray,
    medians: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    *,
    marker: str,
    color: str,
    null_index: int | None = None,
    hollow_null: bool = False,
) -> None:
    style_axes(ax)
    yerr = np.vstack([medians - lows, highs - medians])
    ax.plot(xs, medians, "-", color=color, linewidth=0.9, zorder=1)
    for i in range(len(xs)):
        ax.errorbar(
            [xs[i]],
            [medians[i]],
            yerr=yerr[:, i : i + 1],
            fmt="none",
            color=color,
            capsize=ERRORBAR_CAPSIZE,
            elinewidth=0.9,
            zorder=2,
        )
        if hollow_null and null_index is not None and i == null_index:
            ax.scatter(
                [xs[i]],
                [medians[i]],
                s=22,
                facecolors="none",
                edgecolors=SERIES_COLORS[2],
                linewidths=0.9,
                zorder=5,
            )
        else:
            ax.plot(
                xs[i],
                medians[i],
                marker=marker,
                linestyle="none",
                color=color,
                markersize=3.5,
                mew=0.6,
                zorder=3,
            )


def panel_diagonal_ratios(
    ax: plt.Axes, summary: dict, keys: list[str], xs: np.ndarray
) -> None:
    medians, lows, highs = [], [], []
    for key in keys:
        diag_blocks = summary["batches"][key]["covariance_vs_ref"]["diag_ratios"]
        medians.append(float(np.mean([block["median"] for block in diag_blocks])))
        lows.append(float(np.mean([block["p05"] for block in diag_blocks])))
        highs.append(float(np.mean([block["p95"] for block in diag_blocks])))
    medians_arr = np.asarray(medians)
    _errorbar_panel(
        ax,
        xs,
        medians_arr,
        np.asarray(lows),
        np.asarray(highs),
        marker="o",
        color=SERIES_COLORS[0],
    )
    ax.axhline(1.0, **IDENTITY_LINE_KW)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels(keys, summary["null_key"]))
    ax.set_ylabel(r"$C_1$ ratio")
    ax.set_title("Variance ratios")


def panel_stein(
    ax: plt.Axes, summary: dict, keys: list[str], xs: np.ndarray
) -> None:
    null_key = summary["null_key"]
    medians, lows, highs = [], [], []
    for key in keys:
        block = summary["batches"][key]["covariance_vs_ref"]["stein"]
        medians.append(block["median"])
        lows.append(block["p05"])
        highs.append(block["p95"])
    medians_arr = np.asarray(medians)
    null_index = keys.index(null_key) if null_key in keys else None
    _errorbar_panel(
        ax,
        xs,
        medians_arr,
        np.asarray(lows),
        np.asarray(highs),
        marker="s",
        color=SERIES_COLORS[1],
        null_index=null_index,
        hollow_null=True,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels(keys, null_key))
    ax.set_ylabel("Stein discrepancy")
    ax.set_title("Matrix discrepancy")


def plot_n_stability(summary_path: Path, figure_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    keys = plot_keys(summary)
    if not keys:
        raise ValueError(f"No batch keys found in {summary_path}")

    xs = x_positions(keys, summary["null_key"])
    plt.style.use(MPLSTYLE_PATH)
    fig, axes = plt.subplots(
        1, 2, figsize=column_figsize(2, 1, aspect=1.0), sharex=True
    )
    panel_diagonal_ratios(axes[0], summary, keys, xs)
    panel_stein(axes[1], summary, keys, xs)
    for ax in axes:
        ax.set_box_aspect(1)
    fig.subplots_adjust(wspace=PANEL_WSPACE)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = args.results_dir or (repo_root / "results" / args.slug)
    summary_path = results_dir / "n_stability_summary.json"
    figure_path = args.figure_path or (
        repo_root / "figures" / args.slug / "n_stability.pdf"
    )
    if not summary_path.exists():
        print(f"[plot_n_stability] FAIL: missing {summary_path}", file=sys.stderr)
        sys.exit(1)
    plot_n_stability(summary_path, figure_path)
    print(f"[plot_n_stability] Wrote {figure_path}")


if __name__ == "__main__":
    main()

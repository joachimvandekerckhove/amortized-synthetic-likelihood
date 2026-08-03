"""Figure helpers for VPW08 paper illustrations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paths import (
    COLLAPSE_JSON,
    DDM3_JSON,
    DDM4_JSON,
    EZ_JSON,
    FIG_DELTA_KAPPA,
    FIG_DDM4_STAR,
    FIG_EZ_STAR,
    FIGURES_DIR,
)

COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")

FAMILY_STYLE = {
    "mu": {"color": COLORS[2], "marker": MARKERS[2]},
    "lambda": {"color": COLORS[0], "marker": MARKERS[0]},
    "gamma": {"color": COLORS[1], "marker": MARKERS[1]},
}


def _block(fit: dict, key: str) -> dict:
    if key.startswith("gamma_"):
        return fit["gamma"][key]
    return fit[key]


def _posterior_interval(block: dict) -> tuple[float, float, float]:
    mean = float(block["mean"])
    lo = block.get("lo95", block.get("2.5%"))
    hi = block.get("hi95", block.get("97.5%"))
    if lo is None or hi is None:
        sd = float(block.get("sd", 0.0))
        lo = mean - 2.0 * sd
        hi = mean + 2.0 * sd
    return mean, float(lo), float(hi)


def _precision_to_sd(mean: float, lo: float, hi: float) -> tuple[float, float, float]:
    if min(mean, lo, hi) <= 0:
        raise ValueError(f"precision summaries must be positive; got {mean=}, {lo=}, {hi=}")
    return mean**-0.5, hi**-0.5, lo**-0.5


def plot_star_comparison(
    rows: list[tuple[float, float, float, float, float, float, str]],
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    values = [v for row in rows for v in row[:6]]
    lo, hi = min(values), max(values)
    pad = 0.06 * (hi - lo) if hi > lo else 0.1
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.8, zorder=1)
    for x, xlo, xhi, y, ylo, yhi, family in rows:
        sty = FAMILY_STYLE[family]
        ax.errorbar(
            x,
            y,
            xerr=[[x - xlo], [xhi - x]],
            yerr=[[y - ylo], [yhi - y]],
            fmt=sty["marker"],
            color=sty["color"],
            ecolor=sty["color"],
            elinewidth=0.85,
            capsize=2.0,
            capthick=0.85,
            markersize=4.5,
            mew=0.6,
            linestyle="none",
            zorder=2,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_STYLE[fam]["marker"],
            color=FAMILY_STYLE[fam]["color"],
            markerfacecolor=FAMILY_STYLE[fam]["color"],
            markeredgecolor=FAMILY_STYLE[fam]["color"],
            markeredgewidth=0.6,
            linestyle="none",
            markersize=4.5,
        )
        for fam in ("mu", "lambda", "gamma")
    ]
    ax.legend(
        legend_handles,
        [r"$\mu$", r"$\lambda$", r"$\gamma_j$"],
        loc="upper left",
        frameon=False,
        handlelength=1.2,
        fontsize=8,
    )
    fig.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


STAR_PARAMS: list[tuple[str, str]] = [
    ("drift_mu", "mu"),
    ("drift_lambda", "lambda"),
    ("gamma_1", "gamma"),
    ("gamma_2", "gamma"),
    ("gamma_3", "gamma"),
    ("gamma_4", "gamma"),
]


def _star_rows(left: dict, right: dict) -> list[tuple[float, float, float, float, float, float, str]]:
    rows = []
    for key, family in STAR_PARAMS:
        lx, lxlo, lxhi = _posterior_interval(_block(left, key))
        ry, rylo, ryhi = _posterior_interval(_block(right, key))
        if key == "drift_lambda":
            lx, lxlo, lxhi = _precision_to_sd(lx, lxlo, lxhi)
            ry, rylo, ryhi = _precision_to_sd(ry, rylo, ryhi)
        rows.append((lx, lxlo, lxhi, ry, rylo, ryhi, family))
    return rows


def plot_ez_vs_ddm3() -> Path:
    if not EZ_JSON.exists() or not DDM3_JSON.exists():
        raise FileNotFoundError(f"Missing {EZ_JSON} or {DDM3_JSON}")
    ez = json.loads(EZ_JSON.read_text())
    ddm3 = json.loads(DDM3_JSON.read_text())
    rows = _star_rows(ez, ddm3)
    plot_star_comparison(rows, "EZ posterior mean", "ASL posterior mean", FIG_EZ_STAR)
    return FIG_EZ_STAR


def plot_ddm3_vs_ddm4() -> Path:
    if not DDM3_JSON.exists() or not DDM4_JSON.exists():
        raise FileNotFoundError(f"Missing {DDM3_JSON} or {DDM4_JSON}")
    ddm3 = json.loads(DDM3_JSON.read_text())
    ddm4 = json.loads(DDM4_JSON.read_text())
    rows = _star_rows(ddm3, ddm4)
    plot_star_comparison(rows, "DDM3 posterior mean", "DDM4 posterior mean", FIG_DDM4_STAR)
    return FIG_DDM4_STAR


def plot_delta_kappa() -> Path:
    if not COLLAPSE_JSON.exists():
        raise FileNotFoundError(f"Missing {COLLAPSE_JSON}")
    fit = json.loads(COLLAPSE_JSON.read_text())
    posterior = fit["delta_kappa"]
    mean = float(posterior["mean"])
    lo = float(posterior["lo95"])
    hi = float(posterior["hi95"])
    sty = FAMILY_STYLE["lambda"]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    ax.errorbar(
        mean,
        0,
        xerr=[[mean - lo], [hi - mean]],
        fmt=sty["marker"],
        color=sty["color"],
        ecolor=sty["color"],
        capsize=2.5,
        elinewidth=1.0,
        markersize=5,
        linestyle="none",
    )
    ax.axvline(0.0, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel(r"$\Delta\kappa$")
    ax.set_yticks([])
    span = hi - lo
    pad = max(0.15 * span, 0.05)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.5, 0.5)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DELTA_KAPPA, bbox_inches="tight")
    plt.close(fig)
    return FIG_DELTA_KAPPA


def plot_all() -> list[Path]:
    paths = [plot_ez_vs_ddm3(), plot_ddm3_vs_ddm4(), plot_delta_kappa()]
    for path in paths:
        print(f"[vpw08:figures] Wrote {path}")
    return paths

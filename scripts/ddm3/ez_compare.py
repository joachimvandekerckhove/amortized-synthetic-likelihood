#!/usr/bin/env python3
"""Benchmark ASL vs Bayesian EZ on shared ddm3 simulations; write figure PDF."""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox, offset_copy

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asl.data import load_target_transform
from asl.figures import recovery_param_style
from asl.onnxruntime_sdk import ensure_onnxruntime_lib_on_path
from asl.recovery import (
    N_CHAINS,
    iqr_interval,
    resolve_recovery_settings,
    resolve_recovery_workers,
    simulate_subject_observations,
)
from models.catalog import get_model

SLUG = "ddm3"
PARAM_NAMES = ("v", "a", "t0")
PARAM_ORDER = ("v", "a", "t0")
PARAM_LABELS = {
    "v": r"Drift $v$",
    "t0": r"Non-decision $t_0$",
    "a": r"Boundary $a$",
}
COL_LABELS = ("Posterior mean", "Posterior SD")
RHAT_THRESHOLD = 1.1
THIN = 2

# JAGS/BUGS dnorm(mean, precision). Archive EZ uses precision throughout:
#   v ~ dnorm(0, 0.25)           -> tau_v = 0.25, sigma_v = 2
#   M_ob ~ dnorm(M_pr, N / V_pr) -> tau = N / V_pr (precision of mean RT)
ARCHIVE_EZ_V_TAU = 0.25
ARCHIVE_EZ_V_SD = 1.0 / np.sqrt(ARCHIVE_EZ_V_TAU)
ARCHIVE_EZ_V_SIGMA_MULT = 4.0  # uniform envelope for comparing to ASL dunif

# Archive ASL priors (ddm3mv): dunif on DDM3_PRIOR_BOUNDS.
ARCHIVE_ASL_PRIOR = {
    "v": (-3.0, 3.0),
    "a": (0.5, 3.0),
    "t0": (0.15, 0.45),
}
# Archive EZ priors: v normal (mapped to dunif envelope); a,t0 dunif below.
ARCHIVE_EZ_PRIOR = {
    "v": (
        -ARCHIVE_EZ_V_SIGMA_MULT * ARCHIVE_EZ_V_SD,
        ARCHIVE_EZ_V_SIGMA_MULT * ARCHIVE_EZ_V_SD,
    ),
    "a": (0.3, 2.5),
    "t0": (0.1, 0.5),
}

JSON_PATH = ROOT / "results" / SLUG / "ez_compare.json"
PDF_PATH = ROOT / "figures" / SLUG / "ez_compare.pdf"

# Narrower archive envelope per parameter; uniform priors for ASL, EZ, and simulation.
EZ_COMPARE_PRIOR_BOUNDS: tuple[tuple[float, float], ...] = tuple(
    (
        max(ARCHIVE_ASL_PRIOR[name][0], ARCHIVE_EZ_PRIOR[name][0]),
        min(ARCHIVE_ASL_PRIOR[name][1], ARCHIVE_EZ_PRIOR[name][1]),
    )
    for name in PARAM_NAMES
)

# Plot styling (matches project-archive paper_figures/ez_compare.py).
COLUMN_WIDTH_IN = 3.45
SCATTER_SIZE = 10
ACCENT_COLOR = "#c0392b"
IDENTITY_LINE_KW = {"color": "#222222", "linewidth": 0.7, "alpha": 0.55}
INSET_TITLE_SIZE = 6
INSET_STAT_SIZE = 5.5
INSET_TITLE_CONTENT_GAP_PT = 2


def uniform_prior_jags(bounds: tuple[tuple[float, float], ...]) -> str:
    return "\n    ".join(
        f"{name} ~ dunif({lo}, {hi})" for name, (lo, hi) in zip(PARAM_NAMES, bounds)
    )


def build_asl_model_string(model, obs: dict) -> str:
    priors = uniform_prior_jags(EZ_COMPARE_PRIOR_BOUNDS)
    lines = ["model {", f"    {priors}"]
    lines.extend(f"    {line}" for line in model.build_jags_likelihood(obs))
    lines.append("}")
    return "\n".join(lines)


def build_ez_model_string() -> str:
    priors = uniform_prior_jags(EZ_COMPARE_PRIOR_BOUNDS)
    return f"""model {{
    {priors}

    q <- exp(-a * v)
    R_pr <- 1 / (q + 1)
    M_pr <- t0 + (a / (2 * v)) * (1 - q) / (1 + q)
    V_pr <- (a / (2 * pow(v, 3))) * (1 - 2 * a * v * q - q * q) / pow(q + 1, 2)

    T_ob ~ dbin(R_pr, N)
    M_ob ~ dnorm(M_pr, N / V_pr)
    V_ob ~ dgamma((N - 1) / 2, (N - 1) / (2 * V_pr))
}}"""


def compute_compare_inits(rng_seed: int) -> list[dict]:
    rng = np.random.default_rng(rng_seed)
    inits = []
    for _ in range(N_CHAINS):
        inits.append(
            {
                name: float(rng.uniform(*iqr_interval(lo, hi)))
                for name, (lo, hi) in zip(PARAM_NAMES, EZ_COMPARE_PRIOR_BOUNDS)
            }
        )
    return inits


def posterior_stats(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "sd": float(np.std(samples, ddof=1)),
        "pct2_5": float(np.percentile(samples, 2.5)),
        "pct97_5": float(np.percentile(samples, 97.5)),
    }


def method_converged(method_result: dict | None) -> bool:
    if method_result is None:
        return False
    return all(method_result[p]["rhat"] <= RHAT_THRESHOLD for p in PARAM_NAMES)


def fit_asl(model, obs: dict, settings: dict, subj_seed: int) -> dict | None:
    from py2jags import run_jags

    model_string = build_asl_model_string(model, obs)
    obs_raw = np.asarray(obs["obs"], dtype=np.float64)
    data = {"obs": obs_raw.tolist(), "n_trials": settings["n_trials"]}
    inits = compute_compare_inits(subj_seed)

    try:
        result = run_jags(
            model_string=model_string,
            data_dict=data,
            monitorparams=list(PARAM_NAMES),
            nchains=settings["n_chains"],
            nsamples=settings["n_iter"],
            nburnin=settings["n_burnin"],
            thin=THIN,
            init=inits,
            modules=[f"{model.slug}_emulator"],
            parallel=True,
            maxcores=settings["n_chains"],
        )
    except Exception:
        return None

    out: dict[str, dict[str, float]] = {}
    for name in PARAM_NAMES:
        samples = result.get_samples(name)
        stats = posterior_stats(samples)
        stats["rhat"] = float(result.rhat(name))
        out[name] = stats
    return out


def fit_ez(
    acc: float,
    rt_mean: float,
    rt_var: float,
    settings: dict,
    subj_seed: int,
) -> dict | None:
    from py2jags import run_jags

    n_trials = settings["n_trials"]
    t_ob = int(round(acc * n_trials))
    t_ob = max(0, min(n_trials, t_ob))
    v_ob = max(float(rt_var), 1e-12)
    data = {
        "T_ob": t_ob,
        "M_ob": float(rt_mean),
        "V_ob": v_ob,
        "N": n_trials,
    }
    inits = compute_compare_inits(subj_seed)

    try:
        result = run_jags(
            model_string=build_ez_model_string(),
            data_dict=data,
            monitorparams=list(PARAM_NAMES),
            nchains=settings["n_chains"],
            nsamples=settings["n_iter"],
            nburnin=settings["n_burnin"],
            thin=THIN,
            init=inits,
            parallel=True,
            maxcores=settings["n_chains"],
        )
    except Exception:
        return None

    out: dict[str, dict[str, float]] = {}
    for name in PARAM_NAMES:
        samples = result.get_samples(name)
        stats = posterior_stats(samples)
        stats["rhat"] = float(result.rhat(name))
        out[name] = stats
    return out


def compare_one_subject(args: tuple) -> dict:
    subj_idx, true_params, subj_seed, settings = args
    model = get_model(SLUG)
    load_target_transform(SLUG)

    obs = simulate_subject_observations(
        model, true_params, settings["n_trials"], subj_seed
    )
    if not obs["valid"]:
        return {
            "subj": subj_idx,
            "valid": False,
            "both_converged": False,
        }

    acc, rt_mean, rt_var = map(float, obs["obs"])
    asl = fit_asl(model, obs, settings, subj_seed)
    ez = fit_ez(acc, rt_mean, rt_var, settings, subj_seed)
    both = method_converged(asl) and method_converged(ez)

    return {
        "subj": subj_idx,
        "valid": True,
        "true": {n: float(true_params[i]) for i, n in enumerate(PARAM_NAMES)},
        "summaries": {"acc": acc, "rt_mean": rt_mean, "rt_var": rt_var},
        "asl": asl,
        "ez": ez,
        "asl_converged": method_converged(asl),
        "ez_converged": method_converged(ez),
        "both_converged": both,
    }


def pack_method_results(
    rows: list[dict], method_key: str
) -> dict[str, dict[str, list[float]]]:
    packed: dict[str, dict[str, list[float]]] = {
        stat: {p: [] for p in PARAM_NAMES}
        for stat in ("mean", "sd", "pct2_5", "pct97_5", "rhat")
    }
    for row in rows:
        method = row[method_key]
        for p in PARAM_NAMES:
            for stat in packed:
                packed[stat][p].append(float(method[p][stat]))
    return packed


def aggregate_metrics(rows: list[dict]) -> dict:
    correlations = {"asl": {}, "ez": {}}
    coverage = {"asl": {}, "ez": {}}
    sd_ratio_ez_over_asl = {}
    for method_key in ("asl", "ez"):
        for p in PARAM_NAMES:
            t = np.array([row["true"][p] for row in rows])
            m = np.array([row[method_key][p]["mean"] for row in rows])
            lo = np.array([row[method_key][p]["pct2_5"] for row in rows])
            hi = np.array([row[method_key][p]["pct97_5"] for row in rows])
            correlations[method_key][p] = float(np.corrcoef(t, m)[0, 1])
            coverage[method_key][p] = float(np.mean((t >= lo) & (t <= hi)))

    for p in PARAM_NAMES:
        ez_sd = np.array([row["ez"][p]["sd"] for row in rows])
        asl_sd = np.array([row["asl"][p]["sd"] for row in rows])
        sd_ratio_ez_over_asl[p] = float(np.median(ez_sd / asl_sd))

    return {
        "correlations": correlations,
        "coverage_95": coverage,
        "median_sd_ratio_ez_over_asl": sd_ratio_ez_over_asl,
    }


def run_comparison(settings: dict) -> dict:
    ensure_onnxruntime_lib_on_path()
    model = get_model(SLUG)
    onnx_path = ROOT / "results" / SLUG / "model.onnx"
    transform_path = ROOT / "results" / SLUG / "target_transform.pkl"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing emulator: {onnx_path}")
    if not transform_path.exists():
        raise FileNotFoundError(f"Missing transform: {transform_path}")

    load_target_transform(SLUG)
    max_workers = resolve_recovery_workers(settings["n_chains"])
    print(
        f"[ez_compare] START n_subjects={settings['n_subjects']} "
        f"trials={settings['n_trials']} iter={settings['n_iter']} "
        f"burnin={settings['n_burnin']} workers={max_workers} "
        f"prior_bounds={EZ_COMPARE_PRIOR_BOUNDS}",
        flush=True,
    )

    rng = np.random.default_rng(42)
    work_items = []
    for subj in range(settings["n_subjects"]):
        true_params = np.empty(len(PARAM_NAMES))
        for i, (lo, hi) in enumerate(EZ_COMPARE_PRIOR_BOUNDS):
            true_params[i] = rng.uniform(lo, hi)
        subj_seed = 1000 + subj
        work_items.append((subj, true_params, subj_seed, settings))

    results: list[dict] = []
    t0 = time.monotonic()
    report_every = max(1, settings["n_subjects"] // 20)
    with Pool(processes=max_workers) as pool:
        for i, row in enumerate(
            pool.imap_unordered(compare_one_subject, work_items, chunksize=1)
        ):
            results.append(row)
            if (i + 1) % report_every == 0 or (i + 1) == settings["n_subjects"]:
                both = sum(r.get("both_converged", False) for r in results)
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                remaining = settings["n_subjects"] - (i + 1)
                eta = remaining / rate if rate > 0 else float("nan")
                print(
                    f"[ez_compare] {i + 1}/{settings['n_subjects']} done, "
                    f"both_converged={both}, elapsed={elapsed:.0f}s, "
                    f"rate={rate:.2f} subj/s, ETA ~{eta / 60:.1f}m",
                    flush=True,
                )

    results.sort(key=lambda r: r["subj"])
    valid_rows = [r for r in results if r.get("valid", False)]
    converged_rows = [r for r in valid_rows if r["both_converged"]]
    if len(converged_rows) < 3:
        raise RuntimeError(
            f"Too few subjects with both methods converged "
            f"({len(converged_rows)}/{settings['n_subjects']})"
        )
    agg = aggregate_metrics(converged_rows)

    report = {
        "n_subjects": settings["n_subjects"],
        "n_valid": len(valid_rows),
        "n_converged_both": len(converged_rows),
        "n_dropped_rhat": len(valid_rows) - len(converged_rows),
        "prior_bounds": {
            name: [lo, hi] for name, (lo, hi) in zip(PARAM_NAMES, EZ_COMPARE_PRIOR_BOUNDS)
        },
        "settings": settings,
        "asl": pack_method_results(converged_rows, "asl"),
        "ez": pack_method_results(converged_rows, "ez"),
        "correlations": agg["correlations"],
        "coverage_95": agg["coverage_95"],
        "median_sd_ratio_ez_over_asl": agg["median_sd_ratio_ez_over_asl"],
        "subj_index": [row["subj"] for row in converged_rows],
        "true": {p: [row["true"][p] for row in converged_rows] for p in PARAM_NAMES},
    }
    elapsed_total = time.monotonic() - t0
    print(
        f"[ez_compare] FINISH elapsed={elapsed_total:.0f}s "
        f"converged_both={report['n_converged_both']}/{settings['n_subjects']}",
        flush=True,
    )
    return report


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=2.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _axis_limits(*arrays: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if len(a)])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = 0.06 * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


def _identity_line(ax: plt.Axes, lo: float, hi: float) -> None:
    ax.plot([lo, hi], [lo, hi], **IDENTITY_LINE_KW)


def _inset_top_left(
    ax: plt.Axes, title: str, stat_text: str, highlight: bool = False
) -> None:
    x = 0.04
    y_top = 0.97
    ax.text(
        x,
        y_top,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=INSET_TITLE_SIZE,
        fontweight="bold",
    )
    below_title = offset_copy(
        ax.transAxes,
        fig=ax.figure,
        x=0,
        y=-(INSET_TITLE_SIZE + INSET_TITLE_CONTENT_GAP_PT),
        units="points",
    )
    stat_props: dict = {"ha": "left", "va": "top", "fontsize": INSET_STAT_SIZE}
    if highlight:
        stat_props.update({"fontweight": "bold", "color": ACCENT_COLOR})
    ax.text(x, y_top, stat_text, transform=below_title, **stat_props)


def _inset_bottom_right(ax: plt.Axes, title: str, stat_text: str) -> None:
    x = 0.96
    y_bottom = 0.04
    ax.text(
        x,
        y_bottom,
        stat_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=INSET_STAT_SIZE,
    )
    above_stat = offset_copy(
        ax.transAxes,
        fig=ax.figure,
        x=0,
        y=INSET_STAT_SIZE + INSET_TITLE_CONTENT_GAP_PT,
        units="points",
    )
    ax.text(
        x,
        y_bottom,
        title,
        transform=above_stat,
        ha="right",
        va="bottom",
        fontsize=INSET_TITLE_SIZE,
        fontweight="bold",
    )


def _add_row_label(fig: plt.Figure, axes: np.ndarray, row: int, label: str) -> None:
    bbox_left = axes[row, 0].get_position()
    bbox_right = axes[row, 1].get_position()
    x_center = (bbox_left.x0 + bbox_right.x1) / 2
    y_top = bbox_left.y1 + 0.012
    fig.text(x_center, y_top, label, ha="center", va="bottom", fontsize=8)


def plot_figure(data: dict, output_path: Path) -> None:
    _apply_paper_style()
    fig, axes = plt.subplots(
        3, 2, figsize=(COLUMN_WIDTH_IN + 0.1, 5.45), sharex=False, sharey=False
    )
    asl = data["asl"]
    ez = data["ez"]
    corr = data["correlations"]
    cov = data["coverage_95"]

    for row, pname in enumerate(PARAM_ORDER):
        sty = recovery_param_style(pname)
        asl_mean = np.asarray(asl["mean"][pname])
        ez_mean = np.asarray(ez["mean"][pname])
        asl_sd = np.asarray(asl["sd"][pname])
        ez_sd = np.asarray(ez["sd"][pname])
        ax_mean = axes[row, 0]
        ax_sd = axes[row, 1]
        for ax in (ax_mean, ax_sd):
            _style_axes(ax)

        ax_mean.scatter(
            asl_mean,
            ez_mean,
            s=SCATTER_SIZE,
            alpha=0.45,
            color=sty["color"],
            rasterized=True,
        )
        lo, hi = _axis_limits(asl_mean, ez_mean)
        _identity_line(ax_mean, lo, hi)
        ax_mean.set_xlim(lo, hi)
        ax_mean.set_ylim(lo, hi)
        ax_mean.set_aspect("equal", adjustable="box")
        ax_mean.set_xlabel("ASL mean")
        ax_mean.set_ylabel("EZ mean")

        ax_sd.scatter(
            asl_sd,
            ez_sd,
            s=SCATTER_SIZE,
            alpha=0.45,
            color=sty["color"],
            rasterized=True,
        )
        lo, hi = _axis_limits(asl_sd, ez_sd)
        _identity_line(ax_sd, lo, hi)
        ax_sd.set_xlim(lo, hi)
        ax_sd.set_ylim(lo, hi)
        ax_sd.set_aspect("equal", adjustable="box")
        ax_sd.set_xlabel("ASL SD")
        ax_sd.set_ylabel("EZ SD")

        _inset_top_left(ax_mean, "EZ", f"$r$ = {corr['ez'][pname]:.4f}")
        _inset_bottom_right(ax_mean, "ASL", f"$r$ = {corr['asl'][pname]:.4f}")
        _inset_top_left(
            ax_sd,
            "EZ",
            f"Cov = {cov['ez'][pname]:.4f}",
            highlight=pname == "a",
        )
        _inset_bottom_right(ax_sd, "ASL", f"Cov = {cov['asl'][pname]:.4f}")

    fig.subplots_adjust(hspace=0.36, wspace=0.52, top=0.94, bottom=0.07)
    for row, pname in enumerate(PARAM_ORDER):
        _add_row_label(fig, axes, row, PARAM_LABELS[pname])
    for col, label in enumerate(COL_LABELS):
        bbox = axes[0, col].get_position()
        fig.text(
            bbox.x0 + bbox.width / 2,
            bbox.y1 + 0.055,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bb = Bbox.from_bounds(0, 0, fig.get_figwidth(), fig.get_figheight())
    fig.savefig(output_path, bbox_inches=bb, pad_inches=0.02)
    plt.close(fig)
    print(f"[ez_compare] Wrote {output_path}", flush=True)


def main() -> None:
    settings = resolve_recovery_settings()
    settings["n_chains"] = N_CHAINS

    report = run_comparison(settings)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ez_compare] Saved {JSON_PATH}", flush=True)
    for method in ("asl", "ez"):
        print(f"  {method.upper()} correlations: {report['correlations'][method]}")
        print(f"  {method.upper()} coverage:     {report['coverage_95'][method]}")

    plot_figure(report, PDF_PATH)


if __name__ == "__main__":
    main()

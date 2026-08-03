"""Repository paths for the VPW08 application pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/vpw08/vpw08.csv"
RESULTS_DIR = ROOT / "results/vpw08"
FIGURES_DIR = ROOT / "figures/vpw08"

EZ_JSON = RESULTS_DIR / "ez_fit.json"
DDM3_JSON = RESULTS_DIR / "ddm3_ezmatched_fit.json"
DDM4_JSON = RESULTS_DIR / "ddm4_fit.json"
COLLAPSE_JSON = RESULTS_DIR / "collapse_delta_kappa_fit.json"

FIG_EZ_STAR = FIGURES_DIR / "ez_vs_ddm3_star.pdf"
FIG_DDM4_STAR = FIGURES_DIR / "ddm3_vs_ddm4_star.pdf"
FIG_DELTA_KAPPA = FIGURES_DIR / "delta_kappa.pdf"

JNNX_SLUGS = ("ddm3", "ddm4", "ddmcollapsesig")

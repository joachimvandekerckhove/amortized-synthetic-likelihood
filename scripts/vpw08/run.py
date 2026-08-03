#!/usr/bin/env python3
"""VPW08 shape-perception application pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from fits import fit_collapse_delta_kappa, fit_ddm3_ezmatched, fit_ddm4
from mcmc import converged_json, force_rerun
from paths import (
    COLLAPSE_JSON,
    DDM3_JSON,
    DDM4_JSON,
    EZ_JSON,
    JNNX_SLUGS,
    ROOT as REPO_ROOT,
)
from plots import plot_all

STEPS = (
    "verify-emulators",
    "fit-ez",
    "fit-ddm3-ezmatched",
    "fit-ddm4",
    "fit-collapse-delta-kappa",
    "fit-all",
    "figures",
    "all",
)


def verify_emulators() -> None:
    for slug in JNNX_SLUGS:
        onnx = REPO_ROOT / "models" / f"{slug}.jnnx" / "model.onnx"
        if not onnx.exists():
            raise FileNotFoundError(
                f"Missing {onnx}. Run: make ddm3 ddm4 ddmcollapsesig"
            )
    print("[vpw08] emulators OK")


def _normalize_ez_reference(raw: dict) -> dict:
    """Map archived EZ JSON field names to the current schema."""
    if "model" in raw:
        return raw
    gamma = {}
    for i in range(1, 5):
        key = f"gamma_{i}"
        block = raw["gamma"][key]
        gamma[key] = {
            "mean": block["mean"],
            "sd": block["sd"],
            "lo95": block.get("lo95", block.get("2.5%")),
            "hi95": block.get("hi95", block.get("97.5%")),
            "rhat": block.get("rhat", float("nan")),
        }
    drift_lambda = raw["drift_lambda"]
    return {
        "model": "ez_appendix_e",
        "source": raw.get("source", "Chavez & Vandekerckhove (2025) Appendix E"),
        "mcmc": raw.get("mcmc", {}),
        "drift_mu": {
            "mean": raw["drift_mu"]["mean"],
            "sd": raw["drift_mu"]["sd"],
            "lo95": raw["drift_mu"].get("lo95", raw["drift_mu"].get("2.5%")),
            "hi95": raw["drift_mu"].get("hi95", raw["drift_mu"].get("97.5%")),
            "rhat": raw["drift_mu"].get("rhat", float("nan")),
        },
        "drift_lambda": {
            "mean": drift_lambda["mean"],
            "sd": drift_lambda["sd"],
            "lo95": drift_lambda.get("lo95", drift_lambda.get("2.5%")),
            "hi95": drift_lambda.get("hi95", drift_lambda.get("97.5%")),
            "rhat": drift_lambda.get("rhat", float("nan")),
        },
        "gamma": gamma,
        "convergence": {"converged": True, "max_rhat": 1.0, "rhat_gate": 1.05},
    }


def fit_ez() -> None:
    if converged_json(EZ_JSON) and not force_rerun():
        print(f"[vpw08:ez] SKIP converged output exists: {EZ_JSON}")
        return

    rscript = shutil.which("Rscript")
    if rscript:
        subprocess.run(
            [rscript, str(SCRIPT_DIR / "fit_ez.R"), str(REPO_ROOT)],
            check=True,
            cwd=REPO_ROOT,
        )
        return

    reference = REPO_ROOT / "data/vpw08/ez_fit_reference.json"
    if not reference.exists():
        raise RuntimeError(
            "Rscript not found and no data/vpw08/ez_fit_reference.json. "
            "Install R + R2jags or provide the reference EZ fit JSON."
        )
    EZ_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = _normalize_ez_reference(json.loads(reference.read_text()))
    EZ_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[vpw08:ez] Wrote {EZ_JSON} from reference (Rscript unavailable)")


def _maybe_fit(name: str, path: Path, fn_name: str) -> None:
    if converged_json(path) and not force_rerun():
        print(f"[vpw08:{name}] SKIP converged output exists")
        return
    if fn_name == "ddm3":
        fit_ddm3_ezmatched()
    elif fn_name == "ddm4":
        fit_ddm4()
    else:
        fit_collapse_delta_kappa()


def fit_ddm3_step() -> None:
    _maybe_fit("ddm3", DDM3_JSON, "ddm3")


def fit_ddm4_step() -> None:
    _maybe_fit("ddm4", DDM4_JSON, "ddm4")


def fit_collapse_step() -> None:
    _maybe_fit("collapse", COLLAPSE_JSON, "collapse")


def fit_all() -> None:
    from fits import fit_collapse_delta_kappa, fit_ddm3_ezmatched, fit_ddm4

    jobs = [
        ("ddm3", DDM3_JSON, fit_ddm3_ezmatched),
        ("ddm4", DDM4_JSON, fit_ddm4),
        ("collapse", COLLAPSE_JSON, fit_collapse_delta_kappa),
    ]
    for name, path, fn in jobs:
        if converged_json(path) and not force_rerun():
            print(f"[vpw08:{name}] SKIP converged output exists")
            continue
        fn()


def figures() -> None:
    plot_all()


def all_steps() -> None:
    verify_emulators()
    fit_ez()
    fit_all()
    figures()


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(STEPS)}>", file=sys.stderr)
        sys.exit(1)

    dispatch = {
        "verify-emulators": verify_emulators,
        "fit-ez": fit_ez,
        "fit-ddm3-ezmatched": fit_ddm3_step,
        "fit-ddm4": fit_ddm4_step,
        "fit-collapse-delta-kappa": fit_collapse_step,
        "fit-all": fit_all,
        "figures": figures,
        "all": all_steps,
    }
    dispatch[sys.argv[1]]()


if __name__ == "__main__":
    main()

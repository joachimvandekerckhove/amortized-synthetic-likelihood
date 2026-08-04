from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

VPW08_DIR = Path(__file__).resolve().parents[1]
ROOT = VPW08_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VPW08_DIR))

from data import build_cells, load_vpw08  # noqa: E402
from jags_models import (  # noqa: E402
    build_collapse_delta_kappa_model,
    build_ddm3_ezmatched_model,
    build_ddm4_model,
)


def test_load_vpw08_rows():
    df = load_vpw08()
    assert len(df) > 0
    assert set(df.columns) >= {"sub", "cond", "response", "rt"}


@pytest.mark.parametrize("summary", ["ez", "ddm4", "collapse"])
def test_build_cells_shape(summary: str):
    cells = build_cells(summary=summary)
    assert cells["K"] == 45
    assert cells["N_SUBJ"] == 9
    assert cells["obs_raw"].shape == (45, cells["obs_raw"].shape[1])
    assert np.all(np.isfinite(cells["obs_raw"]))


def test_ez_summary_dim():
    cells = build_cells(summary="ez")
    assert cells["obs_raw"].shape[1] == 3


def test_ddm4_summary_dim():
    cells = build_cells(summary="ddm4")
    assert cells["obs_raw"].shape[1] == 5


def test_collapse_summary_dim():
    cells = build_cells(summary="collapse")
    assert cells["obs_raw"].shape[1] == 4


def test_jags_models_use_sl_and_raw_obs():
    for builder, slug in (
        (build_ddm3_ezmatched_model, "ddm3"),
        (build_ddm4_model, "ddm4"),
        (build_collapse_delta_kappa_model, "ddmcollapsesig"),
    ):
        model = builder(include_ppc=True)
        assert f"{slug}_sl" in model
        assert "obs[cell," in model
        assert "obs_std" not in model
        assert "obs_rep[cell," in model

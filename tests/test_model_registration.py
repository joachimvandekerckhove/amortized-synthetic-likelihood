"""Unit tests for model registration contract."""

from __future__ import annotations

import pytest

from asl.data import TargetTransform
from asl.spec import Model
from models.catalog import get_model


def test_all_catalog_models_have_valid_registration():
    for slug in ("ddm3", "ddm4", "ddmcollapsesig", "dw"):
        model = get_model(slug)
        assert len(model.prior_bounds) == model.n_params
        assert len(model.summary_transforms) == model.n_summaries


def test_invalid_summary_transform_rejected():
    with pytest.raises(ValueError, match="summary_transforms length"):
        Model(
            slug="bad",
            param_names=("v",),
            param_bounds=((-1.0, 1.0),),
            prior_bounds=((-0.5, 0.5),),
            summary_names=("acc", "rt"),
            summary_transforms=("identity",),
            simulate_summaries=lambda p, n, s: None,
        )


def test_target_transform_uses_registered_transforms(toy_model):
    tt = TargetTransform.from_model(toy_model)
    assert tt.n_rt_columns == 2
    assert toy_model.summary_transforms == ("identity", "log1p", "log1p")

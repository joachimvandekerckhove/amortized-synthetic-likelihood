"""Unit tests for asl.data."""

from __future__ import annotations

import numpy as np
import pytest

from asl.data import (
    TargetTransform,
    count_rt_columns,
    summary_column_masks,
)


class TestSummaryColumnMasks:
    def test_ddm3_layout(self, toy_model):
        rt_mask, prop_mask = summary_column_masks(toy_model)
        assert rt_mask.tolist() == [False, True, True]
        assert prop_mask.tolist() == [True, False, False]

    def test_count_rt_columns(self, toy_model):
        assert count_rt_columns(toy_model) == 2


class TestTargetTransform:
    @pytest.fixture
    def transform(self, toy_model):
        rt_mask, _ = summary_column_masks(toy_model)
        return TargetTransform(rt_mask)

    def test_from_model(self, toy_model):
        tt = TargetTransform.from_model(toy_model)
        assert tt.n_rt_columns == 2

    def test_fit_transform_and_inverse(self, transform):
        y = np.array([[0.6, 0.5, 0.02], [0.4, 0.7, 0.03]], dtype=np.float64)
        y_t = transform.fit_transform(y)
        assert y_t.shape == y.shape
        y_back = transform.inverse_transform(y_t)
        np.testing.assert_allclose(y_back, y, rtol=1e-5)

    def test_transform_after_fit(self, transform):
        y_train = np.array([[0.6, 0.5, 0.02]], dtype=np.float64)
        transform.fit_transform(y_train)
        y_new = np.array([[0.5, 0.6, 0.01]], dtype=np.float64)
        out = transform.transform(y_new)
        assert out.shape == (1, 3)

    def test_inverse_clips_negative_rt(self, transform):
        y = np.array([[0.5, 0.3, 0.01]], dtype=np.float64)
        transform.fit_transform(y)
        y_t = transform.transform(y)
        y_t[0, 1] -= 10.0
        y_back = transform.inverse_transform(y_t)
        assert y_back[0, 1] >= 0.0

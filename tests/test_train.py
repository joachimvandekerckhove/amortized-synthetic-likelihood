"""Unit tests for asl.train helper functions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asl.cholesky import pack_upper_tri
from asl.data import summary_column_masks
from asl.mlp import build_architecture
from asl.train import (
    DualHeadNet,
    build_C1_std_array,
    build_target_transform,
    evaluate_cov_stein,
    evaluate_mean_r2,
)


class TestDualHeadNet:
    def test_forward_shapes(self):
        base = build_architecture("DeepWide_24x4", in_dim=2, out_dim=99)()
        net = DualHeadNet(base, n_summaries=3)
        x = torch.randn(4, 2)
        mu, chol = net(x)
        assert mu.shape == (4, 3)
        assert chol.shape == (4, 6)

    def test_count_trainable_parameters(self):
        base = build_architecture("DeepWide_24x4", in_dim=2, out_dim=99)()
        net = DualHeadNet(base, n_summaries=3)
        assert net.count_trainable_parameters() > 0


class TestBuildHelpers:
    def test_build_target_transform(self, toy_model):
        rt_mask, _ = summary_column_masks(toy_model)
        z_mean = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        tt = build_target_transform(rt_mask, z_mean)
        assert tt.scaler.mean_.shape == (3,)

    def test_build_C1_std_array(self, toy_model):
        n = toy_model.n_summaries
        C1_z = np.stack(
            [pack_upper_tri(np.eye(n) * (i + 1)) for i in range(5)],
            axis=0,
        ).astype(np.float32)
        scale = np.ones(n, dtype=np.float32)
        out = build_C1_std_array(C1_z, scale, n)
        assert out.shape == (5, n, n)


class TestEvaluation:
    @pytest.fixture(autouse=True)
    def _setup_net(self, toy_model):
        self.toy_model = toy_model
        base = build_architecture("DeepWide_24x4", in_dim=2, out_dim=99)()
        self.net = DualHeadNet(base, n_summaries=3)
        self.X = torch.randn(20, 2)
        rt_mask, _ = summary_column_masks(toy_model)
        z_mean = np.abs(np.random.default_rng(1).normal(size=(20, 3))).astype(
            np.float32
        )
        z_mean[:, 0] = np.clip(z_mean[:, 0], 0.01, 0.99)
        self.tt = build_target_transform(rt_mask, z_mean)
        self.y_raw = self.tt.inverse_transform(
            self.tt.scaler.transform(
                np.column_stack(
                    [
                        z_mean[:, 0],
                        np.log1p(z_mean[:, 1]),
                        np.log1p(z_mean[:, 2]),
                    ]
                )
            )
        )

    def test_evaluate_mean_r2_returns_float(self):
        overall, per_target = evaluate_mean_r2(self.net, self.X, self.y_raw, self.tt)
        assert isinstance(overall, float)
        assert per_target.shape == (3,)

    def test_evaluate_cov_stein_returns_float(self):
        C1 = torch.eye(3).unsqueeze(0).repeat(20, 1, 1)
        val = evaluate_cov_stein(self.net, self.X, C1, 3)
        assert isinstance(val, float)

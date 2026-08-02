"""Unit tests for asl.cholesky."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from asl.cholesky import (
    build_L_and_logdet,
    build_sl_likelihood_line,
    cov_stein_loss,
    debias_emulator_error_cov,
    emulator_error_cov_path,
    emulator_output_names_for,
    load_emulator_error_cov,
    n_chol,
    pack_upper_tri,
    precision_from_chol,
    save_emulator_error_cov,
    std_cov_from_logcov,
    unpack_upper_tri,
    upper_tri_index_pairs,
)


class TestCholLayout:
    def test_n_chol(self):
        assert n_chol(3) == 6
        assert n_chol(1) == 1

    def test_upper_tri_pairs(self):
        pairs = upper_tri_index_pairs(3)
        assert pairs == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    def test_pack_unpack_roundtrip(self):
        matrix = np.array([[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 3.0]])
        flat = pack_upper_tri(matrix)
        restored = unpack_upper_tri(flat, 3)
        np.testing.assert_allclose(restored, matrix)

    def test_std_cov_from_logcov(self):
        C1 = np.eye(2)
        scale = np.array([2.0, 3.0])
        out = std_cov_from_logcov(C1, scale)
        expected = C1 / np.outer(scale, scale)
        np.testing.assert_allclose(out, expected)


class TestTorchCholOps:
    def test_build_L_positive_diagonal(self):
        n = 2
        raw = torch.tensor([[0.0, 0.1, 0.0]], dtype=torch.float32)
        L, logdet = build_L_and_logdet(raw, n)
        diag = torch.diagonal(L[0])
        assert (diag > 0).all()
        assert logdet.shape == (1,)

    def test_precision_from_chol_is_symmetric(self):
        raw = torch.tensor([[0.5, 0.1, 0.2]], dtype=torch.float32)
        P = precision_from_chol(raw, 2)
        np.testing.assert_allclose(P[0].numpy(), P[0].T.numpy(), atol=1e-5)

    def test_cov_stein_loss_scalar(self):
        n = 2
        raw = torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float32)
        C1 = torch.eye(n).unsqueeze(0)
        loss = cov_stein_loss(raw, C1, n)
        assert loss.ndim == 0
        assert float(loss) >= 0.0


class TestEmulatorErrorCov:
    def test_debias_subtracts_correction(self):
        residual = np.eye(2) * 2.0
        mean_C1 = np.eye(2)
        out = debias_emulator_error_cov(residual, mean_C1, n_rep=10, n_replicates=5)
        correction = mean_C1 / (5 * 10)
        np.testing.assert_allclose(out, residual - correction, atol=1e-9)

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sigma = np.eye(3) * 0.01
        save_emulator_error_cov("toy", sigma, n_rep=100, n_replicates=50)
        path = emulator_error_cov_path("toy")
        assert path.exists()
        loaded = load_emulator_error_cov("toy")
        np.testing.assert_allclose(loaded, sigma)
        with open(path) as f:
            payload = json.load(f)
        assert payload["n_rep"] == 100
        assert payload["R"] == 50

    def test_load_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_emulator_error_cov("missing")


class TestJagsHelpers:
    def test_build_sl_likelihood_line(self):
        lines = build_sl_likelihood_line("ddm3", ("v", "a", "t0"), 3)
        assert lines == ["obs[1:3] ~ ddm3_sl(v, a, t0, n_trials)"]

    def test_emulator_output_names(self):
        names = emulator_output_names_for(2, ("acc", "rt"))
        assert names[0] == "mu_acc"
        assert len(names) == 2 + n_chol(2)

"""Unit tests for asl.cov_data."""

from __future__ import annotations

import json

import numpy as np
import pytest

from asl.cov_data import (
    N_REP,
    N_THETA,
    R,
    SEED_DEFAULT,
    c1_column_names,
    check_summary_mi_gate,
    check_summary_variance_gate,
    cov_settings_path,
    draw_parameters,
    expected_columns,
    load_cov_settings,
    logspace_to_raw,
    resolve_cov_settings,
    save_cov_settings,
    summaries_to_logspace,
    validate_cov_training_data,
    z_mean_column_names,
)
from asl.data import summary_column_masks


class TestColumnNames:
    def test_z_mean_columns(self, toy_model):
        names = z_mean_column_names(toy_model.summary_names)
        assert names == ["z_mean_acc", "z_mean_rt_mean", "z_mean_rt_var"]

    def test_c1_columns(self):
        names = c1_column_names(3)
        assert names == ["c1_0_0", "c1_0_1", "c1_0_2", "c1_1_1", "c1_1_2", "c1_2_2"]

    def test_expected_columns(self, toy_model):
        cols = expected_columns(toy_model)
        assert cols[:2] == ["v", "a"]
        assert "z_mean_acc" in cols
        assert "c1_2_2" in cols


class TestResolveCovSettings:
    def test_defaults(self, config_file):
        config_file("")
        n_theta, n_rep, n_r, seed = resolve_cov_settings()
        assert (n_theta, n_rep, n_r, seed) == (N_THETA, N_REP, R, SEED_DEFAULT)

    def test_overrides(self, config_file):
        config_file(
            "[cov_data]\n"
            "parameter_draws = 10\n"
            "trials_per_replicate = 20\n"
            "replicates_per_parameter = 5\n"
            "random_seed = 7\n"
        )
        assert resolve_cov_settings() == (10, 20, 5, 7)


class TestDrawParameters:
    def test_within_bounds(self, toy_model):
        rng = np.random.default_rng(0)
        params = draw_parameters(toy_model, rng)
        assert params.shape == (2,)
        for val, (lo, hi) in zip(params, toy_model.param_bounds):
            assert lo <= val < hi


class TestLogspaceTransforms:
    def test_roundtrip(self, toy_model):
        rt_mask, _ = summary_column_masks(toy_model)
        raw = np.array([0.5, 0.4, 0.02])
        z = summaries_to_logspace(raw, rt_mask)
        assert z[0] == raw[0]
        assert z[1] == np.log1p(raw[1])
        restored = logspace_to_raw(z, rt_mask)
        np.testing.assert_allclose(restored, raw, rtol=1e-6)


class TestCovSettingsIO:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_cov_settings("toy", n_rep=100, n_replicates=50, seed=11)
        path = cov_settings_path("toy")
        assert path.exists()
        with open(path) as f:
            payload = json.load(f)
        assert payload == {"n_rep": 100, "R": 50, "seed": 11}
        assert load_cov_settings("toy") == (100, 50)

    def test_load_missing_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert load_cov_settings("missing") == (N_REP, R)


class TestCovTrainingQA:
    def test_summary_variance_gate_passes(self, toy_model):
        rng = np.random.default_rng(0)
        y_raw = np.column_stack(
            [
                rng.uniform(0.4, 0.6, 300),
                rng.uniform(0.3, 0.5, 300),
                rng.uniform(0.01, 0.05, 300),
            ]
        )
        check_summary_variance_gate(y_raw, toy_model)

    def test_summary_variance_gate_fails_constant(self, toy_model):
        y_raw = np.tile(np.array([0.5, 0.4, 0.02]), (50, 1))
        with pytest.raises(SystemExit):
            check_summary_variance_gate(y_raw, toy_model)

    def test_validate_cov_training_data_runs(self, toy_model, config_file):
        config_file(
            "[cov_data]\n"
            "summary_mi_permutations = 8\n"
            "summary_mi_subsample = 200\n"
        )
        rng = np.random.default_rng(1)
        n = 400
        X = np.column_stack(
            [rng.uniform(lo, hi, n) for lo, hi in toy_model.param_bounds]
        )
        y_raw = np.column_stack(
            [
                np.clip(0.5 + 0.1 * X[:, 0], 0.01, 0.99),
                0.3 + 0.05 * X[:, 1] + 0.01 * rng.standard_normal(n),
                0.01 + 0.001 * np.abs(X[:, 0]) + 0.001 * rng.standard_normal(n),
            ]
        )
        validate_cov_training_data(X, y_raw, toy_model)

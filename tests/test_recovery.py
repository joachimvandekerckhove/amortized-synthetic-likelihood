"""Unit tests for asl.recovery."""

from __future__ import annotations

import pytest

from asl.recovery import (
    COVERAGE_HI,
    COVERAGE_LO,
    N_CHAINS,
    check_coverage_gate,
    compute_chain_initial_values,
    format_recovery_progress,
    iqr_interval,
    recovery_report_interval,
    resolve_recovery_settings,
    run_recovery_study,
)


class TestFormatRecoveryProgress:
    def test_includes_counts_and_rate(self):
        line = format_recovery_progress(
            done=10, total=100, n_converged=8, n_failed=2, t0=0.0
        )
        assert "10/100" in line
        assert "8 converged" in line
        assert "2 failed" in line
        assert "subj/s" in line


class TestRecoveryReportInterval:
    def test_default_scales_with_subjects(self, config_file):
        config_file("")
        assert recovery_report_interval(500) == 10
        assert recovery_report_interval(50) == 1

    def test_config_override(self, config_file):
        config_file("[recovery]\nprogress_log_interval = 25\n")
        assert recovery_report_interval(500) == 25


class TestResolveRecoverySettings:
    def test_defaults(self, config_file):
        config_file("")
        settings = resolve_recovery_settings()
        assert settings["n_subjects"] == 500
        assert settings["n_trials"] == 500
        assert settings["n_chains"] == 4

    def test_overrides(self, config_file):
        config_file("[recovery]\nsynthetic_subjects = 12\ntrials_per_subject = 99\n")
        settings = resolve_recovery_settings()
        assert settings["n_subjects"] == 12
        assert settings["n_trials"] == 99


class TestCheckCoverageGate:
    def test_passes_in_range(self, toy_model):
        coverages = [0.95] * toy_model.n_params
        check_coverage_gate(coverages, toy_model.param_names)

    def test_fails_below_lo(self, toy_model):
        coverages = [COVERAGE_LO] * toy_model.n_params
        with pytest.raises(SystemExit) as exc:
            check_coverage_gate(coverages, toy_model.param_names)
        assert exc.value.code == 1

    def test_fails_above_hi(self, toy_model):
        coverages = [COVERAGE_HI] * toy_model.n_params
        with pytest.raises(SystemExit):
            check_coverage_gate(coverages, toy_model.param_names)


class TestChainInitialValues:
    def test_iqr_interval_for_unit_range(self):
        assert iqr_interval(0.0, 1.0) == (0.25, 0.75)

    def test_chain_inits_within_iqr(self, toy_model):
        bounds = toy_model.prior_bounds
        inits = compute_chain_initial_values(toy_model, rng_seed=42)
        assert len(inits) == N_CHAINS
        for init in inits:
            for name, (lo, hi) in zip(toy_model.param_names, bounds):
                q1, q3 = iqr_interval(lo, hi)
                assert q1 <= init[name] <= q3

    def test_chain_inits_reproducible(self, toy_model):
        a = compute_chain_initial_values(toy_model, rng_seed=7)
        b = compute_chain_initial_values(toy_model, rng_seed=7)
        assert a == b


class TestRunRecoveryStudy:
    def test_sets_onnxruntime_lib_before_pool(self, toy_model, monkeypatch):
        called = False

        def fake_ensure():
            nonlocal called
            called = True

        monkeypatch.setattr(
            "asl.recovery.ensure_onnxruntime_lib_on_path",
            fake_ensure,
        )
        monkeypatch.setattr(
            "asl.recovery.resolve_recovery_settings",
            lambda: {
                "n_subjects": 0,
                "n_trials": 10,
                "n_iter": 10,
                "n_burnin": 5,
                "n_chains": 1,
            },
        )

        with pytest.raises(SystemExit) as exc:
            run_recovery_study(toy_model)
        assert called
        assert exc.value.code == 1

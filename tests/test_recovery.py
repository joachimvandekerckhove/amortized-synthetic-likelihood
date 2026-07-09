"""Unit tests for asl.recovery."""

from __future__ import annotations

import pytest

from asl.recovery import (
    COVERAGE_HI,
    COVERAGE_LO,
    N_SUBJECTS_FULL,
    N_SUBJECTS_SMOKE,
    check_coverage_gate,
    format_recovery_progress,
    recovery_report_interval,
    resolve_recovery_settings,
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
    def test_full_defaults(self, config_file):
        config_file("")
        settings = resolve_recovery_settings()
        assert settings["n_subjects"] == N_SUBJECTS_FULL
        assert settings["n_trials"] == 500
        assert settings["n_chains"] == 4

    def test_smoke_defaults(self, config_file):
        config_file("[run]\nsmoke = true\n")
        settings = resolve_recovery_settings()
        assert settings["n_subjects"] == N_SUBJECTS_SMOKE

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

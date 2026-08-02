"""Unit tests for asl.spec."""

from __future__ import annotations

import numpy as np

from asl.spec import Model


def _simulate(params, n_trials, seed):
    return np.ones(2)


class TestModel:
    def test_counts(self, toy_model):
        assert toy_model.n_params == 2
        assert toy_model.n_summaries == 3
        assert toy_model.n_outputs == 3

    def test_output_names_default_to_summaries(self):
        model = Model(
            slug="scalar",
            param_names=("v",),
            param_bounds=((-1.0, 1.0),),
            summary_names=("acc", "rt"),
            simulate_summaries=_simulate,
        )
        assert model.output_names == ("acc", "rt")

    def test_output_names_from_emulator(self, toy_model):
        model = Model(
            slug="emu",
            param_names=("v",),
            param_bounds=((-1.0, 1.0),),
            summary_names=("acc",),
            simulate_summaries=_simulate,
            emulator_output_names=("mu_acc", "chol_1"),
        )
        assert model.output_names == ("mu_acc", "chol_1")
        assert model.n_outputs == 2

    def test_supports_recovery_false_without_hooks(self, toy_model):
        assert not toy_model.supports_recovery()

    def test_supports_recovery(self):
        model = Model(
            slug="emu",
            param_names=("v",),
            param_bounds=((-1.0, 1.0),),
            summary_names=("acc",),
            simulate_summaries=_simulate,
            emulator_output_names=("mu_acc", "chol_1"),
            build_jags_likelihood=lambda obs: ["line"],
        )
        assert model.supports_recovery()

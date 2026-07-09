"""Unit tests for asl.mlp."""

from __future__ import annotations

import importlib

import pytest
import torch

from asl.mlp import DeepWideMLP, build_architecture, resolve_training_settings


class TestDeepWideMLP:
    def test_forward_shape(self):
        model = DeepWideMLP(width=8, depth=2, in_dim=3, out_dim=4)
        x = torch.randn(5, 3)
        out = model(x)
        assert out.shape == (5, 4)

    def test_count_trainable_parameters(self):
        model = DeepWideMLP(width=8, depth=2, in_dim=3, out_dim=4)
        count = model.count_trainable_parameters()
        assert count > 0
        assert count == sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )


class TestBuildArchitecture:
    def test_known_architecture(self):
        builder = build_architecture("DeepWide_24x4", in_dim=3, out_dim=2)
        net = builder()
        assert isinstance(net, DeepWideMLP)

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_architecture("MissingArch", in_dim=1, out_dim=1)


class TestResolveTrainingSettings:
    def test_full_defaults(self, config_file):
        config_file("")
        settings = resolve_training_settings()
        assert settings == {
            "subsample": None,
            "n_epochs": 10000,
            "batch_size": 4096,
            "lr": 1e-3,
        }

    def test_smoke_mode(self, config_file):
        config_file("[run]\nsmoke = true\n")
        settings = resolve_training_settings()
        assert settings["subsample"] == 5000
        assert settings["n_epochs"] == 300
        assert settings["batch_size"] == 512

    def test_n_epochs_override(self, config_file):
        config_file("[training]\ntraining_epochs = 42\n")
        settings = resolve_training_settings()
        assert settings["n_epochs"] == 42

    def test_smoke_with_n_epochs_override(self, config_file):
        config_file("[run]\nsmoke = true\n\n[training]\ntraining_epochs = 99\n")
        settings = resolve_training_settings()
        assert settings["n_epochs"] == 99
        assert settings["subsample"] == 5000


class TestSelectDevice:
    def test_unset_cuda_visible_devices_from_config(self, config_file, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        config_file("[device]\nunset_cuda_visible_devices = true\n")
        import asl.mlp as mlp_module

        importlib.reload(mlp_module)
        assert "CUDA_VISIBLE_DEVICES" not in mlp_module.os.environ

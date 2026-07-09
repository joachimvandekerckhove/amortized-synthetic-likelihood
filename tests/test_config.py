"""Unit tests for asl.config."""

from __future__ import annotations

from asl.config import load_config, reset_config


class TestConfig:
    def test_defaults_without_file(self, repo_root, monkeypatch):
        monkeypatch.chdir(repo_root)
        reset_config()
        config = load_config()
        assert config.smoke is False
        assert config.get("training", "training_epochs") == 10000

    def test_smoke_presets(self, config_file):
        config_file("[run]\nsmoke = true\n")
        config = load_config()
        assert config.smoke is True
        assert config.get("training", "training_epochs") == 300
        assert config.get("cov_data", "parameter_draws") == 800

    def test_explicit_override_beats_preset(self, config_file):
        config_file(
            "[run]\nsmoke = true\n\n[training]\ntraining_epochs = 42\n"
        )
        config = load_config()
        assert config.get("training", "training_epochs") == 42

    def test_has_detects_explicit_keys(self, config_file):
        config_file('[training]\narchitecture = "DeepWide_32x6"\n')
        config = load_config()
        assert config.has("training", "architecture")
        assert not config.has("training", "training_epochs")

"""Unit tests for asl.config."""

from __future__ import annotations

from asl.config import load_config, reset_config


class TestConfig:
    def test_defaults_without_file(self, repo_root, monkeypatch):
        monkeypatch.chdir(repo_root)
        reset_config()
        config = load_config()
        assert config.get("training", "training_epochs") == 10000

    def test_explicit_override(self, config_file):
        config_file("[training]\ntraining_epochs = 42\n")
        config = load_config()
        assert config.get("training", "training_epochs") == 42

    def test_missing_key_returns_default(self, config_file):
        config_file("")
        config = load_config()
        assert config.get("training", "nonexistent", 99) == 99

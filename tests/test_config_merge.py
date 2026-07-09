"""Unit tests for asl.config merging."""

from __future__ import annotations

from pathlib import Path

from asl.config import CONFIG_PATH_ENV, load_config, reset_config


class TestConfigMerge:
    def test_asl_config_merges_overrides(
        self, repo_root: Path, tmp_path, monkeypatch
    ):
        override = tmp_path / "recovery_highn.toml"
        override.write_text(
            "[recovery]\nsynthetic_subjects = 12\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(repo_root)
        monkeypatch.setenv(CONFIG_PATH_ENV, str(override))
        reset_config()

        config = load_config()
        assert config.get("recovery", "synthetic_subjects") == 12
        assert config.get("recovery", "trials_per_subject") == 500

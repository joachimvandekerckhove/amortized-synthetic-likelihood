"""Tests for JAGS module installation and active-module verification."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from asl.wire import _install_jags_module, assert_active_jags_mean_parity


def test_install_jags_module_requires_system_install(tmp_path, monkeypatch):
    """A failed system install must not silently use a process-local fallback."""
    calls: list[list[str]] = []

    class Result:
        returncode = 1
        stdout = "stdout"
        stderr = "sudo denied"

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr("asl.wire.subprocess.run", fake_run)
    monkeypatch.setenv("LTDL_LIBRARY_PATH", "preserved")
    (tmp_path / "dw_emulator.so").touch()

    with pytest.raises(RuntimeError, match="System-wide JAGS module installation failed"):
        _install_jags_module(tmp_path, {})

    assert calls == [["sudo", "make", "install"]]
    assert os.environ["LTDL_LIBRARY_PATH"] == "preserved"


def test_install_jags_module_uses_sudo_make_install(tmp_path, monkeypatch):
    """JAGS searches its system module directory, so installation must target it."""
    calls: list[tuple[list[str], Path]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, Path(kwargs["cwd"])))
        return Result()

    monkeypatch.setattr("asl.wire.subprocess.run", fake_run)

    _install_jags_module(tmp_path, {"ONNXRUNTIME_DIR": "/sdk"})

    assert calls == [(["sudo", "make", "install"], tmp_path)]


def test_active_jags_mean_parity_rejects_a_stale_module():
    """Wiring must stop before recovery when JAGS loads different ONNX weights."""
    with pytest.raises(RuntimeError, match="does not match the packaged ONNX"):
        assert_active_jags_mean_parity(
            expected=[-1.3610, -1.4487],
            observed=[-0.9109, -0.9513],
        )

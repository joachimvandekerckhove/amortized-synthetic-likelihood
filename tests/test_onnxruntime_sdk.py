"""Tests for ONNX Runtime SDK resolution."""

import os
from pathlib import Path

import pytest

from asl.onnxruntime_sdk import (
    _find_vendor_sdk,
    _sdk_is_valid,
    ensure_onnxruntime_lib_on_path,
    platform_archive_name,
    resolve_onnxruntime_sdk_dir,
)


def test_platform_archive_name_linux():
    name = platform_archive_name()
    assert name.startswith("onnxruntime-")
    assert name.endswith("-1.23.2")


def test_sdk_is_valid_requires_headers_and_lib(tmp_path: Path):
    sdk = tmp_path / "onnxruntime-linux-x64-1.23.2"
    sdk.mkdir()
    assert not _sdk_is_valid(sdk)

    (sdk / "include").mkdir()
    (sdk / "include" / "onnxruntime_cxx_api.h").write_text("stub")
    lib = sdk / "lib"
    lib.mkdir()
    (lib / "libonnxruntime.so").write_text("stub")
    assert _sdk_is_valid(sdk)


def test_find_vendor_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    sdk = vendor / "onnxruntime-linux-x64-1.23.2"
    sdk.mkdir()
    (sdk / "include").mkdir()
    (sdk / "include" / "onnxruntime_cxx_api.h").write_text("stub")
    lib = sdk / "lib"
    lib.mkdir()
    (lib / "libonnxruntime.so").write_text("stub")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "asl.toml").write_text("[wire]\n")
    found = _find_vendor_sdk(tmp_path)
    assert found == sdk


def test_resolve_prefers_asl_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sdk = tmp_path / "custom-ort"
    sdk.mkdir()
    (sdk / "include").mkdir()
    (sdk / "include" / "onnxruntime_cxx_api.h").write_text("stub")
    lib = sdk / "lib"
    lib.mkdir()
    (lib / "libonnxruntime.so").write_text("stub")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "asl.toml").write_text(
        f'[wire]\nonnxruntime_dir = "{sdk}"\n'
    )
    from asl.config import reset_config

    reset_config()
    resolved = resolve_onnxruntime_sdk_dir(tmp_path)
    assert resolved == str(sdk.resolve())
    reset_config()


def test_ensure_onnxruntime_lib_on_path_prepends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sdk = tmp_path / "custom-ort"
    sdk.mkdir()
    (sdk / "include").mkdir()
    (sdk / "include" / "onnxruntime_cxx_api.h").write_text("stub")
    lib = sdk / "lib"
    lib.mkdir()
    (lib / "libonnxruntime.so").write_text("stub")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "asl.toml").write_text(f'[wire]\nonnxruntime_dir = "{sdk}"\n')
    from asl.config import reset_config

    reset_config()
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    lib_dir = ensure_onnxruntime_lib_on_path(tmp_path)
    assert lib_dir == str(lib.resolve())
    assert os.environ["LD_LIBRARY_PATH"].startswith(str(lib.resolve()))
    reset_config()


def test_ensure_onnxruntime_lib_on_path_no_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sdk = tmp_path / "custom-ort"
    sdk.mkdir()
    (sdk / "include").mkdir()
    (sdk / "include" / "onnxruntime_cxx_api.h").write_text("stub")
    lib = sdk / "lib"
    lib.mkdir()
    (lib / "libonnxruntime.so").write_text("stub")
    lib_str = str(lib.resolve())

    monkeypatch.chdir(tmp_path)
    (tmp_path / "asl.toml").write_text(f'[wire]\nonnxruntime_dir = "{sdk}"\n')
    from asl.config import reset_config

    reset_config()
    monkeypatch.setenv("LD_LIBRARY_PATH", lib_str)
    ensure_onnxruntime_lib_on_path(tmp_path)
    assert os.environ["LD_LIBRARY_PATH"] == lib_str
    reset_config()

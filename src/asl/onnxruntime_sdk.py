"""Download and resolve the ONNX Runtime C/C++ SDK for JAGS module builds."""

from __future__ import annotations

import os
import platform
import shutil
import tarfile
import urllib.request
from pathlib import Path

from asl.config import load_config

ONNXRUNTIME_VERSION = "1.23.2"
VENDOR_DIR_NAME = "vendor"


def find_repo_root(start: Path | None = None) -> Path:
    """Return repo root (directory containing asl.toml or pyproject.toml)."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / "asl.toml").exists() or (directory / "pyproject.toml").exists():
            return directory
    return here


def platform_archive_name() -> str:
    """Return ONNX Runtime release archive base name for this machine."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"x86_64", "amd64"}:
        arch = "linux-x64"
    elif system == "darwin" and machine == "arm64":
        arch = "osx-arm64"
    elif system == "darwin" and machine == "x86_64":
        arch = "osx-x64"
    else:
        raise RuntimeError(
            f"Automatic ONNX Runtime SDK download is not supported on {system}/{machine}. "
            "Download a C/C++ SDK from https://github.com/microsoft/onnxruntime/releases "
            "and set wire.onnxruntime_dir in asl.toml."
        )
    return f"onnxruntime-{arch}-{ONNXRUNTIME_VERSION}"


def vendor_dir(repo_root: Path | None = None) -> Path:
    return find_repo_root(repo_root) / VENDOR_DIR_NAME


def _sdk_is_valid(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "include" / "onnxruntime_cxx_api.h").is_file()
        and any(path.glob("lib/libonnxruntime*"))
    )


def _find_vendor_sdk(repo_root: Path) -> Path | None:
    vendor = vendor_dir(repo_root)
    if not vendor.is_dir():
        return None
    matches = sorted(vendor.glob("onnxruntime-*"))
    for path in matches:
        if _sdk_is_valid(path):
            return path
    return None


def download_onnxruntime_sdk(repo_root: Path | None = None) -> Path:
    """Download and extract the ONNX Runtime C/C++ SDK into vendor/."""
    root = find_repo_root(repo_root)
    vendor = vendor_dir(root)
    vendor.mkdir(parents=True, exist_ok=True)

    archive_name = platform_archive_name()
    sdk_dir = vendor / archive_name
    if _sdk_is_valid(sdk_dir):
        return sdk_dir

    url = (
        f"https://github.com/microsoft/onnxruntime/releases/download/"
        f"v{ONNXRUNTIME_VERSION}/{archive_name}.tgz"
    )
    tgz_path = vendor / f"{archive_name}.tgz"
    print(f"[onnxruntime] Downloading SDK {archive_name} ...", flush=True)
    urllib.request.urlretrieve(url, tgz_path)

    try:
        with tarfile.open(tgz_path, "r:gz") as archive:
            archive.extractall(vendor, filter="data")
    finally:
        tgz_path.unlink(missing_ok=True)

    if not _sdk_is_valid(sdk_dir):
        raise RuntimeError(f"Downloaded SDK at {sdk_dir} is missing headers or lib/")
    print(f"[onnxruntime] SDK ready: {sdk_dir}", flush=True)
    return sdk_dir


def ensure_onnxruntime_sdk(repo_root: Path | None = None) -> Path:
    """Return a valid SDK path, downloading into vendor/ when needed."""
    root = find_repo_root(repo_root)
    existing = _find_vendor_sdk(root)
    if existing is not None:
        return existing
    return download_onnxruntime_sdk(root)


def resolve_onnxruntime_sdk_dir(repo_root: Path | None = None) -> str:
    """Resolve SDK directory: asl.toml, env, vendor/, or download."""
    configured = str(load_config().get("wire", "onnxruntime_dir", "") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not _sdk_is_valid(path):
            raise RuntimeError(
                f"wire.onnxruntime_dir points to invalid SDK: {path}"
            )
        return str(path.resolve())

    env_dir = os.environ.get("ONNXRUNTIME_DIR", "").strip()
    if env_dir:
        path = Path(env_dir).expanduser()
        if not _sdk_is_valid(path):
            raise RuntimeError(
                f"ONNXRUNTIME_DIR points to invalid SDK: {path}"
            )
        return str(path.resolve())

    return str(ensure_onnxruntime_sdk(repo_root))


def ensure_onnxruntime_lib_on_path(repo_root: Path | None = None) -> str:
    """Prepend the ONNX Runtime SDK lib/ directory to LD_LIBRARY_PATH for JAGS."""
    lib_dir = str(Path(resolve_onnxruntime_sdk_dir(repo_root)) / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in existing.split(":") if part]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{lib_dir}:{existing}" if existing else lib_dir
        )
    return lib_dir

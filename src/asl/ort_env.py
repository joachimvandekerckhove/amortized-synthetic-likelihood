"""Configure ONNX Runtime before the pip package is imported."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_configured = False
_ORT_NOISE_MARKERS = (
    "device_discovery",
    "GetGpuDevices",
    "/sys/class/drm",
)


class _OrtImportNoiseFilter:
    """Drop ONNX Runtime GPU probe noise on stderr during import."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text: str) -> int:
        if any(marker in text for marker in _ORT_NOISE_MARKERS):
            return len(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()


@contextmanager
def _silence_stderr_fd() -> Iterator[None]:
    """Redirect fd 2 during import (ORT logs below Python stderr)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def configure_onnxruntime_env() -> None:
    """Best-effort log/GPU settings (recovery uses CPU inference)."""
    global _configured
    if _configured:
        return
    os.environ.setdefault("ORT_LOG_LEVEL", "3")
    os.environ.setdefault("ORT_DISABLE_GPU", "1")
    _configured = True


def _import_onnxruntime():
    configure_onnxruntime_env()
    stderr = sys.stderr
    sys.stderr = _OrtImportNoiseFilter(stderr)
    try:
        with _silence_stderr_fd():
            import onnxruntime as ort_mod
    finally:
        sys.stderr = stderr
    return ort_mod


ort = _import_onnxruntime()


def cpu_inference_session(onnx_path: str | Path) -> ort.InferenceSession:
    """Load ONNX for CPU inference without GPU provider probing."""
    return ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

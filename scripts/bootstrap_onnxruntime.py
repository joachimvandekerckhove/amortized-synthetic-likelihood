#!/usr/bin/env python3
"""Download the ONNX Runtime C/C++ SDK into vendor/ (optional prefetch)."""

from asl.onnxruntime_sdk import ensure_onnxruntime_sdk, find_repo_root

if __name__ == "__main__":
    path = ensure_onnxruntime_sdk(find_repo_root())
    print(path)

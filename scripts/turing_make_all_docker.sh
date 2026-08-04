#!/usr/bin/env bash
# Full pipeline validation on Turing: fresh Docker container with GPU support.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

docker run --rm -v "$ROOT":/work python:3.11-slim-bookworm rm -rf /work/.venv

exec docker run --rm --gpus all --cpus=16 \
  -v "$ROOT":/work -w /work \
  -e HOST_UID="$HOST_UID" -e HOST_GID="$HOST_GID" \
  python:3.11-slim-bookworm \
  bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential jags pkg-config make g++ procps git
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[jags,dev]"
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
make bootstrap-ort
.venv/bin/python -m pytest -q
make all
make vpw08
.venv/bin/python -m pytest -q scripts/vpw08/tests/
chown -R "${HOST_UID}:${HOST_GID}" .venv data results figures models tmp logs 2>/dev/null || true
'

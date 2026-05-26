#!/usr/bin/env bash
set -euo pipefail

# Boot script for a RunPod GPU pod hosting the SoulX-Singer inference service.
# Designed to run from a stock CUDA-flavored PyTorch image on every pod start.
#
# Assumptions:
#   - Base image provides CUDA 12.1 runtime and a system python (any version).
#     We install python3.10 ourselves because SoulX-Singer requires it.
#   - Weights are re-downloaded on every boot (no persistent volume).
#
# Environment knobs:
#   PORT            (default 8000)      HTTP port for the FastAPI server.
#   SOULX_DEVICE    (default cuda)      Device passed to SoulX worker.
#   WORKSPACE       (default /workspace) Base directory for clones and venv.
#   SERVER_URL      Raw URL to fetch server.py from (e.g., a GitHub raw URL or
#                   Gist). If unset and /workspace/server.py is already present,
#                   we use that copy as-is.

WORKSPACE="${WORKSPACE:-/workspace}"
ENGINE_DIR="${WORKSPACE}/SoulX-Singer"
SERVER_PATH="${WORKSPACE}/server.py"
PORT="${PORT:-8000}"

mkdir -p "${WORKSPACE}"

echo "[bootstrap] installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  git curl ca-certificates ffmpeg \
  python3.10 python3.10-venv python3.10-dev
rm -rf /var/lib/apt/lists/*

if [ -n "${SERVER_URL:-}" ]; then
  echo "[bootstrap] fetching server.py from ${SERVER_URL}"
  curl -fsSL "${SERVER_URL}" -o "${SERVER_PATH}"
fi

if [ ! -f "${SERVER_PATH}" ]; then
  echo "[bootstrap] ERROR: ${SERVER_PATH} missing and SERVER_URL not set." >&2
  exit 1
fi

echo "[bootstrap] cloning SoulX-Singer"
if [ ! -d "${ENGINE_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/Soul-AILab/SoulX-Singer.git "${ENGINE_DIR}"
fi

cd "${ENGINE_DIR}"
if [ ! -x ".venv/bin/python" ]; then
  echo "[bootstrap] creating python3.10 venv"
  python3.10 -m venv .venv
fi

PIP=".venv/bin/python -m pip"
${PIP} install --upgrade pip setuptools wheel

echo "[bootstrap] installing CUDA torch first so requirements.txt doesn't pull the CPU wheel"
${PIP} install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.0 torchaudio==2.2.0

echo "[bootstrap] installing SoulX-Singer requirements"
${PIP} install -r requirements.txt
${PIP} install "huggingface_hub>=0.23,<1.0" fastapi "uvicorn[standard]" python-multipart

echo "[bootstrap] downloading SoulX-Singer model weights"
.venv/bin/hf download Soul-AILab/SoulX-Singer model.pt \
  --local-dir pretrained_models/SoulX-Singer

echo "[bootstrap] launching FastAPI server on 0.0.0.0:${PORT}"
cd "${WORKSPACE}"
export SOULX_REPO="${ENGINE_DIR}"
export SOULX_PYTHON="${ENGINE_DIR}/.venv/bin/python"
export SOULX_DEVICE="${SOULX_DEVICE:-cuda}"
exec "${ENGINE_DIR}/.venv/bin/python" -m uvicorn server:app \
  --host 0.0.0.0 --port "${PORT}" --log-level info

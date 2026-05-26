#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  echo "[bootstrap] $*"
}

die() {
  echo "[bootstrap] ERROR: $*" >&2
  exit 1
}

export DEBIAN_FRONTEND=noninteractive

WORKSPACE="${WORKSPACE:-/workspace}"
ENGINE_DIR="${ENGINE_DIR:-${WORKSPACE}/SoulX-Singer}"
SERVER_PATH="${SERVER_PATH:-${WORKSPACE}/server.py}"
PORT="${PORT:-8000}"
SOULX_DEVICE="${SOULX_DEVICE:-cuda}"
SOULX_REPO_URL="${SOULX_REPO_URL:-https://github.com/Soul-AILab/SoulX-Singer.git}"
HF_MODEL_REPO="${HF_MODEL_REPO:-Soul-AILab/SoulX-Singer}"

export HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE}/.cache/pip}"

mkdir -p "${WORKSPACE}" "${HF_HOME}" "${PIP_CACHE_DIR}"

log "installing system packages"

apt-get update -y

apt-get install -y --no-install-recommends \
  git \
  curl \
  ca-certificates \
  ffmpeg \
  python3 \
  python3-venv \
  python3-dev \
  python3-pip

# Prefer Python 3.10 if available in the base image / apt repo.
# Do not fail the whole bootstrap if this package does not exist.
apt-get install -y --no-install-recommends \
  python3.10 \
  python3.10-venv \
  python3.10-dev || true

rm -rf /var/lib/apt/lists/*

if command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.10)"
else
  PYTHON_BIN="$(command -v python3)"
fi

log "using Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python 3.10+ is required, found {sys.version.split()[0]}"
    )

print(f"[bootstrap] Python version OK: {sys.version.split()[0]}")
PY

if [ -n "${SERVER_URL:-}" ]; then
  log "fetching server.py from ${SERVER_URL}"
  curl -fsSL "${SERVER_URL}" -o "${SERVER_PATH}"
fi

if [ ! -f "${SERVER_PATH}" ]; then
  die "${SERVER_PATH} missing. Set SERVER_URL or place server.py in ${WORKSPACE}."
fi

log "checking server.py syntax"

if ! "${PYTHON_BIN}" -m py_compile "${SERVER_PATH}"; then
  die "server.py has syntax errors. Make sure it also has real line breaks in GitHub."
fi

log "cloning SoulX-Singer"

if [ ! -d "${ENGINE_DIR}/.git" ]; then
  git clone --depth 1 "${SOULX_REPO_URL}" "${ENGINE_DIR}"
else
  log "SoulX-Singer already exists, pulling latest"
  git -C "${ENGINE_DIR}" pull --ff-only || true
fi

cd "${ENGINE_DIR}"

# The upstream requirements.txt may be accidentally stored as one long line.
# pip expects one requirement per line, so normalize it safely.
if [ -f requirements.txt ]; then
  log "normalizing requirements.txt if needed"

  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

p = Path("requirements.txt")
text = p.read_text(encoding="utf-8")
stripped = text.strip()

if stripped and "\n" not in stripped:
    items = stripped.split()
    p.write_text("\n".join(items) + "\n", encoding="utf-8")
    print(f"[bootstrap] normalized requirements.txt into {len(items)} lines")
else:
    print("[bootstrap] requirements.txt already looks line-based")
PY
fi

VENV_DIR="${ENGINE_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
  log "creating virtualenv"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PIP=("${VENV_PYTHON}" -m pip)

log "upgrading pip tooling"

"${PIP[@]}" install --upgrade \
  pip \
  setuptools \
  wheel

log "installing CUDA PyTorch first"

"${PIP[@]}" install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.0 \
  torchaudio==2.2.0

log "installing SoulX-Singer requirements"

"${PIP[@]}" install -r requirements.txt

log "installing API server dependencies"

"${PIP[@]}" install \
  "huggingface_hub>=0.23,<1.0" \
  fastapi \
  "uvicorn[standard]" \
  python-multipart

log "downloading SoulX-Singer model weights"

"${VENV_DIR}/bin/hf" download "${HF_MODEL_REPO}" model.pt \
  --local-dir pretrained_models/SoulX-Singer

MODEL_PATH="${ENGINE_DIR}/pretrained_models/SoulX-Singer/model.pt"

if [ ! -f "${MODEL_PATH}" ]; then
  die "model file was not downloaded: ${MODEL_PATH}"
fi

log "checking torch CUDA"

"${VENV_PYTHON}" - <<'PY'
import torch

print(f"[bootstrap] torch: {torch.__version__}")
print(f"[bootstrap] cuda available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"[bootstrap] cuda device: {torch.cuda.get_device_name(0)}")
PY

log "launching FastAPI server on 0.0.0.0:${PORT}"

cd "${WORKSPACE}"

export SOULX_REPO="${ENGINE_DIR}"
export SOULX_PYTHON="${VENV_PYTHON}"
export SOULX_DEVICE="${SOULX_DEVICE}"

exec "${VENV_PYTHON}" -m uvicorn server:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --log-level info

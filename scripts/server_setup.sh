#!/usr/bin/env bash
# HERP v3 — one-time server bootstrap.
#
# Creates conda env ``deeplearning`` (Python 3.12) if missing, installs
# torch (CUDA 12.8), then pip installs the rest of ``requirements.txt``.
# Skip pieces you already have via env vars:
#
#   CONDA_ENV=deeplearning ./scripts/server_setup.sh
#   SKIP_TORCH=1 ./scripts/server_setup.sh          # if torch already installed
#   CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121 ./scripts/server_setup.sh
#
# Also verifies mani-skill can import and prints the PhysX GPU probe result.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    for cand in /opt/miniconda3 /opt/conda /opt/anaconda3 "$HOME/anaconda3"; do
        if [[ -f "$cand/etc/profile.d/conda.sh" ]]; then CONDA_BASE="$cand"; break; fi
    done
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

CONDA_ENV="${CONDA_ENV:-deeplearning}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "[server_setup] creating conda env $CONDA_ENV (python=$PYTHON_VERSION)"
    conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi
conda activate "$CONDA_ENV"

python -m pip install --upgrade pip setuptools wheel

if [[ "${SKIP_TORCH:-0}" == "0" ]]; then
    echo "[server_setup] installing torch from $CUDA_INDEX_URL"
    pip install --extra-index-url "$CUDA_INDEX_URL" "torch==2.11.0" || \
        pip install "torch==2.11.0"
fi

pip install -r requirements.txt
pip install -e .

echo "[server_setup] verifying mani-skill import"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/lib/wsl/lib}" \
python -c "
import torch, mani_skill
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
print('mani_skill', mani_skill.__version__)
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[server_setup] optional: run 'wandb login' for online logging, or export WANDB_API_KEY"
fi

# --- Third-party baselines --------------------------------------------------
# Only run these if the user hasn't opted out; they add ~10-30 min of first-run
# setup for the external SAC/MBRL baselines (BRO, MaxInfoRL, TD-MPC2 upstream).
if [[ "${SKIP_THIRD_PARTY:-0}" == "0" ]]; then
    echo "[server_setup] fetching third_party locked baselines"
    ./scripts/fetch_third_party.sh
fi
if [[ "${SKIP_SECONDARY_VENVS:-0}" == "0" ]]; then
    echo "[server_setup] setting up secondary venvs (tdmpc2/maxinforl/bro)"
    ./scripts/setup_secondary_venvs.sh
fi
echo "[server_setup] done. Launch: ./scripts/server_deploy.sh"

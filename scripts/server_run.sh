#!/usr/bin/env bash
# HERP v3 — one-command server launcher.
#
# Prereqs on the target box (one-time):
#   * a conda install with an env named ``deeplearning`` that satisfies
#     ``requirements.txt`` (torch+CUDA, mani-skill, wandb, ...); ``server_setup.sh``
#     bootstraps this if you don't already have it.
#   * a GPU with PhysX (any recent NVIDIA); the WSL-specific
#     ``LD_LIBRARY_PATH`` is set unconditionally and harmless on native Linux.
#   * ``wandb login`` already run (or ``WANDB_API_KEY`` exported) for online
#     mode; pass ``WANDB_MODE=offline`` here to skip the login.
#
# Usage:
#   ./scripts/server_run.sh                       # 1M-step full campaign
#   OUTPUT_DIR=... ./scripts/server_run.sh
#   WANDB_MODE=offline ./scripts/server_run.sh
#   SKIP_SIGMA=1 ./scripts/server_run.sh          # skip Phase 1
#   ONLY=herp,ppo ./scripts/server_run.sh         # narrow the main queue
#
# The script is idempotent: re-running picks up where the last run stopped
# (per-method ``summary.json`` + latest ``checkpoint_*.pt`` are honored).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --- conda activate --------------------------------------------------------
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    for cand in /opt/miniconda3 /opt/conda /opt/anaconda3 "$HOME/anaconda3"; do
        if [[ -f "$cand/etc/profile.d/conda.sh" ]]; then CONDA_BASE="$cand"; break; fi
    done
fi
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: cannot locate conda; set CONDA_BASE=..." >&2; exit 2
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

CONDA_ENV="${CONDA_ENV:-deeplearning}"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "ERROR: conda env '$CONDA_ENV' not found. Run scripts/server_setup.sh first." >&2
    exit 2
fi
conda activate "$CONDA_ENV"

# --- runtime env -----------------------------------------------------------
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/lib/wsl/lib}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# --- campaign arguments ----------------------------------------------------
STAMP="$(date -u +%Y%m%d_%H%M)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/campaign_1m_${STAMP}}"
ENV_ID="${ENV_ID:-PickCube-v1}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-512}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-32}"
SIGMA_TIMESTEPS="${SIGMA_TIMESTEPS:-200000}"
MAIN_TIMESTEPS="${MAIN_TIMESTEPS:-1000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100000}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-herp-v3}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
PRIORITY_FIRST="${PRIORITY_FIRST:-ppo,herp}"
DEADLINE_HOURS="${DEADLINE_HOURS:-0}"

# --- wandb sanity check ----------------------------------------------------
if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    if ! python -c 'import netrc,os,sys; sys.exit(0 if netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai") else 1)' 2>/dev/null; then
        echo "WARN: WANDB_MODE=online but no WANDB_API_KEY / login found; falling back to offline." >&2
        WANDB_MODE=offline
    fi
fi

extra_flags=()
if [[ "${SKIP_SIGMA:-0}" != "0" ]]; then extra_flags+=(--skip-sigma); fi
if [[ -n "${ONLY:-}" ]]; then extra_flags+=(--only "$ONLY"); fi

mkdir -p "$OUTPUT_DIR"
echo "[server_run] output_dir=$OUTPUT_DIR wandb=$WANDB_MODE project=$WANDB_PROJECT"
echo "[server_run] main_timesteps=$MAIN_TIMESTEPS sigma_timesteps=$SIGMA_TIMESTEPS num_envs=$NUM_ENVS"

exec python -u scripts/run_campaign_1m.py \
    --output-dir "$OUTPUT_DIR" \
    --env-id "$ENV_ID" \
    --seed "$SEED" \
    --num-envs "$NUM_ENVS" \
    --num-eval-envs "$NUM_EVAL_ENVS" \
    --sigma-timesteps "$SIGMA_TIMESTEPS" \
    --main-timesteps "$MAIN_TIMESTEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-episodes "$EVAL_EPISODES" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-group "$WANDB_GROUP" \
    --priority-first "$PRIORITY_FIRST" \
    --deadline-hours "$DEADLINE_HOURS" \
    "${extra_flags[@]}"

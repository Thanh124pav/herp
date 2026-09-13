#!/usr/bin/env bash
# HERP framework — end-to-end deploy for RTX 5090 (or similar >=24GB GPU).
#
# Runs the full paper pipeline in one shot:
#   preflight → pilot → phase 1 ablations → phase 2 performance → merge
#
# Idempotent: every run writes a summary.json; re-running skips completed
# cells. Kill and re-run at any time.
#
# Quick start (5090, seed 0, everything):
#     ./scripts/server_deploy.sh
#
# Overridable env vars (defaults in brackets):
#   STAGE                 which stage(s) to run [all|pilot|phase1|phase2|merge]
#   OUTPUT_ROOT           campaign root dir             [outputs/framework_YYYYMMDD_HHMM]
#   WANDB_PROJECT         W&B project                   [herp-framework]
#   SEEDS                 space-separated seeds         [0]
#   NUM_ENVS_PPO          PPO/HERP-PPO parallel envs    [auto: 512/1024/2048]
#   NUM_ENVS_SAC          SAC/HERP-SAC parallel envs    [auto: 16/32/64]
#   PILOT_TASK            single easy task for pilot    [PickCube-v1]
#   PILOT_BUDGET          headroom for T calibration    [3000000]
#   PHASE1_TASK           easy task for ablations       [PickCube-v1]
#   PHASE1_T              per-run budget                [1500000]
#   PHASE2_TASKS          quoted list, easy→hard        ["PickCube-v1 StackCube-v1 PegInsertionSide-v1"]
#   PHASE2_T              per-run budget                [2000000]
#   PHASE2_EXTRA_SEEDS    additional seeds (Phase 2)    [""]  # e.g. "1 2"
#   SKIP_PREFLIGHT        1 = skip preflight check      [0]
#
# Example single-stage:
#   STAGE=phase1 SEEDS="0 1 2" ./scripts/server_deploy.sh
#
# Example wandb offline:
#   WANDB_MODE=offline ./scripts/server_deploy.sh

set -euo pipefail

STAGE="${STAGE:-all}"
OUTPUT_ROOT_DEFAULT="outputs/framework_$(date -u +%Y%m%d_%H%M)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"
WANDB_PROJECT="${WANDB_PROJECT:-herp-framework}"
SEEDS="${SEEDS:-0}"
PILOT_TASK="${PILOT_TASK:-PickCube-v1}"
PILOT_BUDGET="${PILOT_BUDGET:-3000000}"
PHASE1_TASK="${PHASE1_TASK:-PickCube-v1}"
PHASE1_T="${PHASE1_T:-1500000}"
PHASE2_TASKS="${PHASE2_TASKS:-PickCube-v1 StackCube-v1 PegInsertionSide-v1}"
PHASE2_T="${PHASE2_T:-2000000}"
PHASE2_EXTRA_SEEDS="${PHASE2_EXTRA_SEEDS:-}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

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
    echo "ERROR: conda env '$CONDA_ENV' missing. Run scripts/server_setup.sh first." >&2
    exit 2
fi
conda activate "$CONDA_ENV"

# --- runtime env -----------------------------------------------------------
# LD_LIBRARY_PATH for WSL PhysX is harmless on native Linux.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/lib/wsl/lib}"
# Vulkan CPU ICD only needed for physx_cpu; on 5090 the physx_cuda backend
# uses the NVIDIA driver directly.
if [[ -z "${VK_ICD_FILENAMES:-}" ]] && [[ -f /usr/share/vulkan/icd.d/lvp_icd.json ]]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# --- auto-scale num_envs based on GPU VRAM ---------------------------------
GPU_MEM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($1/1024)}' || echo 0)"
if [[ -z "${NUM_ENVS_PPO:-}" ]]; then
    if   [[ $GPU_MEM_GB -ge 24 ]]; then NUM_ENVS_PPO=2048
    elif [[ $GPU_MEM_GB -ge 12 ]]; then NUM_ENVS_PPO=1024
    else NUM_ENVS_PPO=512; fi
fi
if [[ -z "${NUM_ENVS_SAC:-}" ]]; then
    if   [[ $GPU_MEM_GB -ge 24 ]]; then NUM_ENVS_SAC=64
    elif [[ $GPU_MEM_GB -ge 12 ]]; then NUM_ENVS_SAC=32
    else NUM_ENVS_SAC=16; fi
fi

# --- wandb sanity ----------------------------------------------------------
WANDB_MODE="${WANDB_MODE:-online}"
if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    if ! python -c 'import netrc,os,sys; sys.exit(0 if netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai") else 1)' 2>/dev/null; then
        echo "WARN: WANDB_MODE=online but no WANDB_API_KEY / login found; falling back to offline." >&2
        WANDB_MODE=offline
    fi
fi
export WANDB_MODE

mkdir -p "$OUTPUT_ROOT"
DEPLOY_LOG="$OUTPUT_ROOT/deploy.log"

echo "======================================" | tee -a "$DEPLOY_LOG"
echo "HERP framework deploy — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DEPLOY_LOG"
echo "  stage:          $STAGE" | tee -a "$DEPLOY_LOG"
echo "  output_root:    $OUTPUT_ROOT" | tee -a "$DEPLOY_LOG"
echo "  GPU:            ${GPU_MEM_GB}GB" | tee -a "$DEPLOY_LOG"
echo "  num_envs_ppo:   $NUM_ENVS_PPO" | tee -a "$DEPLOY_LOG"
echo "  num_envs_sac:   $NUM_ENVS_SAC" | tee -a "$DEPLOY_LOG"
echo "  seeds:          $SEEDS" | tee -a "$DEPLOY_LOG"
echo "  wandb:          $WANDB_MODE $WANDB_PROJECT" | tee -a "$DEPLOY_LOG"
echo "======================================" | tee -a "$DEPLOY_LOG"

# --- preflight -------------------------------------------------------------
if [[ "$SKIP_PREFLIGHT" == "0" ]]; then
    echo "[$(date -u +%H:%M:%S)] running preflight_check.py" | tee -a "$DEPLOY_LOG"
    python scripts/preflight_check.py 2>&1 | tee -a "$DEPLOY_LOG"
fi

# --- helper: run one stage -------------------------------------------------
_run_stage() {
    local phase="$1"; shift
    local subdir="$1"; shift
    echo "[$(date -u +%H:%M:%S)] STAGE=$phase → $OUTPUT_ROOT/$subdir" | tee -a "$DEPLOY_LOG"
    python -u scripts/run_framework_campaign.py \
        --phase "$phase" \
        --num-envs-ppo "$NUM_ENVS_PPO" --num-envs-sac "$NUM_ENVS_SAC" \
        --output-dir "$OUTPUT_ROOT/$subdir" \
        --execute \
        "$@" \
        2>&1 | tee -a "$DEPLOY_LOG"
    echo "[$(date -u +%H:%M:%S)] STAGE=$phase done" | tee -a "$DEPLOY_LOG"
}

# --- stage: pilot ----------------------------------------------------------
if [[ "$STAGE" == "pilot" || "$STAGE" == "all" ]]; then
    _run_stage pilot pilot \
        --tasks $PILOT_TASK --seeds $SEEDS --budget "$PILOT_BUDGET"
fi

# --- stage: phase1 (HERP-PPO sigma ablations) ------------------------------
if [[ "$STAGE" == "phase1" || "$STAGE" == "all" ]]; then
    _run_stage ablation phase1 \
        --tasks $PHASE1_TASK --seeds $SEEDS --selected-t "$PHASE1_T"
fi

# --- stage: phase2 (cross-backbone performance) ----------------------------
if [[ "$STAGE" == "phase2" || "$STAGE" == "all" ]]; then
    _run_stage performance phase2 \
        --tasks $PHASE2_TASKS --seeds $SEEDS --selected-t "$PHASE2_T"

    # Optional extra seeds for the winning cells.
    if [[ -n "$PHASE2_EXTRA_SEEDS" ]]; then
        echo "[$(date -u +%H:%M:%S)] STAGE=phase2 extra seeds $PHASE2_EXTRA_SEEDS" | tee -a "$DEPLOY_LOG"
        _run_stage performance phase2 \
            --tasks $PHASE2_TASKS --seeds $PHASE2_EXTRA_SEEDS --selected-t "$PHASE2_T"
    fi
fi

# --- stage: merge (build the paper tables) ---------------------------------
if [[ "$STAGE" == "merge" || "$STAGE" == "all" ]]; then
    MERGE_ROOTS=()
    for d in pilot phase1 phase2; do
        [[ -d "$OUTPUT_ROOT/$d" ]] && MERGE_ROOTS+=("$OUTPUT_ROOT/$d")
    done
    if [[ ${#MERGE_ROOTS[@]} -eq 0 ]]; then
        echo "[merge] no stage outputs found; skipping" | tee -a "$DEPLOY_LOG"
    else
        echo "[$(date -u +%H:%M:%S)] merging: ${MERGE_ROOTS[*]}" | tee -a "$DEPLOY_LOG"
        python -u scripts/merge_cross_backbone_results.py "${MERGE_ROOTS[@]}" \
            --output-dir "$OUTPUT_ROOT/merged" --budget "$PHASE2_T" \
            2>&1 | tee -a "$DEPLOY_LOG"
        echo "[$(date -u +%H:%M:%S)] merged tables → $OUTPUT_ROOT/merged" | tee -a "$DEPLOY_LOG"
    fi
fi

echo "[$(date -u +%H:%M:%S)] deploy pipeline done → $OUTPUT_ROOT" | tee -a "$DEPLOY_LOG"

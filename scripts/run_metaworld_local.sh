#!/usr/bin/env bash
#
# run_metaworld_local.sh -- reproduce the HERP experiments on the OFFICIAL
# Meta-World suite, on a local machine (e.g. a GTX 1650).
#
# What it does:
#   1. activates the existing miniconda3 env `deeplearning`
#   2. installs the project + RL backbone (PyTorch/CUDA) + Meta-World extras
#      (skips torch if the env already has it, so your setup is respected)
#   3. runs a fast smoke test on metaworld:reach-v2 to verify the install
#   4. runs the full UOT experiment and the matched-budget routing comparison
#
# NOTE ON HARDWARE:
#   - The SAC networks are tiny MLPs (<1 GB VRAM); a GTX 1650 is plenty.
#   - Meta-World physics (MuJoCo) steps on the CPU -- the GPU is NOT the
#     bottleneck; wall-clock is dominated by env stepping. reach-v2 is the
#     fastest-converging task, chosen deliberately for a first real run.
#
# Usage:
#   bash scripts/run_metaworld_local.sh                 # reach-v2, full run
#   TASK=push-v2 STEPS=500000 bash scripts/run_metaworld_local.sh
#   SMOKE_ONLY=1 bash scripts/run_metaworld_local.sh    # just verify install
#
set -euo pipefail

# ---- config (override via env vars) ----------------------------------------
CONDA_ENV="${CONDA_ENV:-deeplearning}"
TASK="${TASK:-reach-v2}"            # metaworld task; reach-v2 converges fastest
STEPS="${STEPS:-300000}"           # total env interactions (all policies)
POP="${POP:-4}"                    # population size N
SEED="${SEED:-0}"
CUDA="${CUDA:-cu121}"              # torch CUDA build; GTX 1650 (Turing) supports cu118/cu121
SMOKE_ONLY="${SMOKE_ONLY:-0}"

# ---- locate repo root (this script lives in <repo>/scripts) ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "==> repo: $REPO_ROOT"

# ---- activate conda env ----------------------------------------------------
echo "==> activating conda env: $CONDA_ENV"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "    python: $(which python)  ($(python --version 2>&1))"

# ---- dependencies ----------------------------------------------------------
# PyTorch: only install if the env doesn't already provide it (respect your setup).
if python -c "import torch" 2>/dev/null; then
  echo "==> torch already present: $(python -c 'import torch;print(torch.__version__, "cuda="+str(torch.cuda.is_available()))')"
else
  echo "==> installing PyTorch ($CUDA) ..."
  pip install torch --index-url "https://download.pytorch.org/whl/${CUDA}"
fi

echo "==> installing project (editable) + core deps (numpy/scipy/POT/matplotlib) ..."
pip install -e .

# Meta-World (Farama). The PyPI package can lag; install from source to be safe.
if python -c "import metaworld" 2>/dev/null; then
  echo "==> metaworld already present"
else
  echo "==> installing Meta-World + MuJoCo + gymnasium ..."
  pip install "mujoco" "gymnasium>=0.29"
  pip install "git+https://github.com/Farama-Foundation/Metaworld.git"
fi

# Headless MuJoCo: stepping physics (state obs) needs no display; force EGL if a
# GL context is ever requested so the run never blocks on a missing display.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ---- sanity: unit tests (incl. the OT gate that must pass before routing) ---
echo "==> running unit tests ..."
pytest -q || { echo "!! unit tests failed -- fix before trusting any run"; exit 1; }

# ---- smoke test on the real env -------------------------------------------
echo "==> SMOKE: metaworld:${TASK} (quick) ..."
python scripts/run_full.py --env "metaworld:${TASK}" --router uot --quick --seed "$SEED"

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "==> SMOKE_ONLY set -- install verified, stopping here."
  exit 0
fi

# ---- full UOT run + dashboard artifacts -----------------------------------
echo "==> FULL UOT run: metaworld:${TASK}, N=${POP}, steps=${STEPS} ..."
python scripts/run_full.py \
  --env "metaworld:${TASK}" \
  --population-size "$POP" \
  --router uot \
  --total-env-steps "$STEPS" \
  --seed "$SEED"

# ---- matched-budget routing comparison (Experiment 4, B1-B7) ---------------
echo "==> ROUTING COMPARISON (matched budget): no_share share_all random td_priority greedy uot ..."
python scripts/run_baseline.py \
  --routers no_share share_all random td_priority greedy uot \
  --population-size "$POP" \
  --total-env-steps "$STEPS" \
  --seed "$SEED" \
  --outdir "outputs/comparison_metaworld_${TASK}"

echo ""
echo "==> DONE."
echo "    Dashboard PNGs:  outputs/metaworld_${TASK}_uot_seed${SEED}/"
echo "    Comparison:      outputs/comparison_metaworld_${TASK}/ (comparison.png + comparison.json)"
echo "    Drop those PNGs next to report/herp_report.tex and uncomment the figure block."

#!/usr/bin/env bash
# Create the isolated Python venvs used by external baselines whose
# dependencies would conflict with the main `deeplearning` conda env
# (mostly TD-MPC2 hydra pins and BRO's JAX stack).
#
# Idempotent: skips venvs that already have the right Python + a
# reasonable package count.
#
# Which venvs get created is controlled by env vars (default: all).
#   ONLY="tdmpc2 bro"         # create just these
#   SKIP="maxinforl"          # create everything except these
#
# Prereqs:
#   * python3.11 available on PATH (only for BRO). Install with e.g.:
#       sudo apt-get install python3.11 python3.11-venv
#   * python3.12 available on PATH (for TD-MPC2, MaxInfoRL — same as main).
#
# Pinned requirements live in docker/requirements-{tdmpc2,bro,maxinforl}.txt

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

_want() {
    local name="$1"
    [[ -n "${ONLY:-}" ]] && ! echo " $ONLY " | grep -q " $name " && return 1
    [[ -n "${SKIP:-}" ]] && echo " $SKIP " | grep -q " $name " && return 1
    return 0
}

_setup() {
    local name="$1" py="$2"
    local dir=".venvs/$name" reqs="docker/requirements-$name.txt"
    if ! _want "$name"; then echo "[venvs] $name — skipped"; return; fi
    if [[ ! -f "$reqs" ]]; then
        echo "[venvs] $name — SKIP: no $reqs"; return
    fi
    if ! command -v "$py" >/dev/null 2>&1; then
        echo "[venvs] $name — SKIP: $py not on PATH (install $py + $py-venv)"; return
    fi
    if [[ -x "$dir/bin/python" ]]; then
        local have_ver="$("$dir/bin/python" -V 2>&1 | awk '{print $2}')"
        local want_ver="$("$py" -V 2>&1 | awk '{print $2}')"
        if [[ "${have_ver%.*}" == "${want_ver%.*}" ]]; then
            local pkg_count="$("$dir/bin/pip" list --format=freeze 2>/dev/null | wc -l)"
            if [[ $pkg_count -gt 50 ]]; then
                echo "[venvs] $name — already installed (py=$have_ver, $pkg_count pkgs)"
                return
            fi
        fi
    fi
    echo "[venvs] $name — creating $dir with $py"
    rm -rf "$dir"
    "$py" -m venv "$dir"
    "$dir/bin/python" -m pip install --upgrade pip setuptools wheel
    echo "[venvs] $name — installing $reqs (this can take 10-20 minutes)"
    # Torch pins in the freeze were CUDA-suffixed locally; strip the +cuXXX
    # suffix in requirements files and add the CUDA index here so the CUDA
    # wheels resolve on servers with matching CUDA runtime (falls through to
    # CPU wheel if index is unreachable).
    "$dir/bin/pip" install --extra-index-url "${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
        -r "$reqs"
    echo "[venvs] $name — done ($(($("$dir/bin/pip" list --format=freeze | wc -l))) packages)"
}

# Python 3.12 for TD-MPC2 and MaxInfoRL (same version as main env).
_setup tdmpc2 python3.12
_setup maxinforl python3.12
# Python 3.11 for BRO (its JAX pins want a slightly older interpreter).
_setup bro python3.11

echo "[venvs] all done — .venvs/"
ls -la .venvs/

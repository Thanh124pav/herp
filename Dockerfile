# HERP — reproducible GPU environment.
#
# Base: NVIDIA CUDA 12.8 runtime on Ubuntu 22.04 — matches driver_max_cuda ≥ 12.0
# on RTX 30/40/50 (Ampere/Ada/Blackwell) hosts. See IMPLEMENTATION.md §2.
#
# Build:  docker build -t herp:cu128 .
# Run:    docker run --gpus all -it --rm -v $(pwd)/outputs:/workspace/outputs herp:cu128
# Verify: pytest -q  (inside the container)
#
# The image installs Python 3.12, apt deps from §2.2, PyTorch cu128 from
# download.pytorch.org, everything in requirements.txt, and this repo in
# editable mode. It does NOT pre-download the ManiSkill PhysX 5.3 GPU library —
# that is a ~200 MB one-time fetch at first ``physx_cuda`` init, cached under
# ``/root/.sapien/physx/`` (or your ``$HOME/.sapien`` when running as a non-root
# user); mount that directory back in with ``-v $HOME/.sapien:/root/.sapien`` to
# reuse it across container rebuilds.

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    LANG=C.UTF-8

# --- System packages (IMPLEMENTATION.md §2.2) -------------------------------
# Notes on the Python install:
#   * Ubuntu 22.04 ships Python 3.10; we install 3.11 AND 3.12 from deadsnakes.
#     3.12 is the main HERP interpreter; 3.11 is only used for the BRO baseline
#     whose JAX pins want a slightly older Python — see docker/requirements-bro.txt.
#   * We do NOT install the distro `python3-pip` — colliding pips between 3.10
#     and 3.12 broke earlier builds. `ensurepip` from the -venv package
#     bootstraps pip inside each interpreter cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl unzip ca-certificates \
        software-properties-common gnupg \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        libglfw3 libglew-dev libglvnd-dev \
        libvulkan1 mesa-vulkan-drivers \
        libegl1 libgl1 libgles2 libosmesa6 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Python ----------------------------------------------------------------
# Symlink `python` -> 3.12, then bootstrap pip *for the 3.12 interpreter*.
# `ensurepip` is shipped with `python3.12-venv` and installs a matching
# `pip` module inside 3.12's site-packages; only after that will
# `python -m pip …` succeed.
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
 && python -m ensurepip --upgrade \
 && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/herp

# Install PyTorch on cu128 FIRST so subsequent wheels don't pull the CPU build.
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.11.0+cu128"

# Copy just the manifests so the layer cache survives source edits.
COPY requirements.txt pyproject.toml ./
RUN python -m pip install -r requirements.txt

# Now the source tree, then install as editable.
COPY . .
RUN python -m pip install -e .

# --- Third-party baselines + isolated venvs --------------------------------
# Clones RFCL / ActiveRL / BRO / MaxInfoRL at the SHAs pinned in
# docs/v3/external_baselines.lock.json, plus a sparse checkout of the
# ManiSkill upstream examples/baselines tree (needed by
# scripts/tdmpc2_official.py). Cached in the image so container starts are
# offline-friendly. Comment out this block if you only need the main HERP
# path and want a smaller image (~4 GB smaller without the venvs below).
RUN chmod +x scripts/fetch_third_party.sh scripts/setup_secondary_venvs.sh \
             scripts/server_deploy.sh scripts/server_run.sh scripts/server_setup.sh \
 && ./scripts/fetch_third_party.sh

# Secondary venvs for TD-MPC2 (py3.12), MaxInfoRL (py3.12), BRO (py3.11)
# are NOT built here — the pinned requirements have cross-repo CUDA-toolkit
# conflicts that make the resolver fragile in unattended CI. HERP-PPO,
# HERP-SAC (vectorized), and vanilla SAC/PPO all run from the main env
# above; the venvs are only needed for the external SAC/MBRL baselines
# (BRO, MaxInfoRL, TD-MPC2). To install them on the server:
#     docker run ... herp:cu128 ./scripts/setup_secondary_venvs.sh
# Or opt in at build time by uncommenting the RUN below (may fail):
# ARG BUILD_SECONDARY_VENVS=0
# RUN if [ "$BUILD_SECONDARY_VENVS" = "1" ]; then ./scripts/setup_secondary_venvs.sh; fi

# --- Optional runtime knobs -----------------------------------------------
# Determinism (§8.3) — cudnn deterministic + no benchmark. The training scripts
# also set torch.manual_seed / np seeds; these env vars help third-party wheels.
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MUJOCO_GL=egl

# Non-interactive default: run the full framework pipeline via server_deploy.sh.
# Override for smoke tests or single stages via `docker run ... <cmd>`.
# Example single-stage run:
#     docker run --gpus all -v $HOME/.sapien:/root/.sapien \
#                -v $(pwd)/outputs:/workspace/herp/outputs \
#                -v $HOME/.netrc:/root/.netrc:ro \
#                -e STAGE=phase1 herp:cu128
CMD ["./scripts/server_deploy.sh"]

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
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl unzip ca-certificates \
        software-properties-common \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        libglfw3 libglew-dev libglvnd-dev \
        libvulkan1 mesa-vulkan-drivers \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- Python ----------------------------------------------------------------
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
 && python -m pip install --upgrade pip

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

# --- Optional runtime knobs -----------------------------------------------
# Determinism (§8.3) — cudnn deterministic + no benchmark. The training scripts
# also set torch.manual_seed / np seeds; these env vars help third-party wheels.
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Non-interactive default: run pytest -q as a smoke on ``docker run``.
# Override with an explicit command to launch a real training run.
CMD ["python", "-m", "pytest", "-q", "tests/", "-k", "not maniskill and not pickcube"]

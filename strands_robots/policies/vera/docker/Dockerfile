# syntax=docker/dockerfile:1.7
#
# VERA policy-server container — runs `vera.server.start_vera_server` with VERA's
# full GPU stack (PyTorch 2.6 / CUDA 12.x + VGGT IDM backbone) isolated from the
# host robots venv. The strands-robots `VeraPolicy` provider connects to this
# container over the websocket protocol (default :8820 PushT / :8800 MimicGen),
# so the host never installs VERA's heavy/conflicting deps.
#
# Build (from the strands-robots repo root):
#   docker build -f strands_robots/policies/vera/docker/Dockerfile \
#       -t strands-vera-server:latest .
#
# Run (PushT, checkpoints mounted from host):
#   docker run --rm --gpus all --ipc=host \
#       -p 8820:8820 -p 8821:8821 \
#       -v /abs/path/vera-ckpts:/ckpts:ro \
#       -e VERA_EMBODIMENT=pusht \
#       strands-vera-server:latest
#
# The NGC PyTorch base ships an optimized torch 2.6 (CUDA 12.x) build. VERA pins
# `torch==2.6.0`; we strip that exact pin at build time so VERA composes with the
# NGC torch (ABI-compatible) instead of pip clobbering it with a stock wheel.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${BASE_IMAGE}

# --- system deps (sim + video IO) -------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Headless MuJoCo / GL: EGL offscreen rendering (no X server in-container).
ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Protect the NGC-optimized torch/torchvision/numpy from being replaced by stock
# wheels: capture their installed versions into a pip constraints file that every
# subsequent install must respect.
RUN python - <<'PY' > /opt/ngc-constraints.txt
import importlib.metadata as m
for pkg in ("torch", "torchvision", "numpy"):
    try:
        print(f"{pkg}=={m.version(pkg)}")
    except m.PackageNotFoundError:
        pass
PY
ENV PIP_CONSTRAINT=/opt/ngc-constraints.txt

# --- VERA (cloned from source, pinned by build arg) -------------------------
ARG VERA_REPO=https://github.com/sizhe-li/VERA.git
ARG VERA_REF=main
WORKDIR /opt
RUN git clone "${VERA_REPO}" vera-src && \
    cd vera-src && git checkout "${VERA_REF}"

WORKDIR /opt/vera-src
# Drop VERA's exact `torch==2.6.0` pin so the NGC torch (2.6.0a, CUDA 12.x,
# ABI-compatible) is kept. The constraints file above keeps torch/vision/numpy
# fixed to the NGC versions for all transitive deps too.
RUN sed -i -E 's/^(\s*)"torch==2\.6\.0".*/\1"torch",/' pyproject.toml && \
    sed -i -E 's/"torchvision[^"]*"/"torchvision"/' pyproject.toml

# Install VERA + the two model-stage extras (idm pulls VGGT git dep) and the sim
# extra used by the PushT/MimicGen env runners. `--no-build-isolation` lets the
# build see the NGC torch. flash-attn is optional (WAN falls back to SDPA).
RUN pip install --no-build-isolation -e ".[idm,video]" && \
    pip install -e ".[eval]"

# Sanity: VERA + torch import and CUDA bindings are intact after install.
RUN python -c "import torch, vera; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- entrypoint -------------------------------------------------------------
COPY strands_robots/policies/vera/docker/entrypoint.sh /usr/local/bin/vera-entrypoint
RUN chmod +x /usr/local/bin/vera-entrypoint

# Checkpoint mount point (read-only bind from host vera-ckpts).
ENV VERA_CKPT_ROOT=/ckpts

# PushT defaults (override via -e for other embodiments).
ENV VERA_EMBODIMENT=pusht \
    VERA_HOST=0.0.0.0 \
    VERA_PORT=8820 \
    VERA_VIS_PORT=8821

EXPOSE 8820 8821 8800 8801

ENTRYPOINT ["/usr/local/bin/vera-entrypoint"]

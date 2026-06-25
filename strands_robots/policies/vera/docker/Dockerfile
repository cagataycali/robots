# syntax=docker/dockerfile:1.7
#
# VERA policy-server container — runs `vera.server.start_vera_server` with VERA's
# full GPU stack (PyTorch 2.6 / CUDA 12.4 + VGGT IDM backbone) isolated from the
# host robots venv. The strands-robots `VeraPolicy` provider connects to this
# container over the websocket protocol (default :8820 PushT / :8800 MimicGen),
# so the host never installs VERA's heavy/conflicting deps.
#
# Build:
#   docker build -f strands_robots/policies/vera/docker/Dockerfile \
#       -t strands-vera-server:latest .
#
# Run (PushT, checkpoints + repo mounted from host):
#   docker run --rm --gpus all \
#       -p 8820:8820 -p 8821:8821 \
#       -v /path/to/vera-ckpts:/ckpts:ro \
#       -e VERA_EMBODIMENT=pusht \
#       strands-vera-server:latest
#
# VERA targets Python 3.11 + torch 2.6 (cu124). The NGC PyTorch base ships a
# matching CUDA 12.4 runtime + cuDNN + flash-attn-friendly toolchain.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
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

# --- VERA (cloned from source, pinned by build arg) -------------------------
# Pin to a commit/tag for reproducibility; override with --build-arg VERA_REF=…
ARG VERA_REPO=https://github.com/sizhe-li/VERA.git
ARG VERA_REF=main
WORKDIR /opt
RUN git clone "${VERA_REPO}" vera-src && \
    cd vera-src && git checkout "${VERA_REF}"

WORKDIR /opt/vera-src
# Install VERA + the two model-stage extras (idm pulls VGGT git dep) + sim extra
# for the PushT/MimicGen env runners used by the example rollouts. torch is
# already provided by the NGC base; let pip respect it.
RUN pip install --no-build-isolation -e ".[idm,video]" && \
    pip install -e ".[eval]"

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

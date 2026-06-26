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
RUN pip install --no-build-isolation -e ".[idm,video]"

# VERA runtime dep fixups (upstream under-pins these; the NGC base ships newer
# versions whose internal APIs VERA's eval-metrics code predates):
#   - colorama: imported by vera.utils.print_utils but not declared.
#   - setuptools<70: VERA imports `pkg_resources.packaging` (removed in 70+).
#   - torch-fidelity: populates FrechetInceptionDistance at torchmetrics.image.
# We KEEP the NGC torchmetrics (ABI-clean with lightning) and instead re-expose
# the pre-1.x lpip symbols VERA's DFoT planner imports (NoTrainLpips /_valid_img)
# via a sitecustomize shim — these are eval-only metrics, not on the inference
# path, so an import-compat shim is sufficient and avoids a version collision.
RUN pip install colorama "setuptools==69.5.1" torch-fidelity

# --- AllTracker (MimicGen/droid motion tracker) -----------------------------
# VERA's WAN planner uses the AllTracker point tracker for eef-delta embodiments
# (mimicgen, droid). It is a sibling-import repo (no packaging): clone it and put
# its PARENT dir on the path via VERA_ALLTRACKER_ROOT so `import alltracker.*`
# resolves. The reference weights auto-download from HF at first use. Its
# requirements re-pin torch/lightning/opencv — install only the few extras it
# actually needs at inference, letting PIP_CONSTRAINT keep the NGC torch.
ARG ALLTRACKER_REPO=https://github.com/aharley/alltracker.git
ARG ALLTRACKER_REF=e7553135e7b361590dbccd10e2b274b024f41cd6
RUN git clone "${ALLTRACKER_REPO}" /opt/third_party/alltracker && \
    git -C /opt/third_party/alltracker checkout "${ALLTRACKER_REF}"
RUN pip install opencv-python einops scikit-image prettytable
# Both paths needed: /opt/third_party for VERA's `import alltracker.utils.*`,
# and the alltracker dir itself (appended LAST so it can't shadow vera/strands
# modules) for alltracker's own flat `import utils.misc` / `import nets.*`.
ENV VERA_ALLTRACKER_ROOT=/opt/third_party
ENV PYTHONPATH=${PYTHONPATH}:/opt/third_party/alltracker

# Install the torchmetrics compat shim as an auto-imported sitecustomize so it
# applies to every interpreter start (the server subprocess included).
COPY strands_robots/policies/vera/docker/sitecustomize_vera.py /opt/vera-shim/sitecustomize.py
ENV PYTHONPATH=/opt/vera-shim:${PYTHONPATH}

# Offline checkpoint resolver + launcher (used for MimicGen/droid whose IDM loads
# by wandb run-id; the resolver maps run-id -> mounted ckpt via provenance.json
# so the server boots without wandb network access). entrypoint.sh runs
# launch_server.py when USE_OFFLINE_RESOLVE=1.
COPY strands_robots/policies/vera/docker/wandb_offline_resolve.py /opt/wandb_offline_resolve.py
COPY strands_robots/policies/vera/docker/launch_server.py /opt/launch_server.py
ENV PYTHONPATH=/opt:${PYTHONPATH}

# PushT sim deps (Wave-1): lightweight, all on PyPI. The MimicGen sim stack
# (robomimic/robosuite/mimicgen) is heavier and robomimic 0.5.0 is git-only AND
# hard-pins huggingface_hub/transformers/diffusers to versions that break the
# NGC base + WAN planner — so install it --no-deps + only its missing runtime deps. So
# it is opt-in via --build-arg INSTALL_MIMICGEN_SIM=1.
RUN pip install "gymnasium==0.29.1" "gym-pusht==0.1.5" "mujoco>=3.5.0" imageio imageio-ffmpeg

ARG INSTALL_MIMICGEN_SIM=0
RUN if [ "${INSTALL_MIMICGEN_SIM}" = "1" ]; then \
        pip install "robosuite==1.4.1" && \
        pip install --no-deps "git+https://github.com/ARISE-Initiative/robomimic.git@v0.5.0" && \
        git clone --depth 1 -b v1.0.0 https://github.com/NVlabs/mimicgen.git /opt/mimicgen-src && \
        pip install --no-deps -e /opt/mimicgen-src && \
        pip install egl_probe tensorboard tensorboardX "mujoco==3.5.0" ; \
    else \
        echo "skipping MimicGen sim stack (set --build-arg INSTALL_MIMICGEN_SIM=1 to include)" ; \
    fi

# Sanity: VERA + torch import and CUDA bindings are intact after install.
RUN python -c "import warnings; warnings.simplefilter('error', UserWarning); \
    import torch, vera, vera.server.start_vera_server; \
    from vera.idm.dfot.dfot_motion_policy import DFoTMotionPolicy; \
    print('sanity OK: torch', torch.__version__, 'cuda', torch.version.cuda, '| DFoT + server import clean')"

# --- entrypoint -------------------------------------------------------------
COPY strands_robots/policies/vera/docker/entrypoint.sh /usr/local/bin/vera-entrypoint
RUN chmod +x /usr/local/bin/vera-entrypoint

# Checkpoint mount point (read-only bind from host vera-ckpts).
ENV VERA_CKPT_ROOT=/ckpts

# PushT defaults (override via -e for other embodiments).
# Default embodiment only — DO NOT set VERA_PORT/VERA_VIS_PORT here: the
# entrypoint derives per-embodiment default ports (pusht 8820/8821,
# mimicgen 8800/8801), and a baked ENV would shadow that ${VAR:-default} logic.
ENV VERA_EMBODIMENT=pusht \
    VERA_HOST=0.0.0.0

EXPOSE 8820 8821 8800 8801

ENTRYPOINT ["/usr/local/bin/vera-entrypoint"]

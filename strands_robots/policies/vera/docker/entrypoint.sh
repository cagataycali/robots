#!/usr/bin/env bash
# VERA policy-server container entrypoint.
#
# Maps a single mounted checkpoint root (VERA_CKPT_ROOT, the layout produced by
# `hf download sizhe-lester-li/VERA --local-dir …`) onto the per-embodiment
# checkpoint env vars that vera.server.start_server_* read, then launches the
# server. Keeps the host-side VeraPolicy provider trivial: it just connects to
# ws://<host>:<port> — no checkpoint path juggling on the client.
#
# MimicGen specifics (see docs/policies/vera.md):
#   * the hosted algo_config.yaml uses OmegaConf env interpolation:
#       text_encoder/vae/clip ckpt_path = ${oc.env:VERA_WAN_CKPT_ROOT}/...
#       dit/flow_decoder      ckpt_path = ${oc.env:VERA_MIMICGEN_CKPT_DIR}/...
#     so we ONLY need to export those two env vars (no yaml patching).
#   * the Jacobian IDM is loaded by wandb run id; offline we resolve it to the
#     locally-mounted ckpt via wandb_offline_resolve.py (provenance.json match).
set -euo pipefail

EMBODIMENT="${VERA_EMBODIMENT:-pusht}"
HOST="${VERA_HOST:-0.0.0.0}"
PORT="${VERA_PORT:-8820}"
VIS_PORT="${VERA_VIS_PORT:-0}"
CKPT_ROOT="${VERA_CKPT_ROOT:-/ckpts}"

echo "[vera-entrypoint] embodiment=${EMBODIMENT} host=${HOST} port=${PORT} vis_port=${VIS_PORT}"
echo "[vera-entrypoint] ckpt_root=${CKPT_ROOT}"

if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "[vera-entrypoint] ERROR: VERA_CKPT_ROOT '${CKPT_ROOT}' not found." >&2
    echo "  Mount your downloaded checkpoints, e.g.:  -v \$PWD/vera-ckpts:/ckpts:ro" >&2
    echo "  Download with:  hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts" >&2
    exit 2
fi

# Per-embodiment checkpoint wiring. Only set a var if the file exists so an
# explicit -e override from `docker run` always wins.
_set_if_exists() {
    # _set_if_exists VAR_NAME /path/to/file_or_dir
    local var="$1" path="$2"
    if [[ -z "${!var:-}" && -e "${path}" ]]; then
        export "${var}=${path}"
        echo "[vera-entrypoint] ${var}=${path}"
    fi
}

# Whether to install the offline wandb->local IDM resolver (mimicgen/omni).
USE_OFFLINE_RESOLVE=0

case "${EMBODIMENT}" in
    pusht)
        # DFoT planner + Jacobian IDM — both LOCAL ckpts with config sidecars.
        # Fully local: no wandb, no WAN base.
        _set_if_exists VERA_PUSHT_PLANNER_CKPT  "${CKPT_ROOT}/pusht-dfot/model.ckpt"
        _set_if_exists VERA_PUSHT_DYNAMICS_CKPT "${CKPT_ROOT}/pusht-idm/model.ckpt"
        PORT="${VERA_PORT:-8820}"
        VIS_PORT="${VERA_VIS_PORT:-8821}"
        ;;
    mimicgen)
        # omni/mimicgen WAN planner (algo_config sidecar, env-interpolated paths) +
        # mimicgen Jacobian IDM (run 37oa162u — the PROVEN stack_d0 stack that pairs
        # with mimicgen-wan-1.3b; NOT the omni-default x21o0cwe).
        _set_if_exists VERA_ALGO_CONFIG       "${CKPT_ROOT}/mimicgen-wan-1.3b/algo_config.yaml"
        _set_if_exists VERA_MIMICGEN_CKPT_DIR "${CKPT_ROOT}/mimicgen-wan-1.3b"
        # The frozen Wan2.1-T2V-1.3B base (text-enc + VAE + CLIP) is a SEPARATE
        # upstream download mounted at /wan. The algo_config reads it via
        # ${oc.env:VERA_WAN_CKPT_ROOT}. Default to /wan unless overridden.
        export VERA_WAN_CKPT_ROOT="${VERA_WAN_CKPT_ROOT:-/wan}"
        echo "[vera-entrypoint] VERA_WAN_CKPT_ROOT=${VERA_WAN_CKPT_ROOT}"
        if [[ ! -e "${VERA_WAN_CKPT_ROOT}/Wan2.1_VAE.pth" ]]; then
            echo "[vera-entrypoint] ERROR: Wan2.1-T2V-1.3B base not found at '${VERA_WAN_CKPT_ROOT}'." >&2
            echo "  MimicGen needs the frozen WAN base (text-enc + VAE + CLIP), a SEPARATE download:" >&2
            echo "    hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B" >&2
            echo "  Then mount it:  -v \$PWD/Wan2.1-T2V-1.3B:/wan:ro   (or set VERA_WAN_CKPT_ROOT)" >&2
            exit 3
        fi
        # IDM run id: prefer the PROVEN mimicgen run unless caller overrides.
        export VERA_DYNAMICS_RUN_ID="${VERA_DYNAMICS_RUN_ID:-37oa162u}"
        echo "[vera-entrypoint] VERA_DYNAMICS_RUN_ID=${VERA_DYNAMICS_RUN_ID}"
        USE_OFFLINE_RESOLVE=1
        PORT="${VERA_PORT:-8800}"
        VIS_PORT="${VERA_VIS_PORT:-8801}"
        ;;
    droid|allegro)
        echo "[vera-entrypoint] NOTE: ${EMBODIMENT} is Wave-2 (checkpoints land upstream later)."
        _set_if_exists VERA_ALGO_CONFIG "${CKPT_ROOT}/omni-wan/algo_config.yaml"
        export VERA_WAN_CKPT_ROOT="${VERA_WAN_CKPT_ROOT:-/wan}"
        USE_OFFLINE_RESOLVE=1
        ;;
    *)
        echo "[vera-entrypoint] ERROR: unknown embodiment '${EMBODIMENT}'." >&2
        echo "  Valid: pusht | mimicgen | droid | allegro" >&2
        exit 2
        ;;
esac

# Offline-safe wandb: never block on the network for IDM resolution. The shim
# resolves run ids to local ckpts via provenance.json; WANDB_MODE=offline stops
# any stray wandb.init/login from reaching out.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # VGGT-1B may still need a first pull

# Assemble server argv (list-style; bash array — no eval/shell-string hacks).
ARGS=(--embodiment "${EMBODIMENT}" --host "${HOST}" --port "${PORT}")
if [[ "${VIS_PORT}" != "0" ]]; then
    ARGS+=(--vis-port "${VIS_PORT}")
fi
if [[ -n "${VERA_ALGO_CONFIG:-}" ]]; then
    ARGS+=(--algo-config "${VERA_ALGO_CONFIG}")
fi
if [[ -n "${VERA_TEXT_PROMPT:-}" ]]; then
    ARGS+=(--text "${VERA_TEXT_PROMPT}")
fi
if [[ -n "${VERA_SAMPLE_STEPS:-}" ]]; then
    ARGS+=(--sample-steps "${VERA_SAMPLE_STEPS}")
fi
if [[ "${VERA_NO_TEACACHE:-0}" == "1" ]]; then
    ARGS+=(--no-teacache)
fi

if [[ "${USE_OFFLINE_RESOLVE}" == "1" ]]; then
    echo "[vera-entrypoint] exec (offline-resolve): python /opt/launch_server.py ${ARGS[*]}"
    # launch_server.py imports wandb_offline_resolve FIRST (patches
    # download_checkpoint -> local), then runs start_vera_server as __main__.
    exec python /opt/launch_server.py "${ARGS[@]}"
else
    echo "[vera-entrypoint] exec: python -m vera.server.start_vera_server ${ARGS[*]}"
    exec python -m vera.server.start_vera_server "${ARGS[@]}"
fi

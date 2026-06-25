#!/usr/bin/env bash
# VERA policy-server container entrypoint.
#
# Maps a single mounted checkpoint root (VERA_CKPT_ROOT, the layout produced by
# `hf download sizhe-lester-li/VERA --local-dir …`) onto the per-embodiment
# checkpoint env vars that vera.server.start_server_* read, then launches the
# server. Keeps the host-side VeraPolicy provider trivial: it just connects to
# ws://<host>:<port> — no checkpoint path juggling on the client.
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
    # _set_if_exists VAR_NAME /path/to/file
    local var="$1" path="$2"
    if [[ -z "${!var:-}" && -f "${path}" ]]; then
        export "${var}=${path}"
        echo "[vera-entrypoint] ${var}=${path}"
    fi
}

case "${EMBODIMENT}" in
    pusht)
        # DFoT planner + Jacobian IDM — both LOCAL ckpts with config sidecars.
        _set_if_exists VERA_PUSHT_PLANNER_CKPT  "${CKPT_ROOT}/pusht-dfot/model.ckpt"
        _set_if_exists VERA_PUSHT_DYNAMICS_CKPT "${CKPT_ROOT}/pusht-idm/model.ckpt"
        PORT="${VERA_PORT:-8820}"
        VIS_PORT="${VERA_VIS_PORT:-8821}"
        ;;
    mimicgen)
        # omni WAN planner (algo_config sidecar) + Jacobian IDM.
        _set_if_exists VERA_ALGO_CONFIG "${CKPT_ROOT}/mimicgen-wan-1.3b/algo_config.yaml"
        # The IDM run id default (x21o0cwe) resolves via VERA's loader; the
        # downloaded idm-mimicgen ckpt is the local artifact for it.
        export VERA_DYNAMICS_RUN_ID="${VERA_DYNAMICS_RUN_ID:-x21o0cwe}"
        PORT="${VERA_PORT:-8800}"
        VIS_PORT="${VERA_VIS_PORT:-8801}"
        ;;
    droid|allegro)
        echo "[vera-entrypoint] NOTE: ${EMBODIMENT} is Wave-2 (checkpoints land upstream later)."
        _set_if_exists VERA_ALGO_CONFIG "${CKPT_ROOT}/omni-wan/algo_config.yaml"
        ;;
    *)
        echo "[vera-entrypoint] ERROR: unknown embodiment '${EMBODIMENT}'." >&2
        echo "  Valid: pusht | mimicgen | droid | allegro" >&2
        exit 2
        ;;
esac

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

echo "[vera-entrypoint] exec: python -m vera.server.start_vera_server ${ARGS[*]}"
exec python -m vera.server.start_vera_server "${ARGS[@]}"

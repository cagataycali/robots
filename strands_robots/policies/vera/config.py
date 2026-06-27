"""Pydantic-free config for the VERA policy provider.

VERA (Video-to-Embodied Robot Action, MIT/CSAIL) is a two-stage video-to-action
policy: an embodiment-agnostic **video planner** (DFoT / WAN) dreams future
frames, and an embodiment-specific **Jacobian IDM** translates the dream into
robot actions. The two stages run inside a single websocket policy server
(``vera.server.start_vera_server``); this provider is a typed client + a managed
server subprocess, mirroring the ``cosmos3`` provider's service pattern.

The config mirrors VERA's server flags 1:1 and is env-overridable so a rollout
can be driven entirely from the environment (CI / fleet) without code changes.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Literal

Embodiment = Literal["pusht", "mimicgen", "allegro", "droid"]
# "mimicgen" is the working, faithful embodiment end-to-end (eef-delta -> IK
# onto a real arm). "pusht" is experimental: its server runs, but VERA's IDM
# "du" action path for pusht is not wired end-to-end upstream (documented in
# VERA's configurations/dataset/pusht.yaml), so it validates the
# provider -> server -> action plumbing rather than producing a solving
# rollout. "allegro"/"droid" are code-present but checkpoint-absent (Wave 2).

# Per-embodiment default ports (policy, viz) — match the VERA examples
# (PushT uses 8820/8821; everything else uses 8800/8801).
# Per-embodiment per-view render width the VERA WAN/DFoT planner expects. The
# server does NOT advertise this (image_resolution is None); it is the client's
# job to send each view at this width (matching VERA's own RemotePolicy
# render_size). pusht: single 252-wide view (PushTImageEnv default); mimicgen:
# 128/view (run_mimicgen_eval default); droid/allegro follow upstream eval.
_DEFAULT_RENDER_WIDTH: dict[str, int] = {
    "pusht": 252,
    "mimicgen": 128,
    "droid": 128,
    "allegro": 128,
}

_DEFAULT_PORTS: dict[str, tuple[int, int]] = {
    "pusht": (8820, 8821),
    "mimicgen": (8800, 8801),
    "allegro": (8802, 8803),
    "droid": (8804, 8805),
}


def _env(*names: str) -> str | None:
    """Return the first set (non-empty) environment variable among ``names``."""
    for n in names:
        v = os.environ.get(n)
        if v is not None and v.strip() != "":
            return v
    return None


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _env_float(name: str) -> float | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


@dataclasses.dataclass
class VeraConfig:
    """Runtime configuration for :class:`VeraPolicy`.

    Every field maps to a VERA server flag / env var. Construction applies
    environment overrides last so callers can pin values in code and still let
    the environment win for deploy-time knobs (ports, checkpoint roots).

    Args:
        embodiment: VERA embodiment — selects the WAN/DFoT planner + Jacobian
            IDM pair and the client-side action adapter.
        host: Policy-server hostname.
        server_port: Policy-server websocket port (per-embodiment default).
        vis_port: MJPEG live-viewer port; ``None`` / ``0`` disables it.
        algo_config: WAN planner ``algo_config.yaml`` path. Point at the omni
            config to swap the planner without retraining the IDM.
        dynamics_run_id: Jacobian/IDM checkpoint id (wandb run id); falls back
            to the embodiment's in-tree default when unset.
        text_prompt: Optional text conditioning for the video planner.
        ckpt_root: Root of the downloaded VERA checkpoints
            (``hf download sizhe-lester-li/VERA --local-dir …``). Exported to
            ``VERA_CKPT_ROOT`` for the server subprocess.
        sample_steps: WAN denoise steps (deploy default is 10; ``None`` uses the
            planner yaml's value).
        n_action_steps: Deploy chunk size (actions executed per infer).
        tracker_backend: IDM point tracker backend override.
        motion_plan_scale: IDM motion-plan scale override (live-tunable).
        teacache: Enable the near-lossless DiT teacache speedup (default True).
        teacache_thresh: teacache rel_l1 threshold (>0.15 hits a quality cliff).
        auto_launch_server: Launch + manage the server subprocess on first use.
        server_ready_timeout: Seconds to wait for the server websocket to come
            up before raising (WAN model load can be slow).
        python_executable: Interpreter used to launch the server subprocess
            (defaults to the current interpreter / ``VERA_PYTHON``).
    """

    embodiment: Embodiment = "pusht"
    host: str = "127.0.0.1"
    server_port: int | None = None
    vis_port: int | None = None
    render_width: int | None = None  # per-view width sent to the server (per-embodiment default)
    algo_config: Path | None = None
    dynamics_run_id: str | None = None
    text_prompt: str | None = None
    ckpt_root: Path | None = None
    wan_ckpt_root: Path | None = None  # frozen Wan2.1-T2V-1.3B base (mimicgen/omni); env VERA_WAN_CKPT_ROOT
    sample_steps: int | None = None
    n_action_steps: int | None = None
    tracker_backend: str | None = None
    motion_plan_scale: float | None = None
    teacache: bool = True
    teacache_thresh: float = 0.10
    auto_launch_server: bool = True
    server_ready_timeout: float = 600.0
    python_executable: str | None = None
    # --- server launch mode -------------------------------------------------
    server_mode: str = "subprocess"  # "subprocess" | "docker"
    docker_image: str = "strands-vera-server:latest"
    docker_container_name: str | None = None  # default: vera-server-<embodiment>
    docker_gpus: str = "all"  # --gpus value (e.g. "all" or "device=0")
    docker_extra_args: list[str] | None = None  # extra `docker run` args (list, no shell)

    def __post_init__(self) -> None:
        # Apply per-embodiment port defaults when not explicitly set.
        default_policy, default_vis = _DEFAULT_PORTS.get(self.embodiment, (8800, 8801))
        if self.server_port is None:
            self.server_port = _env_int("VERA_SERVER_PORT") or default_policy
        if self.vis_port is None:
            env_vis = _env_int("VERA_VIS_PORT")
            self.vis_port = env_vis if env_vis is not None else default_vis

        if self.render_width is None:
            self.render_width = _env_int("VERA_RENDER_WIDTH") or _DEFAULT_RENDER_WIDTH.get(self.embodiment, 128)

        # Environment overrides (deploy/CI win over code defaults).
        if self.algo_config is None:
            ac = _env("VERA_ALGO_CONFIG")
            self.algo_config = Path(ac) if ac else None
        if self.dynamics_run_id is None:
            self.dynamics_run_id = _env("VERA_DYNAMICS_RUN_ID")
        if self.text_prompt is None:
            self.text_prompt = _env("VERA_TEXT_PROMPT")
        if self.ckpt_root is None:
            cr = _env(
                "VERA_CKPT_ROOT",
                f"VERA_{self.embodiment.upper()}_CKPT_ROOT",
                "VERA_MIMICGEN_CKPT_DIR" if self.embodiment == "mimicgen" else "",
            )
            self.ckpt_root = Path(cr) if cr else None
        if self.wan_ckpt_root is None:
            wr = _env("VERA_WAN_CKPT_ROOT")
            self.wan_ckpt_root = Path(wr) if wr else None
        if self.sample_steps is None:
            self.sample_steps = _env_int("VERA_SAMPLE_STEPS")
        if self.n_action_steps is None:
            self.n_action_steps = _env_int("VERA_N_ACTION_STEPS")
        if self.tracker_backend is None:
            self.tracker_backend = _env("VERA_TRACKER_BACKEND")
        if self.motion_plan_scale is None:
            self.motion_plan_scale = _env_float("VERA_MOTION_PLAN_SCALE")
        if self.python_executable is None:
            self.python_executable = _env("VERA_PYTHON")
        _sm = _env("VERA_SERVER_MODE")
        if _sm:
            self.server_mode = _sm
        _di = _env("VERA_DOCKER_IMAGE")
        if _di:
            self.docker_image = _di
        _dg = _env("VERA_DOCKER_GPUS")
        if _dg:
            self.docker_gpus = _dg
        if self.docker_container_name is None:
            self.docker_container_name = _env("VERA_DOCKER_CONTAINER") or f"vera-server-{self.embodiment}"

        # Coerce string paths to Path (defensive — callers may pass str).
        if self.algo_config is not None and not isinstance(self.algo_config, Path):
            self.algo_config = Path(self.algo_config)
        if self.ckpt_root is not None and not isinstance(self.ckpt_root, Path):
            self.ckpt_root = Path(self.ckpt_root)
        if self.wan_ckpt_root is not None and not isinstance(self.wan_ckpt_root, Path):
            self.wan_ckpt_root = Path(self.wan_ckpt_root)

    @property
    def server_uri(self) -> str:
        return f"ws://{self.host}:{self.server_port}"

    def server_env(self) -> dict[str, str]:
        """Environment overlay for the server subprocess (checkpoints, tracker)."""
        env: dict[str, str] = {}
        if self.ckpt_root is not None:
            env["VERA_CKPT_ROOT"] = str(self.ckpt_root)
        if self.wan_ckpt_root is not None:
            env["VERA_WAN_CKPT_ROOT"] = str(self.wan_ckpt_root)
        if self.tracker_backend is not None:
            env["VERA_TRACKER_BACKEND"] = self.tracker_backend
        if self.dynamics_run_id is not None:
            env["VERA_DYNAMICS_RUN_ID"] = str(self.dynamics_run_id)
        return env

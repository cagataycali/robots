"""``/api/record``, ``/api/collect``, ``/api/replay`` and ``/api/datasets/labels`` - the record screen.

The session surface (open an arm pair, start/stop/redo/discard episodes, thumbnails, close) lives
in :mod:`strands_robots.dashboard.record_api` and is mounted here late-bound: the controller is
read off ``request.app.state.record`` per request, so the router exists before the app does and
a coordinator that swaps the bridge or the device manager does not pin a stale one.

Every path a client names is contained to the dataset home (``$HF_LEROBOT_HOME``, the same
place :mod:`strands_robots.dataset_source` writes to). A path outside it is refused with 400 and
one fixed body whether or not it exists; the dashboard is a network service and the filesystem
outside the dataset home is not its business to describe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request

from strands_robots.dashboard import access, record_api
from strands_robots.dashboard.dataset_check import OUTSIDE_DATASET_HOME
from strands_robots.dashboard.log_redaction import one_line
from strands_robots.utils import refusal_repr

logger = logging.getLogger(__name__)

router = APIRouter(tags=["record"], dependencies=[Depends(access.require_session)])


def dataset_home() -> Path:
    """Where datasets live on this machine - lerobot's own constant when it is installed."""
    from strands_robots.dataset_source import _lerobot_home

    return _lerobot_home().expanduser().resolve()


def contained_path(raw: Any) -> Path:
    """Resolve a client-named path and refuse it unless it sits under :func:`dataset_home`.

    The path is folded first (``~`` expanded, made absolute, symlinks followed and ``..``
    collapsed with ``realpath`` and ``normpath``) and the result must be the home or start with
    it, so a link out of the home does not count as inside it. The refusal carries no trace of
    the target.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(422, "path required")
    home = str(dataset_home())
    try:
        real = os.path.normpath(os.path.realpath(os.path.expanduser(raw.strip())))
    except (OSError, RuntimeError):
        raise HTTPException(400, OUTSIDE_DATASET_HOME) from None
    # Two single tests rather than one ``==`` or ``startswith``: the folded path must
    # begin with the home, and a longer path must continue with a separator, so a
    # sibling such as ``<home>2`` is refused. The pair admits exactly what the one
    # compound test did; a path-injection checker reads a lone ``startswith`` as the
    # barrier and does not read a disjunction as one.
    if not real.startswith(home):
        raise HTTPException(400, OUTSIDE_DATASET_HOME)
    if real != home and not real.startswith(home + os.sep):
        raise HTTPException(400, OUTSIDE_DATASET_HOME)
    return Path(real)


# ---------------------------------------------------------------- controller


def controller(request: Request) -> record_api.RecordController:
    """The app's controller, built on first use when :func:`attach` could not build it eagerly."""
    state = request.app.state
    existing = getattr(state, "record", None)
    if existing is None:
        existing = _build(request.app)
        state.record = existing
    return cast("record_api.RecordController", existing)


def _build(app: FastAPI) -> record_api.RecordController:
    devices = getattr(app.state, "devices", None)
    bridge = getattr(app.state, "bridge", None)
    return record_api.RecordController(devices, bridge=bridge)


def _activity(request: Request) -> Callable[..., None] | None:
    # Late-bound on purpose: capturing a bound method would pin the router to one bridge instance.
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None:
        return None
    return cast("Callable[..., None] | None", getattr(bridge, "record_activity", None))


def attach(app: FastAPI) -> None:
    """Set ``app.state.record``; when the device manager is not attached yet, finish at startup."""
    if getattr(app.state, "devices", None) is not None:
        app.state.record = _build(app)
        return

    async def _late() -> None:
        # The devices lane may attach after this one; rebuild once everything has run.
        if getattr(app.state, "record", None) is None or app.state.record._devices is None:
            app.state.record = _build(app)

    hooks = getattr(app.state, "startup_hooks", None)
    if isinstance(hooks, list):
        hooks.append(_late)
    else:
        app.state.record = _build(app)


router.include_router(record_api.build_router(controller, _activity, late_bound=True))


# ------------------------------------------------------------ one-shot runs


def _devices_or_503(request: Request) -> Any:
    devices = getattr(request.app.state, "devices", None)
    if devices is None:
        raise HTTPException(503, "no device manager is attached to this dashboard - nothing can run a one-shot sim")
    return devices


#: The ``policy_config`` keys a collect may forward: the ones the mesh dispatcher forwards from a
#: validated ``execute`` (its ``extra`` tuple) plus the two host knobs the wire grades. Every one
#: has a validator in :func:`strands_robots.mesh.security.validate_command`; a key outside this
#: set is a Policy constructor kwarg the wire would never carry, so the route refuses it by name
#: instead of dropping it.
WIRE_POLICY_CONFIG_KEYS: tuple[str, ...] = (
    "model_path",
    "server_address",
    "policy_type",
    "pretrained_name_or_path",
    "policy_host",
    "policy_port",
)


def contained_policy_request(policy_provider: Any, policy_config: Any) -> tuple[str, dict[str, Any] | None]:
    """The provider and config a collect may hand to ``run_policy``: what the wire would accept.

    The page's ``policy_provider`` and ``policy_config`` used to reach ``create_policy`` in the
    child process untouched, so a provider off the allowlist, a ``model_path`` anywhere on the
    disk or a ``server_address`` to any host went through a route the mesh would have refused.
    The same validator now runs here, ``model_path`` is additionally contained under the
    checkpoint homes, and only validated keys come back. Raises :class:`HTTPException` 422
    (400 for a path outside its home) with the validator's own sentence.
    """
    from strands_robots.dashboard import training
    from strands_robots.mesh import security

    if policy_config is None:
        config: dict[str, Any] = {}
    elif isinstance(policy_config, dict):
        config = dict(policy_config)
    else:
        raise HTTPException(422, "policy_config must be an object")
    unknown = sorted(str(key) for key in config if key not in WIRE_POLICY_CONFIG_KEYS)
    if unknown:
        raise HTTPException(
            422,
            f"policy_config keys not carried by the wire: {', '.join(refusal_repr(k) for k in unknown)} "
            f"(allowed: {', '.join(WIRE_POLICY_CONFIG_KEYS)})",
        )
    if isinstance(config.get("model_path"), str):
        try:
            config["model_path"] = str(training.contain_checkpoint_path(config["model_path"]))
        except training.PathOutside as exc:
            raise HTTPException(400, exc.refusal()) from exc
    cmd = {"action": "execute", "policy_provider": policy_provider, "instruction": "collect", **config}
    try:
        validated = security.validate_command(cmd)
    except security.ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    forwarded = {key: validated[key] for key in WIRE_POLICY_CONFIG_KEYS if key in validated and key in config}
    return str(validated["policy_provider"]), forwarded or None


@router.post("/api/collect")
async def collect_episodes(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """Collect a policy-driven dataset in a one-shot mesh sim. run_policy drives exactly n_episodes
    rollouts with per-episode parquet boundaries and reports parquet-truth counts.
    """
    dataset_root = str(contained_path(body.get("dataset_root")))
    policy_provider, policy_config = contained_policy_request(
        body.get("policy_provider", "mock"), body.get("policy_config")
    )
    devices = _devices_or_503(request)
    # Remember the root so /api/training/datasets discovers the result even outside the default
    # scan paths. The training lane ships the memory; without it the collection still runs.
    try:
        from strands_robots.dashboard import training as _training

        _training.remember_dataset_root(dataset_root)
    except ImportError:
        logger.debug(
            "[record] training module absent; %s will not be remembered for the picker", one_line(dataset_root)
        )
    result = await asyncio.to_thread(
        lambda: devices.collect(
            dataset_root=dataset_root,
            dataset_repo_id=body.get("dataset_repo_id", "local/collected"),
            robot_name=body.get("robot_name") or "so101",
            policy_provider=policy_provider,
            policy_config=policy_config,
            instruction=body.get("instruction", ""),
            n_episodes=int(body.get("n_episodes", 5)),
            duration=float(body.get("duration", 10.0)),
            fps=int(body.get("fps", 30)),
        )
    )
    # Two recorders writing one dataset directory interleave episodes into each other's files.
    # 409 names the session already holding it.
    if result.get("already_running"):
        raise HTTPException(409, result)
    return cast("dict[str, Any]", result)


@router.post("/api/replay")
async def replay_episode(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """Replay a recorded LeRobotDataset episode in a one-shot mesh sim."""
    repo_id = (body.get("repo_id") or "").strip()
    if not repo_id:
        raise HTTPException(422, "repo_id required")
    root = body.get("root")
    if root is not None:
        # Containment first: validate_replay would otherwise say whether the directory exists.
        root = str(contained_path(root))
    devices = _devices_or_503(request)
    from strands_robots.dashboard.device_manager import validate_replay

    bad = validate_replay(repo_id, body.get("episode", 0), root, body.get("speed", 1.0))
    if bad:
        raise HTTPException(422, bad)
    result = await asyncio.to_thread(
        devices.replay,
        repo_id,
        int(body.get("episode", 0)),
        root,
        float(body.get("speed", 1.0)),
        body.get("robot_name") or "so101",
    )
    # 409, not an error-shaped 200: a second replay of the same episode is a conflict with
    # something that already exists, and the response names the peer already showing it.
    if result.get("already_running"):
        raise HTTPException(409, result)
    return cast("dict[str, Any]", result)


# ------------------------------------------------------------------ labels


@router.get("/api/datasets/labels")
async def dataset_labels(root: str | None = None, path: str | None = None) -> dict[str, Any]:
    """The episode label sidecar of one local dataset, as the label view renders it."""
    from strands_robots import episode_labels as _labels
    from strands_robots.dashboard.episode_label_view import label_view

    target = contained_path(root if root is not None else path)
    if not target.is_dir():
        raise HTTPException(404, "no dataset directory there")

    document: dict[str, Any] | None = None
    sidecar_error: str | None = None
    if _labels.labels_path(target).exists():
        try:
            document = _labels.read_labels(target)
        except Exception as e:  # noqa: BLE001 - a corrupt sidecar must not read as "no labels yet"
            # the parser's own words go to the log; the browser learns the kind of failure
            logger.warning("episode label sidecar could not be read: %r", e)
            sidecar_error = type(e).__name__

    total: int | None = None
    try:
        total = json.loads((target / "meta" / "info.json").read_text(encoding="utf-8")).get("total_episodes")
    except Exception:  # noqa: BLE001 - a dataset mid-recording has no readable info.json yet
        pass

    return label_view(document, total_episodes=total, sidecar_error=sidecar_error)

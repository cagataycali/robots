#!/usr/bin/env python3
"""
Universal Robot Control with Policy Abstraction for Any VLA Provider

This module provides a clean robot interface that works with any LeRobot-compatible
robot and any VLA provider through the Policy abstraction.

Features:
- Async robot task execution with real-time status reporting
- Non-blocking operations - robot moves while tool returns status
- Stop functionality to interrupt running tasks
- Connection state management with proper error handling
- Policy abstraction for any VLA provider

Operator approval: this class is the ``mode="real"`` half of
:func:`strands_robots.Robot`, so every ``execute`` or ``start`` the agent tool
dispatches drives real actuators. Both stop for a human BEFORE the rollout is
dispatched, through the same decision path the ROS transports and the serial
tool use (:func:`~strands_robots.tools._command_gate.gate_motion`):
``STRANDS_ROBOT_COMMAND_ALLOW`` (comma-separated ``execute``/``start``, or
``*``) pre-approves, ``BYPASS_TOOL_CONSENT=true`` lifts the gate with a WARNING,
otherwise the operator is asked through the agent's interrupt and, with no
agent reachable, the call is refused and nothing is dispatched. The dashboard's
:class:`~strands_robots.dashboard.agent_hitl.MotionInterruptHook` may already
have asked; a grant it deposited for this exact call is spent instead of asking
twice. ``status`` and ``stop`` are never gated - stopping must not get harder -
and the simulation tool is a different class that never touches hardware.
Before this gate the README's first path to metal, ``Agent(tools=[Robot("so100",
mode="real")])``, dispatched unasked while the same robot commanded through
``robot_mesh`` was gated (F-011, CWE-862).
"""

from __future__ import annotations

import asyncio
import dataclasses
import difflib
import functools
import importlib
import logging
import math
import os
import pkgutil
import shutil
import threading
import time
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from strands.interrupt import InterruptException
from strands.tools.tools import AgentTool
from strands.types._events import ToolInterruptEvent, ToolResultEvent
from strands.types.tools import ToolContext, ToolResult, ToolSpec, ToolUse

from strands_robots import hardware_observe
from strands_robots._serial_discovery import describe_serial_candidates, scan_serial_devices
from strands_robots.bus_access import bus_lock, read_observation, write_action
from strands_robots.policies.base import instruction_not_read_notice, provider_policy_class
from strands_robots.registry.policies import policy_requires_error
from strands_robots.ros_telemetry import ROS2_SYSTEM_INSTALL_HINT
from strands_robots.teleop_mixin import TeleopMixin, _stop_reported_stopped
from strands_robots.tools._command_gate import gate_motion
from strands_robots.utils import (
    boolean_flag_error,
    camera_token_error,
    dds_domain_id_error,
    positive_count_error,
    positive_finite_number_error,
    refusal_repr,
    refusal_str,
    require_optional,
    tcp_port_error,
    teleoperator_contract_error,
)

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.robot import Robot as LeRobotRobot

    from .policies import Policy

logger = logging.getLogger(__name__)

# The agent-tool actions that dispatch a rollout to real actuators. ``status``
# and ``stop`` only read or halt, so they are never gated.
MOTION_ACTIONS = frozenset({"execute", "start"})

# Every action the tool publishes, in the order the schema lists them. The
# ``action`` enum and the unknown-action refusal are two readings of one
# vocabulary, and they were kept as two literals: the observe verbs were added
# to the enum while the refusal went on naming only the four motion ones, so an
# agent that misspelled ``get_state`` was told the valid actions were "execute,
# start, status, stop" and could conclude that reading the arm was not offered.
# Both now read this, so a verb cannot be published without being named.
_PUBLISHED_ACTIONS = (
    "get_state",
    "get_robot_state",
    "list_cameras",
    "render",
    "execute",
    "start",
    "status",
    "stop",
)

# Pre-approve motion actions by name (comma-separated, ``*`` for all) for
# headless runs. Read by the shared gate, which also honours BYPASS_TOOL_CONSENT.
COMMAND_ALLOW_ENV = "STRANDS_ROBOT_COMMAND_ALLOW"


# Remedy for a missing ``rclpy`` when the caller asked for the rclpy transport.
# The shared hint names the step that supplies rclpy; this adds the alternative
# only a Robot can take, because ``ros2_transport`` is the caller's choice: the
# pure-RTPS bridge publishes the same topics (both transports share the wire
# contract in ``RosTelemetryBase``) over cyclonedds, a pip wheel on macOS,
# Windows and Linux x86_64.  No cyclonedds release publishes a Linux aarch64
# wheel, so there the extra builds the sdist against a Cyclone DDS C install and
# this alternative carries a condition: a caller already blocked on rclpy must be
# told that, not sent into a second failing install.
_RCLPY_TRANSPORT_INSTALL_HINT = (
    f"{ROS2_SYSTEM_INSTALL_HINT}\n"
    "Or select the pure-RTPS transport, which publishes the same topics and "
    "needs no sourced distro:\n"
    "  pip install 'strands-robots[ros2]'\n"
    "  Robot(..., ros2_bridge=True, ros2_transport='rtps')\n"
    "On Linux aarch64 (Jetson) that extra has no wheel: install Cyclone DDS C "
    "first and set CYCLONEDDS_HOME, see "
    "docs/rtps-integration.md#linux-aarch64-jetson."
)


# ---------------------------------------------------------------------------
# Lazy lerobot RobotConfig registration helper.
# ---------------------------------------------------------------------------
#
# lerobot's robot drivers register themselves with ``RobotConfig`` (a
# draccus ``ChoiceRegistry``) via ``@RobotConfig.register_subclass(...)``
# at module import time. Because ``lerobot.robots.__init__`` does not
# eagerly import every subpackage, the registry is empty until something
# triggers the import. ``_create_minimal_config`` calls this helper once
# per process to populate it. ``@functools.cache`` makes the second call
# a dict lookup, so the per-Robot() overhead amortises to ~0.


# Cross-robot kwargs forwarded to lerobot config constructors.  Exposed
# as a module-level constant so tests can import it (single source of
# truth).
#
# Post-R5, the dual-gate semantics are:
#   - kwargs in this allowlist BUT NOT on the resolved target dataclass
#     are silently dropped (cross-robot polymorphism: passing ``kp=...``
#     to so101 doesn't blow up just because ``kp`` is a unitree_g1
#     kwarg).
#   - kwargs declared on the resolved target dataclass are forwarded
#     automatically, regardless of whether they appear in this list
#     (so a future lerobot field like ``wifi_ssid`` Just Works without
#     a strands_robots release).
#   - kwargs unknown to BOTH are rejected at config-build time
#     (typos like ``prot=``, kwargs from another subsystem entirely).
#
# So this allowlist's job is narrow: it's the set of kwargs whose
# silent-drop on a non-matching robot we tolerate as a documented
# polymorphism win.  It is not a forwarding gate -- ``valid_fields``
# is.
_FORWARDABLE_KWARGS = (
    "port",  # serial robots (so100/so101, koch, openarm, ...)
    "robot_ip",  # network robots (unitree_g1, lekiwi, reachy2, ...)
    "kp",
    "kd",  # PD-controlled robots (g1, h1, ...)
    "default_positions",  # humanoids
    "control_dt",  # humanoids / locomotion
    "is_simulation",  # robots that share a sim/real driver
    "gravity_compensation",  # arms with IK comp
    "controller",  # locomotion controller selection
    "calibration_dir",
    "mock",
    "use_degrees",
    "max_relative_target",
    "disable_torque_on_disconnect",
)


# ---------------------------------------------------------------------------
# Per-camera config construction.
# ---------------------------------------------------------------------------
#
# ``Robot(..., cameras={"front": {...}})`` describes each camera with a
# free-form dict. That dict is a serialized lerobot ``CameraConfig``: ``type``
# is draccus' own choice discriminator (``CameraConfig.type`` returns
# ``get_choice_name(cls)``) and every other key is a field of the dataclass that
# discriminator selects. So both halves of the vocabulary are derived, never
# hand-picked:
#
#   - the set of accepted ``type`` values is ``CameraConfig.get_known_choices()``
#     -- the same draccus ``ChoiceRegistry`` lookup ``_create_minimal_config``
#     already uses for ``robot_type``, and the one ``make_cameras_from_configs``
#     dispatches on. A camera backend lerobot ships or a vendor plugin registers
#     is therefore attachable the day it lands, with no mapping to maintain here.
#   - the set of accepted option keys is ``dataclasses.fields()`` of the
#     resolved class. A hand-picked list leaves every field it forgets
#     unreachable (no caller can set it at all) and silently discards every key
#     it does not recognise, so a typo like ``heigth=1080`` reports success
#     having configured the default resolution.
#
# ``strands_robots`` supplies its own defaults for the three fields lerobot
# leaves as ``None`` (meaning "whatever the device negotiates") so an
# unconfigured camera has a predictable, documented stream. Those three are
# declared on the ``CameraConfig`` base, so they are fields of every registered
# choice and the defaults apply to a RealSense exactly as they do to a webcam.
# Every other field keeps lerobot's own default.
#
# Invariant: every key here must be a field lerobot still declares. If one is
# renamed upstream the stale key reaches the config constructor and fails
# loudly for every camera rather than being silently dropped -- and the test
# suite asserts the containment directly, so the drift is caught before a
# release rather than at an operator's ``Robot()`` call.
_CAMERA_STREAM_DEFAULTS: dict[str, Any] = {"fps": 30, "width": 640, "height": 480}

# ``type`` selects which camera backend to build. It is consumed by the
# registry lookup below, not forwarded to the config dataclass.
_CAMERA_TYPE_KEY = "type"

# Where a network driver's port is reached. A lerobot driver config spells the
# host it talks to differently per family - ``ip_address`` (Reachy 2),
# ``remote_ip`` (LeKiwi client), ``robot_ip`` (Unitree G1), ``host`` - and
# ``Robot._device_facts`` names it beside a port that is not a device path, so
# a bare TCP port number is not the whole answer an agent gets.
_ADDRESS_FIELDS = ("ip_address", "remote_ip", "robot_ip", "host")


@functools.cache
def _ensure_lerobot_cameras_registered() -> None:
    """Import every camera backend subpackage so CameraConfig is populated.

    The mirror of :func:`_ensure_lerobot_robots_registered`, and for the same
    reason: each backend registers its config via
    ``@CameraConfig.register_subclass`` at module-import time, but
    ``lerobot.cameras.__init__`` deliberately does not import them -- it says so
    in a comment, to avoid pulling backend-specific dependencies into every
    ``import lerobot``. Until they are imported ``CameraConfig`` has *no*
    registered choices at all, so a registry lookup that skips this step reports
    every camera type as unknown.

    Walks ``lerobot.cameras`` with ``pkgutil`` so a backend lerobot adds in a
    future release needs no change here, then registers third-party
    ``lerobot_camera_*`` distributions through lerobot's own plugin loader.

    Idempotent via ``@functools.cache`` -- the first call walks the tree,
    subsequent calls are dict lookups.
    """
    try:
        import lerobot.cameras as _lr_cameras
    except ImportError as exc:
        # Mirrors the robot walk: lerobot wholly absent is expected on
        # sim-only hosts (debug), while lerobot present but
        # ``lerobot.cameras`` unimportable is a partial install worth a
        # warning. Either way the caller gets a clean "Unsupported camera
        # type" naming the choices that did register.
        try:
            import lerobot  # noqa: F401  (probe-only)
        except ImportError:
            logger.debug("lerobot not installed: %s", exc)
        else:
            logger.warning(
                "lerobot is installed but lerobot.cameras is not importable (partial install?): %s",
                exc,
            )
        return

    for _, sub_name, is_pkg in pkgutil.iter_modules(_lr_cameras.__path__):
        if not is_pkg:
            continue
        full_name = f"{_lr_cameras.__name__}.{sub_name}"
        try:
            importlib.import_module(full_name)
        except (ImportError, OSError) as exc:
            # A backend whose SDK is absent (``pyrealsense2``, ``reachy2_sdk``)
            # or whose ``__init__`` probes the OS. It simply does not appear in
            # the choice registry, which is the correct outcome: naming it later
            # raises "Unsupported camera type" listing what is available.
            # ``(ImportError, OSError)`` is the canonical narrow pair for a
            # hardware-probing import per AGENTS.md > Review Learnings (#86).
            logger.debug("[hardware_robot] skip %s: %s", full_name, exc)

    _ensure_lerobot_plugins_registered()


def _resolve_camera_config_class(camera_name: str, cam_type: Any) -> type:
    """Resolve a camera ``type`` to the lerobot config class it names.

    Args:
        camera_name: The key this camera was registered under, named in the
            refusal so a multi-camera rig reports which entry is at fault.
        cam_type: The requested ``type`` value, as the caller spelled it.

    Returns:
        The registered ``CameraConfig`` subclass for ``cam_type``.

    Raises:
        ValueError: If ``cam_type`` is not a registered choice. The refusal
            lists every choice that did register and, when the spelling is
            close to one of them, names it -- lerobot registers Intel RealSense
            as ``intelrealsense``, so the obvious guess ``realsense`` is a
            dead end without the suggestion.
    """
    from lerobot.cameras.configs import CameraConfig

    _ensure_lerobot_cameras_registered()
    try:
        return cast(type, CameraConfig.get_choice_class(cam_type))
    except (KeyError, TypeError):
        # KeyError: not a registered choice. TypeError: an unhashable value
        # (a list, a dict) can never be a registry key, so it is the same
        # refusal rather than a traceback out of the registry's dict lookup.
        known = sorted(CameraConfig.get_known_choices())
        close = difflib.get_close_matches(str(cam_type), known, n=1, cutoff=0.7)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        # ``from None`` -- the registry's KeyError is an internal detail of
        # draccus; suppress the chained traceback for a cleaner error.
        raise ValueError(
            f"Unsupported camera type for camera {camera_name!r}: {cam_type!r}.{hint} "
            f"Known lerobot camera types: {known}."
        ) from None


def _camera_option_vocabulary(camera_name: str, config: Mapping[str, Any]) -> tuple[type, dict[str, Any]]:
    """Resolve the config class one camera entry names and the options it may state.

    The one owner of the camera option vocabulary. ``type`` selects the class
    through lerobot's ``CameraConfig`` choice registry, and the options an entry
    may then name are that class's declared dataclass fields - so a backend
    lerobot adds, or a field it renames, is admitted here by construction rather
    than by a list kept in step by hand. Every surface that accepts the
    serialized ``cameras`` shape reads it from here: the ``Robot`` factory
    constructs the config, and ``lerobot_teleoperate`` renders the same entry
    into the ``--robot.cameras`` argv of a detached subprocess, where an option
    the class does not declare would be refused minutes later in that process's
    log rather than here.

    Args:
        camera_name: The key this camera was registered under, named in every
            refusal so a multi-camera rig reports which entry is at fault.
        config: The per-camera options, already known to be a mapping.

    Returns:
        The resolved ``CameraConfig`` subclass and its declared fields by name.

    Raises:
        ValueError: If ``type`` is not a registered camera backend, or the entry
            names an option the resolved class does not declare. An unknown
            option is refused rather than dropped per AGENTS.md > Review
            Learnings (#86): a silently discarded option reports success while
            the camera streams at the default. The suggestion is drawn from the
            resolved class's own fields: an ``index_or_path`` sent to a
            RealSense is a real mistake, and pointing at
            ``serial_number_or_name`` is what makes it fixable.
    """
    ConfigClass = _resolve_camera_config_class(camera_name, config.get(_CAMERA_TYPE_KEY, "opencv"))
    fields = {f.name: f for f in dataclasses.fields(ConfigClass)}
    accepted = sorted(set(fields) | {_CAMERA_TYPE_KEY})

    unknown = sorted(set(config) - set(fields) - {_CAMERA_TYPE_KEY}, key=repr)
    if unknown:
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(str(key), accepted, n=1, cutoff=0.7)
            if close:
                hints.append(f"{key!r} -> {close[0]!r}")
        hint = f" Did you mean: {', '.join(hints)}?" if hints else ""
        raise ValueError(
            f"Unknown option(s) for camera {camera_name!r}: {unknown}.{hint} "
            f"{ConfigClass.__name__} accepts: {accepted} (where {_CAMERA_TYPE_KEY!r} selects "
            f"the camera backend). (If this is a typo, fix it.)"
        )
    return ConfigClass, fields


def _requires_a_caller_value(field: dataclasses.Field) -> bool:
    """Answer whether a config dataclass field is one the caller must supply.

    A field is the caller's to supply only if the constructor accepts it at all.
    ``dataclasses.field(init=False)`` names a value the class derives for itself
    -- lerobot's ``UnitreeG1Config.sim_env`` is assigned in ``__post_init__`` --
    so it is absent from ``__init__`` and carries no default either. Reading the
    absent default alone counts such a field as required, which refuses a call
    the dataclass would have accepted, and the remedy that refusal prints
    (``sim_env=...``) then raises ``TypeError: __init__() got an unexpected
    keyword argument``: a dead end whichever way the caller turns.

    The one owner of the rule, so the robot-config and camera-option scans
    cannot come to disagree about what a caller can be asked for.
    ``strands_robots.training.lerobot`` reads ``field.init`` for the same
    question about training-config kwargs.
    """
    return field.init and field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING


def _build_camera_config(camera_name: str, config: Any) -> Any:
    """Build the lerobot camera config for one entry of a ``cameras`` dict.

    Args:
        camera_name: The key this camera was registered under. Named in every
            error so a multi-camera rig reports which entry is at fault.
        config: The per-camera options. ``type`` selects the camera backend
            from lerobot's ``CameraConfig`` choice registry (default
            ``opencv``); every other accepted key is a declared field of the
            config class that choice resolves to.

    Returns:
        An instance of the ``CameraConfig`` subclass the requested ``type``
        names, ready for ``lerobot.cameras.make_cameras_from_configs``.

    Raises:
        ValueError: If ``camera_name`` is not a bare token
            (:func:`~strands_robots.utils.camera_token_error`), or if ``config``
            is not a mapping, names a camera ``type`` lerobot does not register,
            carries a key that is not a declared field of the resolved class,
            omits a field that has no default, or holds a value lerobot's own
            config validation refuses. An unknown key is refused rather than
            dropped per AGENTS.md > Review Learnings (#86): a silently discarded
            option reports success while the camera streams at the default.
    """
    # The name is graded before the options because it is what every consumer
    # keys this camera's frames by -- a mesh topic level, an S3 object key, a
    # dataset feature key -- so no option is worth checking under a name none of
    # them can carry. ``lerobot_teleoperate`` holds its ``robot_cameras`` to the
    # same rule at the same point, through the same owner.
    if (name_err := camera_token_error("Robot(cameras=...)", "camera name", camera_name)) is not None:
        raise ValueError(name_err)
    if not isinstance(config, Mapping):
        raise ValueError(
            f"Camera {camera_name!r} config must be a mapping of option name to value, "
            f"got {type(config).__name__}: {config!r}."
        )

    ConfigClass, fields = _camera_option_vocabulary(camera_name, config)
    class_name = ConfigClass.__name__
    accepted = sorted(set(fields) | {_CAMERA_TYPE_KEY})

    missing = sorted(
        name
        for name, field in fields.items()
        if name not in config and name not in _CAMERA_STREAM_DEFAULTS and _requires_a_caller_value(field)
    )
    if missing:
        raise ValueError(
            f"Camera {camera_name!r} is missing required option(s): {missing}. {class_name} accepts: {accepted}."
        )

    # strands defaults first so an explicitly configured value always wins.
    options = {**_CAMERA_STREAM_DEFAULTS, **{name: config[name] for name in fields if name in config}}
    try:
        return ConfigClass(**options)
    except (TypeError, ValueError) as exc:
        # Names the camera, which lerobot's own message cannot: its
        # ``__post_init__`` validation (e.g. a 3-character ``fourcc``) raises
        # with no idea which entry of the ``cameras`` dict it came from.
        raise ValueError(
            f"Failed to construct {class_name} for camera {camera_name!r}: {exc}. Options: {options}"
        ) from exc


def _normalize_max_relative_target(value: Any, robot_type: str) -> float | dict[str, float] | None:
    """Validate and normalize the ``max_relative_target`` servo safety clamp.

    ``max_relative_target`` caps how far each commanded goal position may move
    from the joint's present position. It is the only per-command travel limit
    on the hardware path, so a value the driver cannot honor is a safety defect
    rather than a cosmetic one - and lerobot's own consumer,
    ``lerobot.robots.utils.ensure_safe_goal_position``, honors a narrower set of
    values than the config dataclass accepts:

    - a non-finite limit disables the clamp with no signal. ``min(diff, nan)``
      and ``max(diff, -nan)`` both return ``diff`` unchanged, so the caller
      believes travel is limited while every goal is applied in full.
    - a negative limit inverts it. ``min(diff, -c)`` then ``max(..., c)``
      resolves to ``present + c`` for any requested goal, turning the clamp
      into a fixed-magnitude step generator that ignores the policy.
    - a zero limit discards every commanded motion, so the rollout is a no-op
      reported as success - the same outcome the rollout horizon guards refuse
      for ``duration`` and ``control_substeps``.
    - an ``int`` limit is refused outright at the first servo command, with a
      bare ``TypeError(10)`` naming neither the parameter nor the reason: that
      consumer dispatches on ``isinstance(value, float)``, while both the
      dataclass field (annotated ``float | dict[str, float] | None``) and PEP
      484's numeric tower accept an ``int``. So the one spelling a type checker
      is happiest with is the one that cannot reach the motors.

    Normalizing to ``float`` is therefore load-bearing, not cosmetic: it is what
    lets a type-correct ``max_relative_target=10`` reach the bus at all.

    Per-motor mapping values are held to the same domain. Their KEYS are not
    checked here - the motor names are known only once the bus is connected, so
    a mismatch is left to the driver's own ``keys must match`` refusal.

    Args:
        value: The caller-supplied limit: ``None`` to leave the clamp disabled,
            a positive finite number, or a mapping of motor name to one.
        robot_type: The lerobot robot type, named in the error so a fleet
            reports which arm was misconfigured.

    Returns:
        ``None`` unchanged, or the limit with every magnitude coerced to
        ``float``.

    Raises:
        ValueError: If the limit is not one the driver can honor.
    """
    param = "max_relative_target"
    context = f"Robot(robot_type={robot_type!r})"
    advice = (
        f"{param} caps how far each commanded goal position may move from the present "
        f"position, so only a finite positive limit can be honored. Omit it (or pass "
        f"None) to leave the clamp disabled."
    )

    if value is None:
        return None

    if isinstance(value, Mapping):
        if not value:
            raise ValueError(
                f"{context}: {param} must not be an empty mapping. A per-motor mapping has to name "
                f"every motor the driver reads, so an empty one is refused at the first servo "
                f"command. {advice}"
            )
        normalized: dict[str, float] = {}
        for key, limit in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{context}: {param} keys must be motor names (str), got "
                    f"{type(key).__name__}: {key!r}. A non-name key can never match a motor, so "
                    f"the mapping is refused at the first servo command."
                )
            if err := positive_finite_number_error(limit, f"{param}[{key!r}]", context):
                raise ValueError(f"{err} {advice}")
            normalized[key] = float(limit)
        return normalized

    if err := positive_finite_number_error(value, param, context):
        raise ValueError(f"{err} {advice}")
    return float(value)


@functools.cache
def _ensure_lerobot_robots_registered() -> None:
    """Import every robot driver subpackage so RobotConfig is populated.

    Walks ``lerobot.robots`` with ``pkgutil`` so we automatically pick up
    every robot lerobot ships -- past, present, and future -- including
    those whose ``robot_type`` doesn't match its subpackage name (e.g.
    ``hope_jr_arm`` in ``hope_jr/``, ``lekiwi_client`` in ``lekiwi/``,
    ``so100_follower`` and ``so101_follower`` both in ``so_follower/``).
    Then invokes lerobot's third-party plugin loader so any installed
    ``lerobot_robot_*`` distribution registers itself too.

    Idempotent via ``@functools.cache`` -- the first call walks the tree,
    subsequent calls are dict lookups.
    """
    try:
        import lerobot.robots as _lr_robots
    except ImportError as exc:
        # Distinguish two failure modes so the log level matches signal
        # value:
        #   1. lerobot wholly absent -- expected on sim-only / CI-only
        #      hosts that never reach hardware code; debug is enough.
        #      Caller will get a clean ``Unsupported robot type`` at the
        #      ChoiceRegistry lookup site.
        #   2. lerobot present but ``lerobot.robots`` unimportable --
        #      genuine partial-install signal worth a warning so the
        #      operator can triage without ``--log-level=DEBUG``.
        try:
            import lerobot  # noqa: F401  (probe-only)
        except ImportError:
            logger.debug("lerobot not installed: %s", exc)
        else:
            logger.warning(
                "lerobot is installed but lerobot.robots is not importable (partial install?): %s",
                exc,
            )
        return

    # Walk every immediate subpackage of ``lerobot.robots`` and import
    # it. Each subpackage's ``__init__`` (or its ``config_*`` module)
    # runs the ``@RobotConfig.register_subclass(...)`` decorator as a
    # side effect.
    for _, sub_name, is_pkg in pkgutil.iter_modules(_lr_robots.__path__):
        if not is_pkg:
            continue
        full_name = f"{_lr_robots.__name__}.{sub_name}"
        try:
            importlib.import_module(full_name)
        except (ImportError, OSError) as exc:
            # Driver-specific runtime dep missing (e.g. ``unitree_sdk2py``,
            # ``reachy2_sdk``) OR an OS-level probe failure inside a
            # driver's ``__init__`` (USB enumeration in ``unitree_sdk2py``
            # raising ``OSError``, ``FileNotFoundError`` on a missing SDK
            # config, etc.). Robot simply won't appear in the choice
            # registry -- that is the correct outcome: trying to construct
            # it later will raise ``Unsupported robot type`` with the
            # actual list of available types. Per AGENTS.md > Review
            # Learnings (#86) > "Exception Clauses Must Be Narrow" the
            # canonical pattern for hardware-probing imports is
            # ``(ImportError, OSError)``; widening further would mask
            # genuine bugs in driver registration code.
            logger.debug("[hardware_robot] skip %s: %s", full_name, exc)

    _ensure_lerobot_plugins_registered()


@functools.cache
def _ensure_lerobot_plugins_registered() -> None:
    """Import every installed third-party lerobot plugin distribution.

    lerobot's own loader imports every distribution whose name starts with one
    of its plugin prefixes (``lerobot_robot_``, ``lerobot_camera_``,
    ``lerobot_teleoperator_``, ...), and each of those registers itself into the
    matching :class:`draccus.ChoiceRegistry` as an import side effect. One call
    therefore populates every registry at once, which is why this is a single
    cached helper rather than a per-kind step: a vendor camera and a vendor
    robot arrive from the same import, so registering one kind while the caller
    happens to be resolving the other would leave the second unreachable.

    Idempotent via ``@functools.cache``.
    """
    try:
        from lerobot.utils.import_utils import register_third_party_plugins
    except ImportError:
        # ``register_third_party_plugins`` lives in modern lerobot only;
        # older versions skip this opt-in step (built-ins still work).
        logger.debug("[hardware_robot] register_third_party_plugins unavailable")
    else:
        try:
            register_third_party_plugins()
        except (ImportError, AttributeError, OSError) as exc:
            # #291: narrowed from bare ``except Exception`` per AGENTS.md
            # Review Learnings (#86). Third-party plugin registration can fail
            # for three benign, recoverable reasons: a plugin distribution
            # whose import chain is broken (ImportError), a lerobot version
            # whose loader entry-point shape differs (AttributeError), or an
            # OS-level probe inside a plugin's registration (OSError). Any of
            # these should degrade to "that plugin is absent from the registry"
            # -- not crash hardware init. A genuinely unexpected exception
            # (e.g. a plugin raising ValueError from buggy registration code)
            # now propagates so it is not silently masked.
            logger.warning("[hardware_robot] third-party plugin registration failed: %s", exc)


class TaskStatus(Enum):
    """Robot task execution status"""

    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RobotTaskState:
    """Robot task execution state"""

    status: TaskStatus = TaskStatus.IDLE
    instruction: str = ""
    #: ``time.monotonic()`` reading taken when the task began. Every reader
    #: subtracts it from a later reading of the same clock to get an elapsed
    #: duration, and none of them reports it as a point in time - so it is
    #: monotonic rather than wall-clock, which a step cannot move. Named for
    #: its clock so a future reader does not reach for ``time.time()``.
    start_mono: float = 0.0
    duration: float = 0.0
    step_count: int = 0
    error_message: str = ""
    task_future: Future | None = None
    #: The policy object the loop drove, set once it is built or handed in, so
    #: the task envelope can describe it (see ``Policy.reads_instruction``).
    #: Cleared with the rest of the state when a new task is claimed.
    policy: Any = None


class Robot(TeleopMixin, AgentTool):
    """Universal robot control with async task execution and status reporting."""

    def __init__(
        self,
        tool_name: str,
        robot: LeRobotRobot | RobotConfig | str,
        cameras: dict[str, dict[str, Any]] | None = None,
        action_horizon: int = 8,
        data_config: str | Any | None = None,
        control_frequency: float = 50.0,
        ros2_bridge: bool = False,
        ros2_domain: int = 0,
        ros2_commands: bool = True,
        ros2_transport: str = "rclpy",
        joint_limits: dict[str, tuple[float, float]] | None = None,
        dds_security_config: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Robot with async capabilities.

        Args:
            tool_name: Name for this robot tool
            robot: LeRobot Robot instance, RobotConfig, or robot type string
            cameras: Camera configuration dict:
                {"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}}
                Each key names one camera and must be a bare token of letters,
                digits, ``_`` or ``-``: it is the identity every consumer keys
                that camera's frames by - a level of the mesh topic they are
                published on, a segment of the S3 key they are offloaded to, and
                the ``observation.images.<name>`` feature key a recording writes
                them under - so a name carrying punctuation any of those reserves
                is refused here
                (:func:`~strands_robots.utils.camera_token_error`).
            action_horizon: Actions consumed from each inferred policy chunk
                before re-querying. Must be a positive integer - it is a lower
                bound on the chunk slice the task loop applies
                (``resolve_chunk_length`` returns
                ``max(action_horizon, policy.execution_horizon)``), and a
                ``0``/negative/float/``bool``/non-numeric value raises
                ``ValueError`` here rather than being silently clamped to a
                horizon the caller never asked for, or aborting the task mid-run
                once the arm is already connected.
            data_config: Data configuration (for GR00T compatibility)
            control_frequency: Control loop frequency in Hz (default: 50Hz).
                Must be a positive finite number - it is the divisor of the
                loop's per-action period (``1 / control_frequency``), the only
                throttle between two servo commands. A ``0``, negative,
                ``nan`` or ``inf`` rate raises ``ValueError`` here rather than
                leaving the loop unthrottled against real hardware.
            ros2_bridge: When True, publish this robot's live observation
                (``joint_states`` + one ``image_raw`` per camera) on a ROS 2
                domain so external ROS 2 nodes can subscribe to the physical
                robot, and the agent's own ``use_ros`` calls reach the same
                graph. The symmetric counterpart of ``SimEngine(ros2_bridge=...)``
                for real hardware. Requires ``rclpy`` (system ROS 2 / the
                official docker image); an ImportError is raised here if it is
                missing. Defaults to False - the robot never touches ROS 2,
                so disabling the bridge is simply the default (opt-in).
            ros2_domain: ROS 2 domain id (``ROS_DOMAIN_ID``) to publish on.
                Only an ``int`` in ``[0, 232]`` names a domain: RTPS derives its
                discovery ports from it, and 233 lands past the end of the port space.
            ros2_commands: When True (default), the bridge also subscribes to
                ``/<robot>/joint_command`` and forwards inbound messages to
                ``send_action`` so an external ROS 2 stack can drive the real
                arm (full duplex). Set False for a read-only telemetry bridge.
                Ignored unless ``ros2_bridge=True``. Only a boolean names a
                posture: the value is checked, not read by truthiness, so
                ``"false"`` cannot open the surface it asks to close.
            ros2_transport: Which ROS 2 backend the bridge uses:
                ``"rclpy"`` (default) - full ``sensor_msgs`` fidelity, needs a
                sourced ROS 2 distro; ``"rtps"`` - pure cyclonedds (a single
                pip wheel, no rclpy / no sourced distro), type coverage bounded
                by the local IDL bundle (joint_states + image_raw). Both emit
                byte-identical topics. Ignored unless ``ros2_bridge=True``.
            joint_limits: Optional ``{"<motor>.pos": (min, max)}`` clamp ranges
                threaded into the ROS 2 bridge, keyed by the joint name as it
                arrives on the wire (the same ``<motor>.pos`` names the bridge
                publishes in ``joint_states``). When set, an inbound
                ``joint_command``
                whose ANY joint is outside its declared range is rejected whole
                (no partial application). Requires ``ros2_bridge=True``.
            dds_security_config: Optional DDS Security credentials
                (``identity_ca``, ``certificate``, ``private_key``,
                ``governance``, ``permissions``; ``permissions_ca`` optional,
                each a non-empty string) for the pure-RTPS bridge. When ``ros2_commands=True`` on the
                ``"rtps"`` transport this (or the
                ``STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1`` opt-out) is
                REQUIRED - the bridge refuses to drive the arm over an unsecured
                DDS graph. Only the ``"rtps"`` transport consumes it (rclpy DDS
                Security is configured via the ROS 2 RMW keystore/env); passing
                it with ``ros2_transport="rclpy"`` raises. Requires
                ``ros2_bridge=True``.
            **kwargs: Robot-specific parameters (port, etc.)
        """
        super().__init__()

        self.tool_name_str = tool_name
        # ``action_horizon`` is how many actions of each inferred chunk the task
        # loop applies to the servo bus before re-querying the policy: it is the
        # value ``_execute_task_async`` hands to ``resolve_chunk_length``. That
        # helper coerces it with ``max(int(action_horizon), 1, ...)``, so a
        # ``0``/negative horizon was silently clamped to a single action per
        # inference - re-querying an open-loop chunked checkpoint every step
        # instead of replaying the chunk it was trained to emit, which is exactly
        # the out-of-distribution operation ``resolve_chunk_length`` documents as
        # the reason not to shrink the interval - while ``2.7`` was truncated,
        # ``"4"`` string-coerced and ``True`` acted as a silent 1. A value
        # ``int()`` cannot convert (``None``/``nan``/``inf``/a list) instead
        # reached that coercion only once the arm was already connected and the
        # first observation inferred on, aborting the task with a bare
        # ``TypeError``/``ValueError``. The identical domain is enforced on the
        # simulation's rollout counts (``SimEngine._validate_positive_int``);
        # validated BEFORE ``_initialize_robot`` opens the serial port, so a
        # rejected horizon never touches the arm.
        if horizon_error := positive_count_error(action_horizon, "action_horizon", "Robot"):
            raise ValueError(horizon_error)
        self.action_horizon = action_horizon
        self.data_config = data_config
        # ``action_sleep_time`` is ``1 / control_frequency`` and is the ONLY
        # thing bounding how fast the task loop commands the physical servo
        # bus: it is what ``_execute_task_async`` awaits between two
        # ``send_action`` calls. A non-positive or non-finite rate turns that
        # period into ``<= 0`` (``asyncio.sleep`` then returns immediately) or
        # into ``nan`` (``asyncio.sleep`` raises mid-task, after the first
        # action has already been applied), so the same rollout the simulation
        # refuses outright would run here against real hardware. The identical
        # domain is enforced on the rollout knobs of the simulation
        # (``SimEngine._validate_positive_frequency``); validated BEFORE
        # ``_initialize_robot`` opens the serial port, so a rejected rate never
        # touches the arm.
        if rate_error := positive_finite_number_error(control_frequency, "control_frequency", "Robot"):
            raise ValueError(rate_error)
        self.control_frequency = control_frequency
        self.action_sleep_time = 1.0 / control_frequency  # Time between actions

        # Task execution state
        self._task_state = RobotTaskState()
        # Stream-telemetry throttle (publish_step from the control loop).
        # A ``time.monotonic()`` reading, like every other elapsed-time base
        # here. ``-inf`` rather than ``0.0`` because monotonic readings are
        # only meaningful relative to each other: the first tick is due
        # regardless of where this platform's monotonic epoch happens to sit.
        self._last_stream_pub: float = float("-inf")
        # Imported here rather than at module top for the reason every other
        # mesh import in this file is: ``strands_robots.mesh`` pulls the
        # transport package, and a hardware Robot must construct without it.
        from strands_robots.mesh.session import stream_min_period_from_env

        # inf when the operator turned step telemetry off or named a rate no
        # loop can honor. Never a bare division: this runs in __init__, so a
        # ZeroDivisionError here fails the whole Robot(mode="real") bring-up.
        self._stream_min_period: float = stream_min_period_from_env()
        # An infinite period is the operator's opt-out, and no finite elapsed
        # time reaches it -- but ``_last_stream_pub`` starts below every
        # reading, so the subtraction alone reads ``inf >= inf`` and lets
        # exactly one publish (a whole observation, action and instruction)
        # past the opt-out per rollout. The period is therefore tested
        # directly, once here rather than on every tick of the control loop.
        self._stream_enabled: bool = math.isfinite(self._stream_min_period)
        # Annotated with the base class rather than the concrete pool: the two
        # uses below are ``submit`` and ``shutdown``, so a caller substituting a
        # different Executor is honouring the contract, not evading it.
        self._executor: Executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{tool_name}_executor")
        self._shutdown_event = threading.Event()
        # A stop request that arrived for the current task. An Event rather
        # than a task-state field because ``stop_task`` is called from the
        # agent-tool thread (and from the mesh dispatch) while the rollout runs
        # on the executor thread, and because ``_task_state.status`` cannot
        # carry the request: ``_execute_task_async`` writes ``RUNNING`` over
        # whatever the status held once the bring-up finishes, so a stop
        # recorded only as a status is lost.
        self._stop_requested = threading.Event()
        # Admission control for the motors bus. The arm has one command bus and
        # this class already models one rollout at a time (a single-slot
        # ``_task_state``, ``max_workers=1``), but ``_task_state.status`` cannot
        # decide admission: ``_execute_task_async`` only reaches ``RUNNING``
        # after ``_connect_robot`` and the policy build, so for the whole
        # bring-up window a second task would read a non-running status. The
        # claim is taken before that window opens and released when the rollout
        # ends, and the lock makes the check-and-claim atomic so two callers
        # racing at the same instant cannot both be admitted.
        self._task_admission = threading.Lock()
        self._task_claimed = False

        # Mesh attributes - populated by the Robot() factory after init.
        # Plain attributes (not properties) so test code can swap a fake mesh
        # in without going through the factory.
        # Set BEFORE _initialize_robot so cleanup()/__del__ never see an
        # AttributeError if construction fails partway through.
        self.mesh: Any = None
        self.peer_id: str | None = None

        # Validate the ROS 2 bridge precondition (transport + its optional
        # dependency) BEFORE _initialize_robot imports lerobot. Otherwise, in an
        # environment without the [lerobot] extra, _initialize_robot raises a
        # lerobot ImportError first and masks the transport-specific install
        # hint that the operator who set ros2_bridge=True actually needs to see.
        # require_optional caches the
        # module, so the real bridge construction in _init_ros_bridge pays nothing.
        # The two posture flags are graded first: ``"false"`` is truthy, so read
        # here by truthiness it would run the rclpy probe and, on a box without a
        # sourced distro, tell a caller who asked for no bridge to install ROS 2.
        # _init_ros_bridge grades them again for callers that enter there.
        for flag_name, flag_value in (("ros2_bridge", ros2_bridge), ("ros2_commands", ros2_commands)):
            if error := boolean_flag_error(flag_value, flag_name, type(self).__name__):
                raise ValueError(error)
        if ros2_bridge:
            self._check_ros2_bridge_deps(ros2_transport=ros2_transport)

        # Initialize robot using lerobot's abstraction
        self.robot = self._initialize_robot(robot, cameras, **kwargs)

        # lerobot 0.5.1 unified the SO-family calibration directory from
        # per-variant subdirs (``so100_follower/``, ``so101_follower/``) to a
        # single shared ``so_follower/``. Customers who calibrated on a
        # pre-0.5 lerobot have their JSON at the OLD path; the new
        # ``calibration_fpath`` resolves to the NEW path, finds nothing, and
        # reports ``is_calibrated=False`` -- which only surfaces as a confusing
        # RuntimeError on the first ``get_observation()``, not on ``connect()``.
        # Migrate (copy, never move -- old lerobot installs may still read it)
        # so a fresh customer's existing calibration Just Works.
        self._migrate_legacy_calibration()

        logger.info("%s initialized with async capabilities", tool_name)
        logger.info("Robot: %s (type: %s)", self.robot.name, getattr(self.robot, "robot_type", "unknown"))
        logger.info("Control frequency: %sHz (%.1fms per action)", control_frequency, self.action_sleep_time * 1000)

        # Get camera info if available
        if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
            cameras_list = list(self.robot.config.cameras.keys())
            logger.info("Cameras: %s", cameras_list)

        if data_config:
            logger.info("Data config: %s", data_config)

        # Optional ROS 2 telemetry bridge - opt-in, mirrors the simulation's
        # ``SimEngine(ros2_bridge=...)`` so a real arm and its digital twin look
        # identical on the ROS 2 graph. Initialized last so a bridge ImportError
        # surfaces only when the operator explicitly asked for it.
        self._init_ros_bridge(
            ros2_bridge=ros2_bridge,
            ros2_domain=ros2_domain,
            ros2_commands=ros2_commands,
            ros2_transport=ros2_transport,
            joint_limits=joint_limits,
            dds_security_config=dds_security_config,
        )

    # ------------------------------------------------------------------
    # ROS 2 telemetry bridge (opt-in) - mirror of SimEngine(ros2_bridge=...)
    # ------------------------------------------------------------------
    @staticmethod
    def _check_ros2_bridge_deps(*, ros2_transport: str) -> None:
        """Validate the ROS 2 bridge transport and its optional dependency.

        Called from ``__init__`` BEFORE ``_initialize_robot`` (which imports
        lerobot) so that, when ``ros2_bridge=True``, an invalid transport or a
        missing backend dependency surfaces its own remedy immediately - rather
        than being masked by the lerobot ImportError ``_initialize_robot`` raises
        first in an environment without the ``[lerobot]`` extra. The two
        transports need different remedies: ``cyclonedds`` is a pip wheel the
        ``[ros2]`` extra installs, while ``rclpy`` comes from a sourced ROS 2
        distro, so only the RTPS branch can name an extra.
        ``require_optional`` caches the resolved module, so constructing the
        real bridge later in ``_init_ros_bridge`` costs nothing extra.

        Args:
            ros2_transport: ``"rclpy"`` or ``"rtps"``.

        Raises:
            ValueError: If ``ros2_transport`` is not ``"rclpy"`` or ``"rtps"``.
            ImportError: If the transport's optional dependency is missing.
        """
        if ros2_transport not in ("rclpy", "rtps"):
            raise ValueError(f"ros2_transport must be 'rclpy' or 'rtps', got {ros2_transport!r}")
        if ros2_transport == "rtps":
            require_optional(
                "cyclonedds",
                extra="ros2",
                purpose="the pure-RTPS hardware bridge (Robot ros2_transport='rtps')",
            )
        else:
            require_optional(
                "rclpy",
                system_install=_RCLPY_TRANSPORT_INSTALL_HINT,
                purpose="the ROS 2 telemetry bridge (ros2_bridge=True)",
            )

    def _init_ros_bridge(
        self,
        *,
        ros2_bridge: bool = False,
        ros2_domain: int = 0,
        ros2_commands: bool = True,
        ros2_transport: str = "rclpy",
        joint_limits: dict[str, tuple[float, float]] | None = None,
        dds_security_config: dict[str, str] | None = None,
    ) -> None:
        """Initialize the optional ROS 2 telemetry bridge state.

        Plain method (not part of an ``__init__`` contract) so the lightweight
        test doubles that build a ``Robot`` via ``__new__`` need not thread it
        through. When ``ros2_bridge`` is True, the selected bridge is created
        eagerly so a missing backend dependency fails fast at construction
        rather than mid-task. The bridge is bound to ``self`` so that, with
        ``ros2_commands=True``, inbound ``/<robot>/joint_command`` messages are
        forwarded to ``self.send_action`` - full duplex, the real arm both
        publishes telemetry and is drivable from the ROS 2 graph.

        Two transports, byte-identical on the wire:

        * ``"rclpy"`` (default) - :class:`~strands_robots.hardware_ros_bridge.HardwareRosBridge`,
          full ``sensor_msgs`` fidelity, needs a sourced ROS 2 distro.
        * ``"rtps"`` - :class:`~strands_robots.hardware_rtps_bridge.HardwareRtpsBridge`,
          pure cyclonedds (a pip wheel, no rclpy / no sourced distro), type
          coverage bounded by the local IDL bundle.

        Args:
            ros2_bridge: Enable the ROS 2 bridge for this robot.
            ros2_domain: ROS 2 / DDS domain id to publish on.
                Only an ``int`` in ``[0, 232]`` names a domain: RTPS derives its
                discovery ports from it, and 233 lands past the end of the port space.
            ros2_commands: When True (default), also subscribe to
                ``joint_command`` and drive the arm; False for read-only. Only a
                boolean names a posture: the value is checked, not read by
                truthiness, so ``"false"`` cannot open the surface it asks to
                close.
            ros2_transport: ``"rclpy"`` or ``"rtps"`` (see above).
            joint_limits: Optional ``{"<motor>.pos": (min, max)}`` clamp ranges
                threaded into whichever bridge is built (both enforce them on
                inbound commands via the shared base), keyed by the joint name
                as it arrives on the wire.
            dds_security_config: Optional DDS Security credentials, consumed
                only by the ``"rtps"`` bridge. Passing it with the ``"rclpy"``
                transport raises (rclpy DDS Security is configured at the RMW
                layer, not by a config dict).

        Raises:
            ValueError: If ``ros2_bridge`` or ``ros2_commands`` is not a boolean;
                if ``ros2_domain`` is outside ``[0, 232]``; if
                ``ros2_transport`` is not ``"rclpy"`` or ``"rtps"``;
                if ``joint_limits`` / ``dds_security_config`` are supplied with
                ``ros2_bridge=False``; or if ``dds_security_config`` is supplied
                with ``ros2_transport="rclpy"``.
        """
        # Both flags decide whether this robot exposes an inbound, arm-driving
        # command surface, so they are checked on the shared boolean domain
        # rather than read by truthiness. ``"false"``, ``"no"``, ``"off"`` and
        # ``"0"`` - the spellings a YAML/env deployment config yields - are every
        # one of them truthy, so reading them would build a bridge for a caller
        # who asked for none and subscribe to ``joint_command`` for a caller who
        # asked for read-only. Answered before the first assignment below and
        # before either bridge is constructed, so a refused flag leaves no
        # bridge state, no ``ROS_DOMAIN_ID`` write and no DDS participant behind.
        for flag_name, flag_value in (("ros2_bridge", ros2_bridge), ("ros2_commands", ros2_commands)):
            if error := boolean_flag_error(flag_value, flag_name, type(self).__name__):
                raise ValueError(error)

        self._ros2_bridge_enabled = bool(ros2_bridge)
        # A domain id outside the RTPS port map cannot be published on by either
        # transport, so refuse it here rather than letting it reach one of them.
        if error := dds_domain_id_error(ros2_domain, "ros2_domain", type(self).__name__):
            raise ValueError(error)
        self._ros2_domain = ros2_domain
        self._ros2_transport = ros2_transport
        self._ros_bridge: Any = None
        if not self._ros2_bridge_enabled:
            # No silent no-op: a security/safety config that never reaches a
            # bridge is almost certainly an operator mistake.
            if joint_limits is not None or dds_security_config is not None:
                raise ValueError(
                    "joint_limits / dds_security_config require ros2_bridge=True "
                    "(they configure the ROS 2 bridge, which is disabled here)."
                )
            return

        if ros2_transport not in ("rclpy", "rtps"):
            raise ValueError(f"ros2_transport must be 'rclpy' or 'rtps', got {ros2_transport!r}")

        # DDS Security credentials are an RTPS (cyclonedds) concept; the rclpy
        # transport gets its DDS Security from the ROS 2 RMW keystore/env, not a
        # config dict. Reject rather than silently ignore.
        if dds_security_config is not None and ros2_transport != "rtps":
            raise ValueError(
                "dds_security_config is only supported with ros2_transport='rtps'; "
                "rclpy DDS Security is configured via the ROS 2 RMW keystore/env."
            )

        # Bind self so the bridge can drive the arm on inbound commands.
        # command_robot_name is pinned to the same namespace we publish
        # joint_states under (lerobot device .name, falling back to the tool
        # name) so a controller can echo our names straight back.
        if ros2_transport == "rtps":
            from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge

            self._ros_bridge = HardwareRtpsBridge(
                self,
                domain_id=self._ros2_domain,
                enable_commands=bool(ros2_commands),
                joint_limits=joint_limits,
                dds_security_config=dds_security_config,
            )
        else:
            from strands_robots.hardware_ros_bridge import HardwareRosBridge

            node = f"strands_hardware_{self.tool_name_str}"
            self._ros_bridge = HardwareRosBridge(
                self,
                domain_id=self._ros2_domain,
                node_name=node,
                enable_commands=bool(ros2_commands),
                joint_limits=joint_limits,
            )

    def _publish_ros_telemetry(self, observation: dict[str, Any], *, skip_images: bool = False) -> None:
        """Publish one ``joint_states`` (+ camera ``image_raw``) for ``observation``.

        No-op when the ROS 2 bridge is disabled or was never initialized
        (``getattr`` guard so test doubles built via ``__new__`` are safe). A
        publish failure never interrupts the control loop: the lerobot
        observation is the source of truth, the ROS 2 mirror is best-effort.

        Joint scalars (``<motor>.pos`` floats / numpy 0-d) become the
        ``JointState`` ``name``/``position`` arrays (sorted for determinism);
        ``(H, W, 3)`` arrays become per-camera ``image_raw`` frames.
        """
        bridge = getattr(self, "_ros_bridge", None)
        if bridge is None:
            return
        robot_name = getattr(self.robot, "name", None) or self.tool_name_str
        try:
            joints: list[tuple[str, float]] = []
            images: list[tuple[str, Any]] = []
            for key, value in observation.items():
                ndim = getattr(value, "ndim", None)
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) or ndim == 0:
                    joints.append((key, float(value)))
                elif not skip_images and ndim == 3 and getattr(value, "shape", (0, 0, 0))[2] == 3:
                    images.append((key, value))
            joints.sort(key=lambda kv: kv[0])
            bridge.publish_joint_states(robot_name, [k for k, _ in joints], [v for _, v in joints])
            for camera, frame in images:
                bridge.publish_image(robot_name, camera, frame)
        except Exception:
            logger.warning(
                "ROS 2 telemetry publish failed for %r; skipping this step",
                self.tool_name_str,
                exc_info=True,
            )

    def publish_ros_observation(self, *, skip_images: bool = False) -> dict[str, Any]:
        """Read the robot's current observation once and publish it on ROS 2.

        On-demand counterpart to the per-step publishing inside a running task:
        lets an agent turn an idle, connected robot into a live ROS 2 device
        without starting a control task. Requires ``ros2_bridge=True`` at
        construction.

        Args:
            skip_images: When True, publish ``joint_states`` only (opt out of
                the heavier camera ``image_raw`` topics).

        Returns:
            ``{"status": "success", ...}`` on publish, or
            ``{"status": "error", "content": [...]}`` when the bridge is
            disabled - tools never raise past dispatch.
        """
        if getattr(self, "_ros_bridge", None) is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"{self.tool_name_str}: ROS 2 bridge is disabled. "
                            "Construct the robot with ros2_bridge=True to publish telemetry."
                        )
                    }
                ],
            }
        observation = read_observation(self.robot)
        self._publish_ros_telemetry(observation, skip_images=skip_images)
        return {
            "status": "success",
            "content": [{"text": f"{self.tool_name_str}: published observation on ROS 2 domain {self._ros2_domain}"}],
        }

    def _shutdown_ros_bridge(self) -> None:
        """Tear down the ROS 2 bridge if one is active. Safe to call repeatedly."""
        bridge = getattr(self, "_ros_bridge", None)
        if bridge is not None:
            try:
                bridge.shutdown()
            finally:
                self._ros_bridge = None

    def _initialize_robot(
        self, robot: LeRobotRobot | RobotConfig | str, cameras: dict[str, dict[str, Any]] | None, **kwargs: Any
    ) -> LeRobotRobot:
        """Initialize LeRobot robot instance using native lerobot patterns.

        Raises:
            ImportError: When lerobot is not installed. Named with the extra
                that supplies it: this is the first lerobot import a
                ``Robot(..., mode="real")`` reaches, and on a core-only install
                a bare ``ModuleNotFoundError: No module named 'lerobot'`` from
                the line below was the whole answer the README's real-arm
                quickstart got. The purpose names the lerobot DRIVER rather
                than real mode at large, because ``driver="strands"`` builds a
                real robot through a native driver and needs no lerobot at all.
            ValueError: When *robot* is neither a lerobot ``Robot`` instance,
                a ``RobotConfig`` nor a robot type string.
        """
        require_optional(
            "lerobot",
            extra="lerobot",
            purpose='real-mode robots built through the lerobot driver (the default for Robot(..., mode="real"))',
        )
        from lerobot.robots.config import RobotConfig
        from lerobot.robots.robot import Robot as LeRobotRobot
        from lerobot.robots.utils import make_robot_from_config

        # Direct robot instance - use as-is
        if isinstance(robot, LeRobotRobot):
            return robot

        # Robot config - use lerobot's factory
        elif isinstance(robot, RobotConfig):
            return make_robot_from_config(robot)

        # Robot type string - create config and use lerobot's factory
        elif isinstance(robot, str):
            config = self._create_minimal_config(robot, cameras, **kwargs)
            return make_robot_from_config(config)

        else:
            raise ValueError(
                f"Unsupported robot type: {type(robot)}. "
                f"Expected LeRobot Robot instance, RobotConfig, or robot type string."
            )

    def _migrate_legacy_calibration(self) -> None:
        """Copy a pre-0.5 SO-family calibration file to the new shared path.

        lerobot 0.5.1 unified ``so100_follower/`` + ``so101_follower/`` (and
        the leader variants) into a single ``so_follower/`` /
        ``so_leader/`` directory under ``HF_LEROBOT_CALIBRATION``. The robot's
        ``calibration_fpath`` now points at the NEW location; an existing
        customer's JSON sits at the OLD location and is never found, so
        ``is_calibrated`` is ``False`` and the first ``get_observation()``
        raises.

        This best-effort migration copies (never moves -- a still-installed
        old lerobot may read the original) the legacy file into place when:
          * the robot exposes a ``calibration_fpath`` (lerobot >=0.5), and
          * the NEW path does not already exist, and
          * exactly one matching legacy file is found.

        Any failure is logged and swallowed -- a calibration that can't be
        migrated simply leaves the robot in its pre-existing (uncalibrated)
        state, which the connect path already reports clearly.
        """
        try:
            new_path = getattr(self.robot, "calibration_fpath", None)
            if new_path is None:
                return
            new_path = Path(new_path)
            if new_path.is_file():
                return  # already calibrated at the new path; nothing to do

            # The shared dir is the parent (e.g. ``.../so_follower``); the
            # legacy dirs are siblings named after the concrete variant.
            shared_dir = new_path.parent  # so_follower / so_leader
            calib_root = shared_dir.parent  # HF_LEROBOT_CALIBRATION/robots
            shared_name = shared_dir.name  # "so_follower"
            file_name = new_path.name  # "<id>.json"

            # Only the SO-family was renamed; restrict to *_follower / *_leader
            # subdirs sharing the same role suffix so we don't pull an
            # unrelated robot's file.
            if shared_name not in ("so_follower", "so_leader"):
                return
            role = shared_name.split("_", 1)[1]  # "follower" | "leader"

            candidates = [
                p for p in calib_root.glob(f"*_{role}/{file_name}") if p.is_file() and p.parent.name != shared_name
            ]
            if len(candidates) != 1:
                # Zero -> nothing to migrate. >1 -> ambiguous, refuse to guess.
                if len(candidates) > 1:
                    logger.warning(
                        "Multiple legacy calibration files found for %s; skipping auto-migration to avoid guessing: %s",
                        file_name,
                        [str(c) for c in candidates],
                    )
                return

            old_path = candidates[0]
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, new_path)
            logger.info("Migrated calibration file: %s -> %s", old_path, new_path)
        except OSError as exc:
            logger.warning("Calibration auto-migration failed (%s); leaving as-is", exc)

    def _create_minimal_config(
        self, robot_type: str, cameras: dict[str, dict[str, Any]] | None, **kwargs: Any
    ) -> RobotConfig:
        """Create a minimal lerobot RobotConfig for ``robot_type``.

        Uses lerobot's draccus ``ChoiceRegistry`` to resolve the registered
        config subclass. This is the same lookup ``make_robot_from_config``
        performs internally and means we automatically support every robot
        lerobot ships (so100/so101, koch, openarm, unitree_g1, aloha, ...)
        without maintaining a hand-rolled mapping.

        Robot-specific kwargs (``port``, ``robot_ip``, ``kp``, ``kd``,
        ``default_positions``, ``calibration_dir``, ``mock``, ``use_degrees``,
        ``is_simulation``, ``control_dt``, ``gravity_compensation``,
        ``controller``, ``max_relative_target``, ``disable_torque_on_disconnect``)
        are forwarded if and only if the resolved config dataclass declares
        a matching field. This means kwargs that exist in the union-of-
        robots allowlist but not on the current robot's dataclass are
        dropped silently -- that is the deliberate cross-robot
        polymorphism (``Robot('so101', kp=[...])`` won't fail just because
        ``kp`` is a unitree_g1 thing).

        A kwarg that is NOT in the allowlist at all is rejected with
        ``ValueError`` rather than dropped, per AGENTS.md > Review
        Learnings (#86) > "Reject silently-dropped kwargs". This catches
        typos like ``prot=`` (instead of ``port=``) at config-build time
        rather than as a delayed connection failure with no kwarg in
        sight.

        The mirror-image mistake -- a required field that no forwarded kwarg
        supplied -- is refused here too, naming the parameter the caller
        supplies rather than letting the dataclass report
        ``SOFollowerRobotConfig.__init__() missing 1 required positional
        argument: 'port'``. A missing port additionally names this host's
        servo-bus candidates and their USB serial numbers, because a port path
        is a position on the bus while the serial number is what survives a
        replug -- see :func:`strands_robots._serial_discovery.scan_serial_devices`.
        The scan answers a missing ``port`` only: naming serial devices for a
        missing ``remote_ip`` would point a network robot's caller at the wrong
        bus entirely.

        Each entry of ``cameras`` follows the same contract twice over: its
        ``type`` is resolved against lerobot's ``CameraConfig`` choice registry
        and its remaining keys against the fields of the class that resolves to
        -- see :func:`_build_camera_config`.

        Forwarded values are otherwise passed through as given, because their
        accepted domains are robot-specific. ``max_relative_target`` is the
        exception: it is the per-command servo travel limit, and the driver
        honors a narrower domain than the dataclass accepts, so it is validated
        and normalized here -- see :func:`_normalize_max_relative_target`.

        Raises:
            ValueError: If a kwarg is unknown to both the cross-robot allowlist
                and the resolved dataclass, if a ``cameras`` entry is malformed,
                if ``max_relative_target`` is not a limit the driver can honor,
                if the resolved dataclass declares a required field that no
                forwarded kwarg supplied, or if the resolved config class
                refuses the assembled values.
            TypeError: If lerobot resolves ``robot_type`` to a non-dataclass
                config class, which would make kwarg filtering unsafe.
        """
        # ``lerobot`` is already a hard dep at this point (``_initialize_robot``
        # imports it eagerly). Importing the camera + config modules here is
        # cheap and the only reason it isn't at module top is that some
        # downstream packagers tree-shake unused submodules.
        from lerobot.robots.config import RobotConfig

        # Convert cameras to lerobot format.
        camera_configs: dict[str, Any] = {}
        if cameras:
            for name, config in cameras.items():
                camera_configs[name] = _build_camera_config(name, config)

        # Trigger lerobot's lazy registration. Each robot driver registers
        # its config via @RobotConfig.register_subclass at module-import
        # time, but ``lerobot.robots.__init__`` does NOT eagerly import
        # every driver subpackage (deliberate: keeps ``import lerobot``
        # cheap when only one robot is needed, and avoids hard deps on
        # robot-specific SDKs). The mapping ``robot_type → import path``
        # is also non-trivial:
        #
        #     so101_follower  → lerobot.robots.so_follower (shared module)
        #     hope_jr_arm     → lerobot.robots.hope_jr     (shared module)
        #     lekiwi_client   → lerobot.robots.lekiwi      (shared module)
        #
        # so we cannot just ``import_module(f"lerobot.robots.{robot_type}")``.
        # Instead we walk every subpackage of ``lerobot.robots`` once
        # (filesystem-driven, future-proof) and use lerobot's own
        # third-party plugin loader for ``lerobot_robot_*`` distributions.
        # Both calls are cached after the first invocation so subsequent
        # ``Robot()`` calls are essentially free.
        _ensure_lerobot_robots_registered()

        # Resolve the config class via lerobot's draccus ChoiceRegistry -
        # this is the source-of-truth lookup that ``make_robot_from_config``
        # uses; staying on it means we track upstream renames automatically.
        try:
            ConfigClass = RobotConfig.get_choice_class(robot_type)
        except KeyError:
            # Two registries can answer better than lerobot's own listing here,
            # and each returns a reason or ``None`` so the listing survives when
            # neither applies.
            #
            # First: a robot THIS package drives natively. Its driver ships
            # here, and lerobot's listing does not mention it, so a caller with
            # no reason to guess ``driver="strands"`` is left at a dead end.
            # Checked before the teleoperator arm because the two populations
            # overlap -- ``unitree_g1`` is a lerobot teleoperator type as well
            # as a natively driven robot -- and for a name in that overlap the
            # native answer is the right one: someone who asked ``Robot()`` for
            # a robot this package drives wants its driver, not an instruction
            # to build the leader that teleoperates it.
            from strands_robots.drivers.registry import _native_driver_refusal

            if native := _native_driver_refusal(robot_type):
                raise ValueError(native) from None

            # Then: a leader arm is a teleoperator, not a robot, and lerobot
            # keeps the two in separate registries. Listing the robot types for
            # a leader name is the retry that torque-enables the arm a human is
            # holding, so name the kind it really is instead. ``Robot()``'s own
            # registry guard does this for an unregistered ``*_leader`` name; a
            # leader that IS registered -- as itself, which is the honest thing
            # to register -- reaches this site instead.
            from strands_robots.teleoperator import _other_lerobot_kind_refusal

            if other := _other_lerobot_kind_refusal(robot_type, wanted="robot"):
                raise ValueError(other) from None

            available = sorted(RobotConfig.get_known_choices().keys())
            # ``from None`` -- the KeyError is an internal detail of
            # lerobot's draccus registry; suppress the chained traceback
            # for a cleaner user-facing error.
            raise ValueError(
                f"Unsupported robot type: {robot_type!r}. Known lerobot robot types: {available}"
            ) from None

        # Build candidate field set so we only pass kwargs the dataclass
        # actually accepts. ``RobotConfig.get_choice_class`` always returns
        # a dataclass today (every ``@RobotConfig.register_subclass`` site
        # is a ``@dataclass``-decorated class). If that contract ever
        # breaks we want a loud error here, not a silent default that
        # blindly forwards every kwarg downstream (per AGENTS.md > Key
        # Conventions #6 -- "no silent defaults on error").
        try:
            valid_fields = {f.name for f in dataclasses.fields(ConfigClass)}
        except TypeError as exc:
            raise TypeError(
                f"lerobot returned a non-dataclass config class "
                f"{ConfigClass!r} for robot_type={robot_type!r}; strands_robots "
                f"cannot filter kwargs safely. Please file an issue against "
                f"lerobot or strands_robots."
            ) from exc

        config_data: dict[str, Any] = {}

        # ``id`` namespaces lerobot's calibration files. Users can override
        # by passing ``id=...`` (e.g. when one calibration file is shared by
        # multiple peer instances of the same robot type -- left_arm.json,
        # right_arm.json). Default to the strands tool name otherwise.
        if "id" in valid_fields:
            config_data["id"] = kwargs.get("id", self.tool_name_str)
        elif "id" in kwargs:
            # #292: every lerobot RobotConfig declares ``id`` today, so an
            # operator-supplied ``id=`` is normally consumed above. If a future
            # RobotConfig subclass drops the field, silently discarding an
            # explicit ``id=`` would namespace calibration files wrong with no
            # signal. Surface it: the generic unknown-kwarg gate below would
            # also catch it, but this names the specific regression so the
            # diagnostic is actionable.
            logger.warning(
                "[hardware_robot] robot_type=%r config %s does not declare an 'id' "
                "field; the explicit id=%r will not namespace calibration files. "
                "This is unexpected for a lerobot RobotConfig -- please file an issue.",
                robot_type,
                ConfigClass.__name__,
                kwargs["id"],
            )

        # Cameras are common to every lerobot Robot.
        if "cameras" in valid_fields:
            config_data["cameras"] = camera_configs

        # Forward known robot-specific kwargs only if the target dataclass
        # declares them. The full set is union-of-all known lerobot robot
        # configs - adding new ones here is safe because we filter against
        # ``valid_fields`` before constructing.
        forwardable = _FORWARDABLE_KWARGS
        for key in forwardable:
            if key in kwargs and key in valid_fields:
                config_data[key] = kwargs[key]
            elif key in kwargs:
                # #294/#297: the kwarg is in the cross-robot allowlist but the
                # resolved dataclass does not declare it -- the documented
                # polymorphism carve-out (e.g. ``Robot('so101', kp=[...])``
                # against a heterogeneous fleet). This is intentional, but a
                # silent drop leaves operators with no way to audit why a kwarg
                # they passed had no effect. Emit a debug signal naming the
                # dropped kwarg and the robot type so the drop is observable
                # without changing the tolerant behaviour.
                logger.debug(
                    "[hardware_robot] dropping cross-robot kwarg %r for robot_type=%r: "
                    "not declared on %s (forwardable-allowlist polymorphism carve-out)",
                    key,
                    robot_type,
                    ConfigClass.__name__,
                )

        # Forward kwargs that are declared on the target dataclass but not
        # in the cross-robot allowlist. This future-proofs new lerobot fields
        # without requiring a strands_robots release to add them to forwardable.
        for key in kwargs:
            if key not in config_data and key not in {"id", "cameras"} and key in valid_fields:
                config_data[key] = kwargs[key]

        # Reject kwargs unknown to BOTH the cross-robot allowlist AND the
        # resolved target dataclass. Per AGENTS.md > Review Learnings (#86)
        # > "Reject silently-dropped kwargs", a typo like ``prot=`` must
        # surface immediately -- but a genuinely new lerobot field that the
        # target dataclass declares should Just Work without a strands_robots
        # release. This keeps typo-rejection while preserving the "zero
        # strands_robots changes for new robots" promise for new *kwargs* too.
        always_allowed = {"id", "cameras"}
        recognised = set(forwardable) | always_allowed | valid_fields
        unknown = set(kwargs) - recognised
        if unknown:
            raise ValueError(
                f"Unknown kwarg(s) for robot_type={robot_type!r}: "
                f"{sorted(unknown)}. This robot's dataclass accepts: "
                f"{sorted(valid_fields)}. The cross-robot allowlist is: "
                f"{sorted(set(forwardable) | always_allowed)}. "
                f"(If this is a typo, fix it.)"
            )

        # ``max_relative_target`` is the per-command servo travel limit, and the
        # driver honors a narrower domain than this dataclass accepts - see
        # ``_normalize_max_relative_target``. Validated after both forwarding
        # loops and the kwarg-name gate, so it is checked exactly when it is the
        # effective knob: a robot whose config does not declare the field never
        # reads it, and refusing a value that robot would drop anyway would
        # report an error for a parameter it has no opinion about.
        if "max_relative_target" in config_data:
            config_data["max_relative_target"] = _normalize_max_relative_target(
                config_data["max_relative_target"], robot_type
            )

        # A required field no forwarded kwarg satisfied is the commonest caller
        # mistake on this path -- 8 of lerobot's serial robot types declare
        # ``port`` with no default -- and letting the dataclass raise reports it
        # as ``SOFollowerRobotConfig.__init__() missing 1 required positional
        # argument: 'port'``: a lerobot internal, naming neither the parameter
        # the caller supplies nor the devices this host has. The unknown-kwarg
        # gate above already refuses the mirror-image mistake (a typo like
        # ``prot=``) by naming the accepted fields, so the two halves of "the
        # caller got a kwarg wrong" are reported the same way. Detected here
        # rather than from the exception because the answer is on the bus, and
        # by the time the dataclass raises nobody is looking at it: the same
        # values would reach the same dataclass either way, so this changes
        # which sentence a refused call gets, not which calls are refused --
        # which is why the scan asks ``_requires_a_caller_value`` rather than
        # reading the absent default alone: a field the class derives for itself
        # is one the constructor does not accept, so counting it as missing
        # would refuse a call that builds.
        missing_required = [
            field.name
            for field in dataclasses.fields(ConfigClass)
            if _requires_a_caller_value(field) and field.name not in config_data
        ]
        if missing_required:
            remedy = ", ".join(f"{name}=..." for name in missing_required)
            hint = (
                f"missing required parameter(s) {missing_required}, which the caller supplies -- e.g. "
                f"Robot({self.tool_name_str!r}, mode='real', {remedy})."
            )
            # Only a port-shaped parameter is answered by a serial scan. Naming
            # this host's serial devices for a missing ``remote_ip`` would point
            # a network robot's caller at the wrong bus entirely.
            if any(name == "port" or name.endswith("_port") for name in missing_required):
                hint += (
                    f" {describe_serial_candidates(scan_serial_devices())}"
                    " A port path is a position on the bus, not an identity: it can change when the device is"
                    " replugged, while the usb id does not."
                )
            raise ValueError(
                f"Failed to construct {ConfigClass.__name__} for robot type {robot_type!r}: "
                f"{hint} Config: {config_data}"
            )

        try:
            return ConfigClass(**config_data)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to construct {ConfigClass.__name__} for robot type {robot_type!r}: {e}. Config: {config_data}"
            ) from e

    async def _get_policy(
        self,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        **policy_kwargs: Any,
    ) -> Policy:
        """Create policy on-the-fly from invocation parameters.

        The public entry points refuse an unusable ``policy_port`` before the
        arm is connected (:meth:`_policy_port_error`), so the falsy check below
        is the floor for a direct call to this private helper rather than the
        first place a caller's port is judged.
        """
        from strands_robots.registry.policies import get_policy_provider

        from .policies import create_policy

        # Per-provider port requirement: the registry's "requires"
        # field is the source of truth - groot/lerobot_async dial a server
        # and need a port, while mock/lerobot_local build in-process and
        # need none. Hardcoding the port demand here made every port-less
        # provider unrunnable on hardware through the mesh execute path.
        requires: tuple[str, ...] = ()
        try:
            spec = get_policy_provider(policy_provider) or {}
            requires = tuple(spec.get("requires", ()) or ()) if spec else ("port",)
        except Exception:
            # Unknown provider or registry unavailable - preserve the
            # historical conservative behaviour (demand the port).
            requires = ("port",)

        if "port" in requires and not policy_port:
            raise ValueError(
                f"policy_port is required for policy_provider={policy_provider!r} (it dials a policy server)"
            )

        policy_config: dict[str, Any] = {}
        if policy_port:
            policy_config["port"] = policy_port
            policy_config["host"] = policy_host

        if self.data_config:
            policy_config["data_config"] = self.data_config

        # Forward checkpoint/provider kwargs (model_path, policy_type,
        # pretrained_name_or_path, server_address, ...) that the mesh
        # dispatch collects from the wire command. Previously the hardware
        # entry points had no way to receive them (TypeError -> generic
        # "dispatch error"), making checkpoint providers (lerobot_local)
        # unrunnable on hardware over the mesh while sim peers accepted the
        # same command. Per-provider validity is create_policy's contract
        # (unknown kwargs are the provider's own refusal to make).
        policy_config.update({k: v for k, v in policy_kwargs.items() if v is not None})

        return create_policy(policy_provider, **policy_config)

    def _close_open_devices(self) -> None:
        """Close every device that is open, one at a time and best-effort.

        Two callers need exactly this, both for a device set that is only
        *partly* open -- the state lerobot's own ``Robot.disconnect()`` cannot
        recover, because it is gated on ``is_connected``
        (``bus.is_connected and all(cam.is_connected ...)``):

        - **A failed connect.** lerobot's ``MotorsBus.connect()`` opens the
          serial port *before* the motor handshake, so a failed handshake
          (e.g. an unpowered bus) leaves ``is_connected`` True while
          fire-and-forget writes to the dead bus keep "succeeding". A robot's
          ``connect()`` then opens its cameras one at a time
          (``for cam in self.cameras.values(): cam.connect()``) with no
          cleanup of its own, so a camera that fails to open leaves every
          camera ahead of it in the set still streaming. Either leak makes the
          *next* attempt unrecoverable rather than merely failing: the retry
          raises ``DeviceAlreadyConnectedError`` on a device that is perfectly
          healthy, which masks the device that actually failed.
        - **Teardown.** ``cleanup()`` prefers the driver's own ``disconnect()``
          while the robot is fully connected, and falls back here when it is
          not -- or when that disconnect raised partway and abandoned the
          devices behind it.

        Each device is closed independently: one close that raises must
        neither mask the caller's original error nor stop the remaining
        devices from being closed, which is the failure mode lerobot's single
        unguarded disconnect loop has.
        """
        # ``getattr`` on ``self``, not on ``self.robot``: this also runs on
        # the partial-construction path, where ``__init__`` raised in the very
        # statement that assigns ``self.robot`` and so the attribute does not
        # exist at all. Guarding the inner attribute cannot help there.
        # ``_shutdown_ros_bridge`` already reads its own handle this way.
        robot = getattr(self, "robot", None)
        bus = getattr(robot, "bus", None)
        try:
            if bus is not None and getattr(bus, "is_connected", False):
                # disable_torque=False: this runs only where the driver's own
                # disconnect is unavailable, would be refused, or has already
                # raised, so a torque write here would most likely raise again
                # -- before ``closePort``, leaving the port open, which is the
                # one outcome this method exists to prevent.
                bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.debug("%s port close failed", self.tool_name_str)

        # lerobot builds ``cameras`` with ``make_cameras_from_configs``, whose
        # contract is ``dict[str, Camera]``; anything else is not a camera set
        # this rollback can walk.
        cameras = getattr(robot, "cameras", None)
        for name, camera in cameras.items() if isinstance(cameras, Mapping) else ():
            try:
                # A camera that failed to open already released its own
                # resources in ``connect()``; disconnecting it again would
                # raise ``DeviceNotConnectedError``.
                if getattr(camera, "is_connected", False):
                    camera.disconnect()
            except Exception:  # noqa: BLE001 - best-effort, and per camera so
                # one stuck camera cannot keep the rest of the set open.
                logger.debug(
                    "%s camera %s close failed",
                    self.tool_name_str,
                    name,
                )

    def _disconnect_devices(self) -> None:
        """Close the physical devices this robot holds: motors bus and cameras.

        Every other teardown branch in ``cleanup()`` releases a resource the
        *library* owns -- the teleop loop, the task executor, the mesh client,
        the ROS bridge. The devices are the only resources that are physical,
        and they were the only ones left open: a serial port is exclusive on
        Linux and macOS, so a second process (or a re-constructed ``Robot`` in
        this one) could not open the same ``/dev/tty*``, which made the
        documented recovery for a wedged arm -- tear down and reconnect --
        unavailable without exiting the process. Each camera also holds a
        ``/dev/video*`` node and a read thread.

        The driver's own ``disconnect()`` is preferred while it is callable,
        because that is where a follower disables torque and releases the
        gripper; closing the port underneath it skips both and leaves the arm
        energised at its last commanded position.
        """
        # ``getattr`` on ``self``, not on ``self.robot``: this also runs on
        # the partial-construction path, where ``__init__`` raised in the very
        # statement that assigns ``self.robot`` and so the attribute does not
        # exist at all. Guarding the inner attribute cannot help there.
        # ``_shutdown_ros_bridge`` already reads its own handle this way.
        robot = getattr(self, "robot", None)
        disconnect = getattr(robot, "disconnect", None)
        if callable(disconnect) and getattr(robot, "is_connected", False):
            try:
                disconnect()
                return
            except Exception as exc:  # noqa: BLE001 - teardown is best-effort
                logger.warning(
                    "%s: robot.disconnect() raised during cleanup: %s",
                    self.tool_name_str,
                    exc,
                )

        # Reached when the driver exposes no ``disconnect``, when it would
        # refuse (lerobot gates it on ``is_connected``, false in every
        # half-open state), or when it raised partway through its own single
        # unguarded loop and so abandoned every device after the one that
        # failed. Closing each device independently is what still gets the
        # port and the surviving cameras shut.
        self._close_open_devices()

    def _observe_ledger(self) -> set[str]:
        """The devices an observe action opened that no connect has yet been handed.

        Entries are ``"bus"`` and ``"camera:<name>"``. Created by the first
        observe action rather than in ``__init__``: a tool that never observed
        has no ledger and nothing to hand back, and that is the same answer.

        Created under the device lock, double-checked. Two first-ever observers
        - an agent's ``get_state`` on a ``to_thread`` worker and the teleop
        loop's lazy connect - would otherwise each build a set and the second
        assignment clobber the first's recorded open, leaving that device
        permanently outside the hand-back. ``bus_access.bus_lock`` guards its
        own creation the same way, for the same reason.
        """
        ledger: set[str] | None = getattr(self, "_observe_opened", None)
        if ledger is not None:
            return ledger
        with bus_lock(self.robot):
            ledger = getattr(self, "_observe_opened", None)
            if ledger is None:
                ledger = set()
                self._observe_opened = ledger
            return ledger

    def _hand_back_observe_devices(self) -> None:
        """Close every device an observe action left open, so ``connect()`` starts from nothing.

        ``get_state`` opens the motor bus alone and ``render`` opens one
        camera, and each leaves its device open so the next read is cheap.
        lerobot's ``Robot.connect()`` cannot start from either: every device's
        own ``connect()`` is ``@check_if_already_connected``, so an open bus is
        refused before any camera opens, and an open camera is refused in the
        camera loop - *before* ``configure()``, which is where the operating
        mode, the PID gains and the gripper's current and torque limits are
        written. The two failures are not alike. The bus one is a refused
        connect. The camera one is a connect that reports success: with one
        camera, ``is_connected`` (``bus and all(cameras)``) reads True once the
        bus is up, so the rollout drives servos the driver never configured.

        The trigger is the ledger, not ``is_connected``, because on an arm with
        **no** cameras an open bus *is* ``is_connected`` - so a gate keyed on
        that field skips ``connect()`` altogether, and ``get_state`` followed by
        a rollout drives an arm whose ``configure()`` never ran. Measured on the
        fake: ``(True, "")`` with ``configure_calls == 0``. What the ledger
        records is what this tool borrowed; a robot the caller connected
        themselves is not in it and is left as they left it.

        A close that fails raises: the caller refuses and rolls back, instead of
        letting the driver's ``DeviceAlreadyConnectedError`` pass for a robot
        that was already up. That is the one way this differs from
        ``_close_open_devices``, which is best-effort because it runs where
        there is nothing left to refuse.

        Synchronous, and called only from :meth:`_bring_up_robot`, which holds
        ``bus_lock(self.robot)`` from before this hand-back until ``connect()``
        has returned. The lock is re-entered here around the bus close so the
        contract is stated where the close is; it is the lock ``get_state``,
        ``render`` and the mesh probes open and read under, so a read in flight
        finishes before the port goes, and no observe action can open a device
        between this hand-back and the connect that follows it.

        Raises:
            Exception: Whatever the device's own ``disconnect()`` raised,
                unchanged - the ledger keeps the entry, so the next connect
                tries again.
        """
        ledger = self._observe_ledger()
        if not ledger:
            return
        robot = self.robot
        if "bus" in ledger:
            bus = getattr(robot, "bus", None)
            if bus is not None and getattr(bus, "is_connected", False):
                logger.info("closing the bus an observe action opened so %s can connect fully", robot)
                with bus_lock(robot):
                    # disable_torque=False: nothing was energised, and a torque
                    # write to a bus that was only ever read is the register
                    # write the observe actions promise not to make.
                    bus.disconnect(disable_torque=False)
            ledger.discard("bus")
        cameras = getattr(robot, "cameras", None)
        for entry in sorted(e for e in ledger if e.startswith("camera:")):
            name = entry.removeprefix("camera:")
            camera = cameras.get(name) if isinstance(cameras, Mapping) else None
            if camera is not None and getattr(camera, "is_connected", False):
                logger.info("closing camera %s an observe action opened so %s can connect fully", name, robot)
                camera.disconnect()
            ledger.discard(entry)

    def _bring_up_robot(self) -> None:
        """Hand back what an observe action borrowed, then connect - one unit under the device lock.

        An observe action leaves its device open: ``get_state`` the motor bus,
        ``render`` one camera. lerobot's ``connect()`` cannot start from either,
        and the two fail differently - an open bus refuses the connect, an open
        camera lets it *succeed* with ``configure()`` skipped. So every borrowed
        device goes back BEFORE ``is_connected`` is read: on a camera-less arm
        the open bus alone reads as connected.

        The three steps are one critical section under ``bus_lock(self.robot)``
        because the observe actions are ungated by design, so an agent or a
        monitor calls them freely beside a bring-up. Serialising each device
        operation was not enough: between a hand-back that had closed the bus
        and the ``is_connected`` read that followed it there was a thread hop,
        and a ``get_state`` landing in that hop reopened the bus - so the read
        answered True, ``connect()`` was skipped, and the rollout drove servos
        whose operating mode, gains and torque limits ``configure()`` never
        wrote. The observe actions take the same lock around their own
        check-and-open, so one that loses the race waits for the connect to
        finish and then finds a robot already up.

        Synchronous, so the lock is released by the thread that took it: the
        teleop loop calls it directly and :meth:`_connect_robot` runs it in one
        ``to_thread`` call. ``RLock``, so the nested acquisitions inside the
        hand-back and the driver are fine.

        Raises:
            Exception: A hand-back that failed, unchanged - the caller refuses
                and rolls back rather than letting the driver's
                ``DeviceAlreadyConnectedError`` pass for a robot that was
                already up. Or whatever ``connect()`` raised, except the
                already-connected refusal, which is reachable here only for a
                device the caller opened themselves and is what the ledger
                deliberately leaves alone.
        """
        from lerobot.utils.errors import DeviceAlreadyConnectedError

        with bus_lock(self.robot):
            self._hand_back_observe_devices()
            # Not a return: the calibration check in ``_connect_robot`` runs on
            # this path too. It used to be skipped for a connected robot, so a
            # refused first call (bus left open) made the second call report
            # success for an arm the gate had refused.
            if self.robot.is_connected:
                logger.info(f"{self.robot} already connected")
                return
            logger.info(f"Connecting to {self.robot}...")
            try:
                self.robot.connect(False)  # calibrate=False
            except DeviceAlreadyConnectedError:
                # Expected and fine - a device the caller connected themselves
                logger.info(f"{self.robot} was already connected")
            except Exception as e:
                # The string version of the same refusal
                error_str = str(e).lower()
                if "already connected" in error_str or "is already connected" in error_str:
                    logger.info(f"{self.robot} connection already established")
                else:
                    raise

    async def _connect_robot(self) -> tuple[bool, str]:
        """Connect to robot hardware with proper error handling.

        Returns:
            tuple[bool, str]: (success, error_message) - error_message is empty on success
        """
        try:
            # Hand-back, ``is_connected`` and ``connect()`` as one locked unit,
            # in one thread: ``_bring_up_robot`` says why the three cannot be
            # separated by an ``await``.
            await asyncio.to_thread(self._bring_up_robot)

            # Final connection check
            if not self.robot.is_connected:
                error_msg = f"Failed to connect to {self.robot}"
                logger.error(f"{error_msg}")
                return False, error_msg

            # Check robot calibration, reading the flag exactly once. On a
            # lerobot arm ``is_calibrated`` is ``read_calibration()`` - a
            # homing-offset and range sweep of every servo - and the guard used
            # to be ``hasattr(self.robot, "is_calibrated") and not
            # self.robot.is_calibrated``, which evaluated it twice. ``hasattr``
            # also swallows an ``AttributeError`` raised *inside* that read,
            # which is indistinguishable from "this driver has no such
            # property": a driver whose ``is_calibrated`` is
            # ``self.bus.is_calibrated`` over a bus built lazily raises exactly
            # that, and the gate was then skipped altogether - measured
            # ``(True, "")``, cameras open and ``configure()`` already run, for
            # an arm whose calibration was never checked. Absent is not the same
            # as unreadable: a driver that declares the attribute at all (on the
            # class as a property, or on the instance as a plain flag) must
            # answer, and a read that raises falls to the handler below, which
            # refuses and closes the port. A driver with no notion of
            # calibration is calibrated by lerobot's own contract for the
            # property: "should be always True if not applicable".
            try:
                is_calibrated = self.robot.is_calibrated
            except AttributeError:
                if hasattr(type(self.robot), "is_calibrated"):
                    raise  # the property exists; its own read failed
                is_calibrated = True
            if not is_calibrated:
                error_msg = (
                    f"Robot {self.robot} is not calibrated. Please calibrate the robot manually"
                    " first using LeRobot's calibration process (lerobot-calibrate)"
                )
                logger.error(f"{error_msg}")
                # Refused, so close what this attempt opened. ``connect()``
                # above succeeded - bus open, cameras open, ``configure()``
                # run - and this check is the first to say no. Left open, the
                # NEXT attempt short-circuits on ``is_connected`` at the top
                # of this method and returns success for an arm this branch
                # just refused: the calibration gate would hold for exactly
                # one call. It also held the serial port for the life of the
                # process, so `lerobot-calibrate` - the remedy this message
                # names - could not open it. Measured on an uncalibrated
                # SO-101: first call refused, second call ``(True, "")``.
                self._close_open_devices()
                return False, error_msg

            logger.info(f"{self.robot} connected and ready")
            return True, ""

        except Exception as e:
            error_msg = self._connect_failure_message(e)
            logger.error(f"{error_msg}")
            # Same rollback as the lazy teleop connect: without it a half-open
            # port makes the NEXT _connect_robot short-circuit on
            # "already connected" and report success against a dead bus.
            self._close_open_devices()
            return False, error_msg

    def _connect_failure_message(self, exc: BaseException) -> str:
        """Name the device that failed to open and the remedy for that device.

        lerobot's ``connect()`` opens the motors bus, then each camera; either
        raises its own message. The wrapper used to append one fixed remedy -
        "Ensure robot is calibrated and accessible on the specified port" - to
        whichever came up, so a camera that did not exist was answered with a
        calibration hint (and an agent relayed "recalibrate if needed" for an
        unplugged arm). A camera message names lerobot's object,
        ``OpenCVCamera(99)``, not the key the operator wrote in ``cameras=``;
        that key is looked up here so the reply says which camera.

        The lookup matches the camera *objects* lerobot built, the way
        :meth:`_close_open_devices` reads them, because every backend writes
        every one of its messages with ``f"{self}"`` -- the object's ``str`` is
        the one identity the text is sure to carry, whatever the backend and
        whatever failed (opening, warmup, a read). No config field can serve as
        that identity: only ``OpenCVCameraConfig`` declares ``index_or_path``,
        so matching a config named a webcam and left every other registered
        backend -- RealSense (``serial_number``), ZMQ
        (``camera_name@address:port``), Reachy 2 (``name, image_type``) -- with
        the general remedy, which is the calibration hint for a camera fault
        this method exists to remove.

        Args:
            exc: What ``connect()`` raised.

        Returns:
            One message naming the device and the remedy for that device.
        """
        text = " ".join(str(exc).split()).rstrip(".")
        cameras = getattr(self.robot, "cameras", None)
        for key, camera in cameras.items() if isinstance(cameras, Mapping) else ():
            if str(camera) in text:
                return (
                    f"Robot connection failed: camera {key!r} did not open - {text}. "
                    "Fix or remove that entry in cameras=; the motors bus is closed again."
                )
        port = getattr(getattr(self.robot, "config", None), "port", None)
        if "port" in text.lower():
            where = f" on port {port!r}" if port else ""
            return (
                f"Robot connection failed: {text}. The motors bus did not open{where}: check the USB "
                "cable and power, then find the port (lerobot-find-port or scan_serial_devices)."
            )
        return f"Robot connection failed: {text}. Ensure the robot is powered, on the right port and calibrated."

    async def _initialize_policy(self, policy: Policy) -> bool:
        """Initialize policy with robot state keys."""
        try:
            # Get robot state keys from observation
            test_obs = await asyncio.to_thread(read_observation, self.robot)

            # Filter out camera keys to get robot state keys
            camera_keys = []
            if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
                camera_keys = list(self.robot.config.cameras.keys())

            robot_state_keys = [k for k in test_obs.keys() if k not in camera_keys]

            # Set robot state keys in policy
            policy.set_robot_state_keys(robot_state_keys)
            return True

        except Exception as e:
            logger.error(f"Failed to initialize policy: {e}")
            return False

    @property
    def _rollout_stop_latched(self) -> bool:
        """Whether either latch says the rollout in flight must not go on.

        Two independent events end a rollout, and a reader that consults one of
        them lets the stage it guards run on after the robot has been asked to
        stop:

        * ``_stop_requested`` -- an operator or fleet stop, set by
          :meth:`stop_task` for a task in ``RUNNING`` or ``CONNECTING``.
        * ``_shutdown_event`` -- a terminal teardown, set by :meth:`cleanup`
          before it does anything else.

        The shutdown latch is the one a reader is apt to miss, because it is the
        only record of a teardown that lands mid-bring-up: ``cleanup()`` gates
        its ``stop_task()`` call on ``status == RUNNING``, so a task still in
        ``CONNECTING`` gets no stop latch at all.

        Three readers ask this question -- this gate, the control loop's exit
        condition and the terminal-status discriminator -- so they read one
        predicate rather than three copies that can drift apart.
        """
        return self._stop_requested.is_set() or self._shutdown_event.is_set()

    def _settle_task_duration(self) -> None:
        """Write how long the task that is ending here actually ran.

        ``duration`` is the figure every reply and every later ``status`` call
        reports for a finished task, and it is reset to ``0.0`` when a task
        starts. Only two writers ever settled it: the rollout's own loop when
        it ran to its budget, and :meth:`get_task_status` while the task is
        RUNNING. Every OTHER way a task can end - stopped from outside,
        stopped during bring-up, a connect that failed, a policy that would
        not initialize, a rollout that raised - left the reset value in place,
        so the task reported ``0.0s`` beside a non-zero step count and kept
        reporting it for good. Measured on a rollout that had applied 39
        commands to the servo bus: ``error: 39 steps in 0.0s``.

        So this is called wherever a terminal state is recorded, and it owns
        the arithmetic that :meth:`get_task_status` also needs - one formula,
        so a stop and the status call after it cannot disagree.

        A ``start_mono`` of ``0.0`` means no task has begun (it is the
        dataclass default), and is left alone: subtracting it would report the
        seconds since boot as the duration of a task that never ran.
        """
        if self._task_state.start_mono:
            self._task_state.duration = time.monotonic() - self._task_state.start_mono

    def _honor_stop_request(self) -> bool:
        """Record a latched stop or shutdown as the task's terminal state.

        Called at each point in ``_execute_task_async`` where the task is about
        to move on to a stage with effects of its own -- a policy-server dial or
        checkpoint load, an observation read, and finally commanding the arm.
        Reads :attr:`_rollout_stop_latched`, so this is the only thing that has
        to run for either latch to be honored during bring-up.

        Reading only ``_stop_requested`` was not enough, and the gap was not the
        reported status -- the terminal block below already discriminates on
        both latches -- but the bring-up that ran after the robot had been told
        to shut down. ``cleanup()`` sets the shutdown latch and only then calls
        ``stop_task()``, gated on ``status == RUNNING``, so a teardown arriving
        while the task was still in ``CONNECTING`` set no stop latch, both stage
        checks passed, and the rollout finished the bring-up it had just been
        asked to abandon. Measured on the two-device double, with the shutdown
        latch already set: the motors bus opened and every camera warmed, one
        observation read off them, and ``Policy.reset()`` called on a policy
        object the caller may still be driving elsewhere -- the documented
        ``policy_object=`` reuse pattern, and the same side effects
        :meth:`_shutdown_error` refuses a *new* task for.

        Returns:
            ``True`` when a stop or a shutdown was latched and the caller must
            abandon the rollout, ``False`` when the task may proceed.
        """
        if not self._rollout_stop_latched:
            return False
        self._task_state.status = TaskStatus.STOPPED
        self._settle_task_duration()
        logger.info(
            "%s: task stopped during bring-up: '%s'",
            self.tool_name_str,
            self._task_state.instruction,
        )
        return True

    async def _execute_task_async(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        policy_object: Policy | None = None,
        n_steps: int | None = None,
        **policy_kwargs: Any,
    ) -> None:
        """Execute robot task in background thread (internal method).

        ``policy_object`` (when given) is driven as-is and the provider/port
        arguments are ignored - the ``run_policy`` path. ``n_steps`` caps the
        number of applied actions; the loop stops at whichever of
        ``duration`` / ``n_steps`` comes first. The two conditions are ANDed,
        so ``duration`` bounds the rollout even with a step cap - it is
        validated at every public entry point rather than only when it is the
        sole horizon. ``n_steps`` is validated there too: a cap the
        ``step_count < n_steps`` comparison cannot be made against never
        reaches this loop.

        Either way the policy's per-episode state is reset before the rollout,
        so a task never begins from the state the previous task left behind.
        """
        # resolve_chunk_length lives in the light policies.base module (no torch);
        # imported lazily to match this file's policy-import convention.
        from .policies.base import resolve_chunk_length

        try:
            # Update task state. The stop latch is cleared here so a stop
            # pressed while the robot was idle does not pre-empt the next task,
            # and ``duration`` is reset with it: a task stopped during bring-up
            # never reaches the terminal block that writes it, and reporting the
            # PREVIOUS task's elapsed time would misdescribe this one.
            self._stop_requested.clear()
            self._task_state.status = TaskStatus.CONNECTING
            self._task_state.instruction = instruction
            self._task_state.start_mono = time.monotonic()
            self._task_state.duration = 0.0
            self._task_state.step_count = 0
            self._task_state.error_message = ""
            self._task_state.policy = None

            # Connect to robot. Remember whether THIS call did the connecting:
            # lerobot's ``connect()`` ends in ``configure()``, whose
            # ``torque_disabled()`` block re-enables torque on exit, so the arm
            # goes stiff where it stands the moment it connects - before any
            # policy exists. A bring-up that then fails (checkpoint missing,
            # server down, trust gate) would leave the caller's arm locked for
            # the rest of the session with nothing to drive it. Measured on the
            # SO-101 tool: policy refused, task ERROR, ``robot.is_connected``
            # True, torque on until process exit. A connection the caller made
            # BEFORE this task is theirs and is left alone.
            connected_here = not self._robot_is_connected()
            connected, connect_error = await self._connect_robot()
            if not connected:
                self._task_state.status = TaskStatus.ERROR
                self._task_state.error_message = connect_error or f"Failed to connect to {self.tool_name_str}"
                self._settle_task_duration()
                return

            # A stop pressed during the bring-up window above (a motors-bus
            # handshake plus per-camera warmup - seconds on a real arm) is
            # honored here, before the policy is even built, so the rollout is
            # abandoned without commanding the arm. The arm this task just
            # energized is released with it: an abandoned bring-up leaves
            # nothing to drive it, exactly as a policy that cannot be built
            # does, and the operator who pressed stop is told so rather than
            # left holding a stiff arm.
            if self._honor_stop_request():
                self._task_state.error_message = await self._release_uncommanded_arm(connected_here)
                return

            # Get policy instance: a caller-supplied pre-built object wins;
            # otherwise build the server-backed one from provider + port.
            if policy_object is not None:
                policy_instance = policy_object
            else:
                try:
                    policy_instance = await self._get_policy(policy_port, policy_host, policy_provider, **policy_kwargs)
                except Exception as e:
                    self._task_state.status = TaskStatus.ERROR
                    self._settle_task_duration()
                    released = await self._release_uncommanded_arm(connected_here)
                    self._task_state.error_message = " ".join(p for p in (str(e), released) if p)
                    logger.error(f"Task execution failed: {e}")
                    return
            self._task_state.policy = policy_instance

            # Initialize policy with robot state keys
            if not await self._initialize_policy(policy_instance):
                self._task_state.status = TaskStatus.ERROR
                self._settle_task_duration()
                released = await self._release_uncommanded_arm(connected_here)
                self._task_state.error_message = " ".join(p for p in ("Failed to initialize policy", released) if p)
                return

            logger.info(f"Starting task: '{instruction}' on {self.tool_name_str}")
            if policy_object is not None:
                logger.info(f"Using pre-built policy object: {type(policy_instance).__name__}")
            else:
                logger.info(f"Using policy: {policy_provider} on {policy_host}:{policy_port}")

            # Real-Time Chunking contract (mirror PolicyRunner._run_policy_rollout
            # in ``strands_robots.simulation.policy_runner``): tell the policy the
            # control rate ONCE before the rollout so RTC-capable providers
            # (pi0/pi0.5/SmolVLA/MolmoAct2) convert their inference latency into a
            # correct count of action steps and blend chunk seams identically to
            # sim. Without it a wrong assumed rate corrupts RTC blending at every
            # frequency except the assumed one. No-op for non-RTC policies.
            policy_instance.set_control_frequency(self.control_frequency)

            # Clear per-episode policy state before the rollout, mirroring the
            # per-episode reset PolicyRunner performs in
            # ``strands_robots.simulation.policy_runner``. A caller may drive one
            # policy object through several tasks (that is the documented
            # ``run_policy(policy_object=...)`` usage), and Policy.reset exists
            # to clear exactly the state that must not cross that boundary -
            # action chunk caches, sampler RNG, KV-caches. Without it the first
            # actions of a task can be the PREVIOUS task's cached chunk, so the
            # arm executes commands inferred for a different instruction.
            # No seed is passed: a hardware task has no per-episode seed to
            # forward, so this asks only for a state clear.
            #
            # Fail-soft, matching PolicyRunner: a policy whose reset raises
            # still gets driven, with the stale state named in the warning.
            try:
                policy_instance.reset()
            except Exception as e:  # noqa: BLE001 - reset is best-effort
                logger.warning(
                    "policy.reset() raised %s; continuing with possibly stale per-episode state",
                    e,
                )

            # Building a server-backed policy is a second multi-second window
            # (a network connect plus a handshake), so the latch is re-checked
            # before the arm is commanded for the first time - and the arm is
            # released here too, for the same reason as at the first gate: it
            # has still never been commanded.
            if self._honor_stop_request():
                self._task_state.error_message = await self._release_uncommanded_arm(connected_here)
                return

            self._task_state.status = TaskStatus.RUNNING
            # The rollout budget is a duration, so it is measured on a clock
            # that only ever moves forward at one second per second. Read on
            # ``time.time()`` an NTP correction, a ``date -s`` or a resume from
            # suspend moved this comparison by the size of the step: forward it
            # ended the rollout early and left the arm parked mid-task, and
            # backward it kept commanding the servo bus past the budget
            # ``_duration_error`` exists to bound. Neither was reported.
            start_mono = time.monotonic()

            while (
                time.monotonic() - start_mono < duration
                and (n_steps is None or self._task_state.step_count < n_steps)
                and self._task_state.status == TaskStatus.RUNNING
                and not self._rollout_stop_latched
            ):
                # Get observation from robot
                observation = await asyncio.to_thread(read_observation, self.robot)

                # Mirror the live observation on ROS 2 (no-op unless the bridge
                # is enabled). Best-effort: never blocks or breaks the loop.
                self._publish_ros_telemetry(observation)

                # Synchronous control loop: observe -> infer -> apply. The arm
                # holds its last commanded position during inference (we issue no
                # servo motion mid-inference), so exactly 0 control steps elapse
                # between issuing the query and applying its first action. A
                # counted 0 (not a wall-clock estimate) is the deterministic RTC
                # seam offset for this loop, matching the synchronous sim runner.
                # No-op for non-RTC policies. Drivers that coast servos during
                # inference would need a non-zero counted delay here instead.
                policy_instance.set_rtc_observed_delay(0)

                # Get actions from policy
                robot_actions = await policy_instance.get_actions(observation, instruction)

                # Consume the chunk by the policy's own re-query interval rather
                # than a raw action_horizon slice. resolve_chunk_length ignores
                # action_horizon for RTC policies (they own execution_horizon;
                # stretching/shrinking it silently degrades cross-chunk blending
                # to open-loop replay) and returns max(action_horizon,
                # execution_horizon) for non-RTC policies, so the prior
                # robot_actions[:self.action_horizon] behaviour is preserved for
                # single-step and open-loop chunked providers.
                chunk_len = resolve_chunk_length(policy_instance, self.action_horizon)
                for action_dict in robot_actions[:chunk_len]:
                    if self._task_state.status != TaskStatus.RUNNING:
                        break
                    if n_steps is not None and self._task_state.step_count >= n_steps:
                        break
                    await asyncio.to_thread(write_action, self.robot, action_dict)
                    self._task_state.step_count += 1
                    # Per-step mesh telemetry: publish_step had consumers
                    # (robot_mesh watch, dashboards) but no producers. Rate-
                    # limited; failures never touch the control loop.
                    _mesh = getattr(self, "mesh", None)
                    if _mesh is not None and self._stream_enabled:
                        _now_stream = time.monotonic()
                        if _now_stream - self._last_stream_pub >= self._stream_min_period:
                            self._last_stream_pub = _now_stream
                            try:
                                _mesh.publish_step(
                                    self._task_state.step_count,
                                    observation,
                                    action_dict,
                                    instruction=instruction,
                                    policy=policy_provider,
                                )
                            except Exception:  # noqa: BLE001 - telemetry only
                                pass
                    # Wait for action to complete before sending next action
                    # Default 50Hz (0.02s)
                    await asyncio.sleep(self.action_sleep_time)

            # Update final state
            elapsed = time.monotonic() - start_mono
            self._task_state.duration = elapsed

            if self._task_state.status == TaskStatus.RUNNING:
                # A stop can land in the gap between the check above and the
                # ``RUNNING`` write, which overwrites the ``STOPPED`` status
                # ``stop_task`` recorded. The latch survives that write, so it
                # is what decides here - otherwise an interrupted task reports
                # itself completed.
                #
                # ``_shutdown_event`` is the other latch that ends the loop and
                # it needs the same treatment, for a reason no entry-point guard
                # can cover: ``cleanup()`` sets it and only then calls
                # ``stop_task()``, gated on ``status == RUNNING``, so a shutdown
                # landing after this task's last stage check sets no stop latch.
                # Without it such a rollout exits the loop on its first
                # evaluation and reports ``completed`` with 0 steps.
                #
                # Both are read through ``_rollout_stop_latched`` -- the one
                # predicate the bring-up gate and the loop condition above also
                # read, so the three cannot drift apart.
                if self._rollout_stop_latched:
                    self._task_state.status = TaskStatus.STOPPED
                    logger.info(f"Task stopped: '{instruction}' after {self._task_state.step_count} steps")
                else:
                    self._task_state.status = TaskStatus.COMPLETED
                    logger.info(
                        f"Task completed: '{instruction}' in {elapsed:.1f}s ({self._task_state.step_count} steps)"
                    )

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            self._task_state.status = TaskStatus.ERROR
            self._task_state.error_message = str(e)
            self._settle_task_duration()

    @property
    def _task_message_label(self) -> str:
        """``"Error"`` or ``"Note"`` for the message the task left behind.

        ``error_message`` is also where an abandoned bring-up records that the
        arm was released - true of a task an operator STOPPED, which is not an
        error. Labelling that "Error" would report a handled interrupt as a
        failure, so the label follows the terminal status.
        """
        return "Error" if self._task_state.status == TaskStatus.ERROR else "Note"

    def _robot_is_connected(self) -> bool:
        """``robot.is_connected`` as a plain bool, False when the driver cannot say."""
        try:
            return bool(getattr(self.robot, "is_connected", False))
        except Exception:  # noqa: BLE001 - a bus that cannot answer is not connected
            return False

    async def _release_uncommanded_arm(self, connected_here: bool) -> str:
        """Disconnect a robot THIS task connected and never commanded.

        Called from every bring-up exit that ends the task after
        :meth:`_connect_robot` and before the loop commands the arm: the policy
        could not be built, it could not be initialized, or a stop/shutdown was
        latched at one of the two stage gates. In all of them the arm has not
        moved - it stands where the caller left it, now with torque on, because
        lerobot's ``connect()`` ends in ``configure()``, whose
        ``torque_disabled()`` block re-enables torque on exit. Disconnecting
        (``disable_torque_on_disconnect`` defaults to True) returns it to
        exactly the state the caller had before the call, and leaves nothing
        energized that nothing is driving.

        That the arm has not moved is also why a rollout which fails while
        RUNNING is NOT released here: an arm mid-motion dropping under gravity
        is the hazard, holding its pose is not.

        Args:
            connected_here: Whether this task opened the connection. One the
                caller made before the task is theirs and is left alone.

        Returns:
            A sentence for the reply saying what was done, or ``""`` when there
            was nothing to release. A disconnect that fails yields a sentence
            saying the arm is still energized; the failure is logged and, on an
            error path, the original cause stays the headline.
        """
        if not connected_here or not self._robot_is_connected():
            return ""
        try:
            await asyncio.to_thread(self.robot.disconnect)
        except Exception as exc:  # noqa: BLE001 - reported, must not mask the cause
            logger.warning("Could not disconnect %s after the abandoned bring-up: %s", self.tool_name_str, exc)
            return f"The robot is still connected (torque on); disconnecting it failed: {exc}"
        logger.info("%s disconnected again: no policy to drive it", self.tool_name_str)
        return "The robot was disconnected again (torque released) since nothing will drive it."

    @staticmethod
    def _duration_error(duration: Any, method: str) -> dict[str, Any] | None:
        """Reject a task ``duration`` the control loop cannot honor.

        ``duration`` is the elapsed-time budget the loop compares against
        (``time.monotonic() - start_mono < duration``), and on hardware it bounds
        every rollout: unlike the simulation's ``run_policy`` - where an
        ``n_steps`` recomputes ``duration`` and supersedes it - the two
        conditions are ANDed here, so ``duration`` is the effective horizon
        even when a step cap is given.

        Only a positive finite value can be honored. A value ``<= 0`` (and
        ``nan``, which is never ``<= 0`` but fails every comparison it reaches)
        makes the loop condition false on its first evaluation: the task used
        to report ``status="success"`` for a rollout that never queried the
        policy and never commanded the arm. ``inf`` is worse than a fast
        budget - it never becomes false, so the loop commands the servo bus
        indefinitely and the blocking entry point never returns. A
        non-numeric value reached the comparison intact and surfaced a bare
        ``TypeError`` naming a comparison internal ("'<' not supported between
        instances of 'float' and 'str'") rather than the parameter.

        The accepted domain is
        :func:`~strands_robots.utils.positive_finite_number_error`, shared with
        the loop's ``control_frequency`` guard and with the simulation's
        :meth:`~strands_robots.simulation.base.SimEngine._validate_duration`,
        so the same budget cannot be refused for a digital twin and accepted
        for the arm it mirrors.

        That parity is over the value domain, and it stops one control period
        short of zero. In sim ``duration`` is a FACTOR of the step count
        (``int(duration * control_frequency)``), so the sim guard additionally
        refuses a duration below one control period - it resolves to no steps
        there. Here it is a deadline the loop compares elapsed time against, so
        the first iteration always runs and any positive budget commands the arm
        at least once. The difference is in what the parameter means, not in the
        values the two layers consider usable.

        Args:
            duration: The caller-supplied value to validate.
            method: Public entry point name, used to prefix the message.

        Returns:
            A tool-shaped error dict naming ``duration``, or ``None`` when the
            value can be honored.
        """
        if error := positive_finite_number_error(duration, "duration", method):
            return {"status": "error", "content": [{"text": error}]}
        return None

    @staticmethod
    def _n_steps_error(n_steps: Any, method: str) -> dict[str, Any] | None:
        """Reject a task ``n_steps`` cap the control loop cannot count against.

        ``n_steps`` is the optional step cap the loop compares the applied-action
        count against (``n_steps is None or step_count < n_steps``), ANDed with
        the wall-clock budget :meth:`_duration_error` bounds. ``None`` is the
        documented "no cap" spelling and leaves ``duration`` the sole horizon;
        every other value has to be a count that comparison can be made against.

        A cap ``<= 0`` (and ``nan``, which is never ``<= 0`` but fails every
        comparison it reaches) makes the condition false on its first
        evaluation, so the task reported ``status="success"`` and "Policy
        rollout completed: 0 steps" for a rollout that never queried the policy
        and never commanded a servo - the same false ``completed`` an unusable
        ``duration`` used to produce. ``inf`` is never false, so the requested
        cap silently vanishes and the rollout runs to the ``duration`` budget
        instead. ``True`` reads as a silent cap of one. A float cap applies a
        count the caller never named: ``2.7`` stops after three applied
        actions. A non-numeric cap reached the comparison intact and surfaced a
        bare ``TypeError`` naming a comparison internal ("'<' not supported
        between instances of 'int' and 'str'") rather than the parameter.

        The accepted domain is
        :func:`~strands_robots.utils.positive_count_error`, already shared with
        this loop's ``action_horizon`` and with the simulation's ``run_policy``
        step horizon, so the same cap cannot be refused for a digital twin and
        accepted for the arm it mirrors.

        Args:
            n_steps: The caller-supplied cap to validate, or ``None`` for no cap.
            method: Public entry point name, used to prefix the message.

        Returns:
            A tool-shaped error dict naming ``n_steps``, or ``None`` when the
            cap can be honored (including when no cap was requested).
        """
        if n_steps is None:
            return None
        if error := positive_count_error(n_steps, "n_steps", method):
            return {"status": "error", "content": [{"text": error}]}
        return None

    @staticmethod
    def _policy_requires_error(
        policy_provider: str | None, policy_kwargs: dict[str, Any], method: str
    ) -> dict[str, Any] | None:
        """Reject a provider build that is missing a keyword it cannot act without.

        This surface's envelope around
        :func:`~strands_robots.registry.policies.policy_requires_error`, which owns the
        domain and states why. :meth:`_policy_port_error` judges the ``port``
        entry; this judges the rest - the checkpoint a ``lerobot_local`` /
        ``lerobot_async`` policy is built from. Here the harm is that
        :meth:`_connect_robot` energizes the arm before the first
        ``get_actions`` fails on the executor thread with nobody left to tell;
        the quickstart's real-arm step did exactly this.

        Args:
            policy_provider: Provider name; unknown or unregistered providers
                are left to ``create_policy`` to refuse.
            policy_kwargs: Checkpoint/provider keywords the caller supplied.
            method: Public entry point name, used to prefix the message.

        Returns:
            A tool-shaped error dict naming the missing keyword(s) and the
            provider, or ``None`` when every required keyword is present.
        """
        # ``port``/``host`` are ignored because they never travel in
        # ``**policy_kwargs``: they arrive as the named ``policy_port`` /
        # ``policy_host`` parameters, so reading them here would find every
        # caller's absent and refuse a port that WAS supplied. The port is
        # judged by :meth:`_policy_port_error`, which runs first, and the host
        # has a default.
        reason = policy_requires_error(
            policy_provider,
            policy_kwargs,
            method,
            "the task would start, energize the arm and fail at its first action",
            ignore=("port", "host"),
        )
        return None if reason is None else {"status": "error", "content": [{"text": reason}]}

    @staticmethod
    def _policy_description(policy_provider: Any, policy_host: Any, policy_port: Any) -> str:
        """Describe the policy an operator is being asked to approve, truthfully.

        The approval prompt named the policy as ``at {host}:{port}``
        unconditionally, so a rollout that supplied no port - the normal way to
        run 9 of the 14 registered providers - was described as a server ``at
        localhost:None``. There is no such endpoint. An operator approving a
        real arm's motion is reading this line to decide, and a location that
        does not exist is the one part of it they cannot check.

        Which of three things is true is the registry's to answer, through
        :func:`~strands_robots.registry.policies.provider_reads_a_port`:

        - a port was supplied, so the policy is that server;
        - no port and the provider declares no ``port`` keyword (``mock``,
          ``lerobot_local``, ``rl``, ...): it is built in this process and
          dials nothing;
        - no port and the provider does dial one (``cosmos3``,
          ``lerobot_async``, ``remote`` default their port rather than requiring
          it): there is a server, at a port this call did not choose. Calling
          that "no server" would be a new false statement, not a fix.

        A provider the registry does not know is described by name only. It
        cannot reach the operator through the agent tool -
        :meth:`_policy_provider_error` refuses an unresolvable name before the
        gate - but ``composite`` and ``persistent`` resolve by auto-discovery
        with no registry entry to read, and claiming either endpoint for them
        would be a guess.

        Args:
            policy_provider: Provider name as supplied by the caller.
            policy_host: Host as supplied by the caller, defaulted upstream.
            policy_port: Port as supplied by the caller, ``None`` when absent.

        Returns:
            A phrase for the approval prompt, e.g. ``"policy groot at
            localhost:5555"`` or ``"policy mock built in this process, no
            server"``.
        """
        from strands_robots.registry.policies import provider_reads_a_port

        # Rendered through the shared plain renderer for the reason every refusal
        # is: this text is built from caller-supplied values on the path whose
        # purpose is to tell an operator what they are approving, so producing it
        # must not be able to raise. Plain rather than quoted - the prompt reads
        # as a sentence, and ``repr`` would spell a host "'localhost'".
        provider = refusal_str(policy_provider)
        if policy_port is not None:
            return f"policy {provider} at {refusal_str(policy_host)}:{refusal_str(policy_port)}"
        reads_a_port = provider_reads_a_port(provider if policy_provider else None)
        if reads_a_port is False:
            return f"policy {provider} built in this process, no server"
        if reads_a_port is True:
            return f"policy {provider} at {refusal_str(policy_host)}, on the provider's default port"
        return f"policy {provider}"

    @staticmethod
    def _policy_provider_error(policy_provider: Any, method: str) -> dict[str, Any] | None:
        """Reject a ``policy_provider`` no policy can be resolved from.

        Every other pre-flight check in this class asks the registry about the
        named provider - :meth:`_policy_port_error` reads ``requires`` to decide
        whether a port is mandatory, :meth:`_policy_requires_error` reads it for
        the checkpoint keywords - so the provider name is the value that decides
        whether any of them can be answered at all. Both were written to defer
        an unresolvable name to ``create_policy``, whose refusal
        (``"Unknown policy provider: 'grooot'"``) is raised from
        :meth:`_get_policy`, *after* :meth:`_connect_robot` has energized the
        arm and after the operator has approved the rollout. That is the very
        shape both of those checks exist to close, left open for the one
        argument that names what is being built.

        Deferring also made the port check answer for it. With no registry entry
        to read, ``requires`` cannot say a port is optional, so a missing port
        was reported as ``"policy_port is required"`` for a provider that does
        not exist - the wrong reason, whose remedy leads to the next wrong
        reason, and indistinguishable from the same message for a correctly
        spelled ``groot``. ``TestAPortTheProviderDoesNotReadIsRefused`` states
        the rule this restores: "answering it here would report a port problem
        for a provider problem". It held for a *supplied* port, which
        :meth:`_policy_port_error` leaves to the provider, and not for a missing
        one. Running this first means an unresolvable name never reaches that
        check, so the port check keeps its rule and its wording unchanged.

        Resolution is asked of
        :func:`~strands_robots.policies.factory.provider_can_be_created`, which
        walks the same three stages ``create_policy`` does - a provider the
        public ``register_policy()`` API registered at runtime (by name or
        alias), a smart string (HF id, ``zmq://`` URL), then the registry's own
        account of what ``import_policy_class`` accepts - so a runtime-registered
        provider, a declared alias (``lerobot``, ``random``, ``c3``) and an
        auto-discovered module (``composite``, ``persistent``) are not refused
        for being absent from
        :func:`~strands_robots.registry.policies.list_policy_providers`.

        Args:
            policy_provider: The provider name as supplied by the caller. A
                falsy value is left alone: it is the "not named" spelling that
                :meth:`_policy_requires_error` also passes over, and the port
                check already reports what a nameless build is missing.
            method: Public entry point name, used to prefix the message.

        Returns:
            A tool-shaped error dict naming the provider and the ones that
            resolve, or ``None`` when a policy can be resolved from the value.
        """
        from strands_robots.policies.factory import list_providers, provider_can_be_created

        if not policy_provider or provider_can_be_created(refusal_str(policy_provider)):
            return None
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"{method}: unknown policy_provider {refusal_repr(policy_provider)}. "
                        f"Available: {', '.join(list_providers())} "
                        "(declared aliases such as 'lerobot' for 'lerobot_local' also resolve). "
                        "Nothing was dispatched and the arm was not energized."
                    )
                }
            ],
        }

    @staticmethod
    def _policy_port_error(policy_port: Any, method: str, policy_provider: str | None = None) -> dict[str, Any] | None:
        """Reject a ``policy_port`` no policy can be built from.

        ``policy_port`` is the one caller-supplied value that decides whether a
        policy exists at all: :meth:`_get_policy` hands it to
        :func:`~strands_robots.policies.create_policy`, whose provider
        constructor dials it. Every other rollout knob is checked before the
        motors bus is claimed - see :meth:`_duration_error` and
        :meth:`_n_steps_error`, whose call sites document why a budget "is
        refused before the arm is commanded rather than after". This one was
        read only inside :meth:`_execute_task_async`, *after*
        :meth:`_connect_robot` had energized the arm, so a port that can never
        build a policy still ran the whole bring-up window that method's own
        comment describes as "a motors-bus handshake plus per-camera warmup -
        seconds on a real arm". :meth:`start_task` additionally reported
        ``status="success"`` and "Task started" for it, because the failure
        surfaced on the executor thread with nobody left to tell.

        Two different questions are asked about a port, from two different
        registry fields, because neither field can answer both. ``requires``
        lists the keywords a caller must supply, so it judges a *missing* port -
        ``cosmos3`` dials a server yet defaults its port, so it is absent there
        and a caller may legally omit one. ``config_keys`` lists the keywords the
        provider understands, so it judges a *supplied* port: a provider outside
        that set is handed a keyword it never declared, which
        :func:`~strands_robots.registry.policies.provider_reads_a_port` reports.

        ``None`` is the "not supplied" spelling and is refused here for the same
        reason it is refused in :meth:`_get_policy`: without a pre-built
        ``policy_object`` there is nothing to build a policy from. Every other
        value is checked against
        :func:`~strands_robots.utils.tcp_port_error`, the shared domain whose
        docstring already names "the policy providers that dial one (``groot``,
        ``moveit2``, ``cosmos3``, ``lerobot_async``)" - the very
        providers this path forwards to - so the same port cannot be accepted by
        the arm's task entry points and refused by the provider they hand it to.

        That domain also names the value the caller supplied. A supplied-but-
        unusable ``0`` / ``False`` used to be reported as
        ``"policy_port is required for robot operation"``: falsy, so
        :meth:`_get_policy` read it as absent and told the caller a port they
        had passed was missing.

        Args:
            policy_port: The caller-supplied port to validate, or ``None`` when
                none was supplied.
            method: Public entry point name, used to prefix the message.
            policy_provider: The provider the port would be handed to, named in
                the refusal - a missing port is only that provider's problem,
                and which one asked for it decides the caller's next step.

        Returns:
            A tool-shaped error dict naming ``policy_port``, or ``None`` when a
            policy can be built from the value.
        """
        from strands_robots.registry.policies import port_reading_providers, provider_reads_a_port

        if policy_port is None:
            # #13: port-less providers (mock, lerobot_local - registry
            # "requires" without "port") legally build with no port; only
            # server-dialing providers refuse None here.
            if policy_provider:
                try:
                    from strands_robots.registry.policies import get_policy_provider

                    spec = get_policy_provider(policy_provider)
                    if spec is not None and "port" not in (spec.get("requires") or ()):
                        return None
                except Exception:  # noqa: BLE001 - registry read is best-effort
                    pass
            # Name the provider that needs the port - the DEFAULT (groot) is
            # one the caller never chose, so "policy_port is required" read as
            # a fact about the arm - and the way to run with no server at all.
            # The old remedy ("use run_policy with a pre-built policy_object")
            # named a verb this tool does not have; an agent reading it asked
            # the operator for a port instead of picking mock.
            provider_clause = f"policy_provider '{policy_provider}'" if policy_provider else "the policy provider"
            if policy_provider == "groot":
                provider_clause += " (the default)"
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"{method}: policy_port is required - {provider_clause} dials a policy "
                            "server; pass the port it listens on. With no server running, choose a "
                            "provider that builds in process: policy_provider='mock' (sinusoidal test "
                            "motion, no model) or 'lerobot_local' (a local HuggingFace checkpoint)."
                        )
                    }
                ],
            }
        if error := tcp_port_error(policy_port, "policy_port", method):
            return {"status": "error", "content": [{"text": error}]}
        # A port the named provider never declared. ``requires`` above answers
        # "must one be supplied"; ``config_keys`` answers "is one understood",
        # and only the second can judge a port that WAS supplied. Without this,
        # ``_get_policy`` forwarded ``port``/``host`` to every provider: the six
        # whose constructor takes ``**kwargs`` swallowed them, so the rollout ran
        # a policy that never dialed the caller's server and still reported
        # success, and the four that take no ``**kwargs`` raised ``TypeError``
        # from inside ``create_policy`` - on the executor thread, after
        # ``_connect_robot`` had energized the arm and after ``start_task`` had
        # already answered "Task started".
        if provider_reads_a_port(policy_provider) is False:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"{method}: policy_provider={policy_provider!r} declares no policy_port, "
                            f"so policy_port={refusal_repr(policy_port)} would not be read. Drop the port, or name "
                            f"a provider that reads one ({', '.join(port_reading_providers())})."
                        )
                    }
                ],
            }
        return None

    def _shutdown_error(self, method: str) -> dict[str, Any] | None:
        """Refuse a rollout on a robot whose ``cleanup()`` has already run.

        ``_shutdown_event`` is one of the control loop's exit conditions
        (``not self._shutdown_event.is_set()``), so once ``cleanup()`` has set
        it the loop body can never execute. Nothing checked it on the way in,
        though, so a task started afterwards ran the whole bring-up and then
        fell out of the loop on its first evaluation - the same shape as the
        unusable ``duration`` that :meth:`_duration_error` refuses, and it
        reported the same false ``completed``.

        Bring-up is not free of side effects, which is what makes this a
        refusal rather than a cosmetic status fix. Measured on a two-device arm
        after ``cleanup()``:

        - ``_connect_robot()`` re-opens the motors bus and warms every camera,
          and ``cleanup()`` does not disconnect the robot, so the re-opened
          devices stay open for the life of the process. The executor is
          already shut down, so no later ``cleanup()`` can close them either.
        - ``Policy.reset()`` is called, clearing the per-episode state (action
          chunk cache, sampler RNG, KV-cache) of a policy object the caller may
          still be driving elsewhere - the documented
          ``run_policy(policy_object=...)`` reuse pattern.
        - The policy is never queried and the arm is never commanded, yet
          ``run_policy`` and the agent-tool ``execute`` action both returned
          ``status="success"`` with ``steps: 0``, indistinguishable from a
          rollout that really ran.

        ``start_task`` did not report success - it raised
        ``RuntimeError("cannot schedule new futures after shutdown")`` from the
        executor submit, naming an executor internal rather than the robot. The
        guard makes all three entry points refuse in the same tool shape, per
        this module's contract that an action handler returns an error dict
        instead of raising.

        Args:
            method: Public entry point name, used to prefix the message.

        Returns:
            A tool-shaped error naming the shut-down robot, or ``None`` when
            the robot can still accept a rollout.
        """
        if not self._shutdown_event.is_set():
            return None
        return {
            "status": "error",
            "content": [
                {
                    "text": f"{method}: {self.tool_name_str} has been shut down - cleanup() has already run, "
                    f"so the control loop cannot execute a single step. Construct a new robot to drive "
                    f"the arm again."
                }
            ],
        }

    def _claim_task(self, instruction: str) -> dict[str, Any] | None:
        """Claim the motors bus for one rollout, or refuse a concurrent one.

        Admission has to be decided here rather than from
        ``_task_state.status``: that status only becomes ``RUNNING`` once
        ``_execute_task_async`` has finished connecting and building the
        policy, so a status-based check admits every caller that arrives during
        bring-up - a motors-bus handshake plus per-camera warmup, seconds on a
        real arm. Two admitted rollouts then interleave ``send_action`` writes
        on one half-duplex bus, and because they share the single
        ``_task_state`` slot they also overwrite each other's step count and
        terminal status, so both report success while only one of them was
        actually driving the arm.

        The claimed instruction is recorded on the task state immediately, so a
        refusal names the rollout that actually holds the bus instead of
        whichever task ran last.

        Args:
            instruction: Instruction of the rollout being admitted, used for
                the refusal text shown to a second caller.

        Returns:
            ``None`` when the caller now owns the bus and must eventually call
            :meth:`_release_task`, or a tool-shaped error naming the rollout
            already in flight.
        """
        with self._task_admission:
            if self._task_claimed:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Task already running: {self._task_state.instruction}\n"
                            f"{self.tool_name_str} drives one rollout at a time - the arm has a single "
                            f"command bus. Wait for it to finish or call action='stop' first."
                        }
                    ],
                }
            self._task_claimed = True
            self._task_state.instruction = instruction
        return None

    def _release_task(self) -> None:
        """Release the motors-bus claim taken by :meth:`_claim_task`."""
        with self._task_admission:
            self._task_claimed = False

    def _execute_task_sync(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        policy_object: Policy | None = None,
        n_steps: int | None = None,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Execute task synchronously in thread - no new event loop.

        This is the chokepoint the agent-tool ``execute`` action and the mesh
        ``execute`` dispatch reach directly, so it validates the peer-supplied
        budget and the policy endpoint, then claims the motors bus, all before
        the arm is commanded.

        Args:
            instruction: Natural-language instruction passed to the policy.
            policy_port: Port of the policy server to query. Required unless
                ``policy_object`` is given, and validated on the shared
                :func:`~strands_robots.utils.tcp_port_error` domain before the
                arm is connected - see :meth:`_policy_port_error`.
            policy_host: Host of the policy server.
            policy_provider: Provider name used to build the policy.
            duration: Wall-clock budget in seconds; positive and finite.
            policy_object: A pre-built policy. When given, the provider/port
                pair is not read and ``policy_port`` is not validated.
            n_steps: Optional cap on applied actions; ``None`` for no cap.

        Returns:
            Tool-shaped result for the finished rollout, or an error naming the
            offending parameter.
        """
        # Validated here as well as at the public methods below: a peer-supplied
        # budget is refused before the arm is commanded rather than after. The
        # check is stateless, so it runs before the claim - a rejected budget
        # must not take the bus away from a rollout that could still start.
        if err := self._shutdown_error("execute_task"):
            return err
        if err := self._duration_error(duration, "execute_task"):
            return err
        if err := self._n_steps_error(n_steps, "execute_task"):
            return err
        # Same reasoning one parameter over: a port no policy can be built from
        # is refused before the arm is connected rather than after. Gated on
        # ``policy_object``, because a pre-built policy makes the port inert -
        # ``_execute_task_async`` never reads it on that path - and refusing a
        # value the call ignores would be a false rejection.
        # Gated on ``policy_object`` for the same reason the port is: a
        # pre-built policy is never resolved from the provider name, so
        # refusing an unresolvable one would be a false rejection.
        if policy_object is None and (err := self._policy_provider_error(policy_provider, "execute_task")):
            return err
        if policy_object is None and (err := self._policy_port_error(policy_port, "execute_task", policy_provider)):
            return err
        if policy_object is None and (
            err := self._policy_requires_error(policy_provider, policy_kwargs, "execute_task")
        ):
            return err
        if err := self._claim_task(instruction):
            return err

        return self._drive_claimed_task(
            instruction,
            policy_port,
            policy_host,
            policy_provider,
            duration,
            policy_object=policy_object,
            n_steps=n_steps,
            **policy_kwargs,
        )

    def _drive_claimed_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        policy_object: Policy | None = None,
        n_steps: int | None = None,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the control loop for an already-admitted task and report it.

        The caller must already hold the claim from :meth:`_claim_task`; this
        method always releases it, including when the rollout raises, so a
        failed task never leaves the robot permanently refusing new ones. The
        claim is taken by the callers rather than here because ``start_task``
        returns before its executor job begins: it has to be able to refuse a
        second caller synchronously instead of reporting a task as started and
        only then turning that caller away on the background thread.
        """
        try:
            return self._run_control_loop(
                instruction,
                policy_port,
                policy_host,
                policy_provider,
                duration,
                **policy_kwargs,
                policy_object=policy_object,
                n_steps=n_steps,
            )
        finally:
            self._release_task()

    def _run_control_loop(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        policy_object: Policy | None = None,
        n_steps: int | None = None,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Drive the rollout on its own event loop and report the outcome."""

        # Run task without creating new event loop - let it run in thread
        async def task_runner() -> None:
            await self._execute_task_async(
                instruction,
                policy_port,
                policy_host,
                policy_provider,
                duration,
                policy_object=policy_object,
                n_steps=n_steps,
                **policy_kwargs,
            )

        # Probe for a running loop separately from dispatching onto one. An
        # ``except RuntimeError`` wrapped around both answers failures it cannot
        # serve: ``stream(action="execute")`` reaches this from a running loop,
        # so the nested branch below is that surface's live path, and a
        # ``RuntimeError`` from it (a thread the pool cannot start, an executor
        # already shut down) used to land in the handler - whose own
        # ``asyncio.run`` is invalid by construction on exactly that branch. The
        # caller was told "asyncio.run() cannot be called from a running event
        # loop" instead of the cause, and the ``task_runner`` coroutine the
        # handler built was left un-awaited. Only the probe's own
        # ``RuntimeError`` means "no loop is running", so only it is caught.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_a_running_loop = False
        else:
            on_a_running_loop = True

        if on_a_running_loop:
            # Already on a loop: ``asyncio.run`` would refuse, so the rollout
            # gets its own loop on a worker thread. A failure here propagates
            # with its own cause rather than being retried on this thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(task_runner())).result()
        else:
            # No event loop running - safe to create one.
            asyncio.run(task_runner())

        # Return final status. A provider that built in process has no
        # server; "on localhost:None" described one anyway.
        if policy_object is not None:
            policy_desc = f"{type(policy_object).__name__} (pre-built object)"
        elif policy_port is None:
            policy_desc = f"{policy_provider} (built in process, no server)"
        else:
            policy_desc = f"{policy_provider} on {policy_host}:{policy_port}"
        # The policy the loop drove, for the instruction notice: the pre-built
        # object when given, else the one ``_get_policy`` built for the task.
        driven = policy_object if policy_object is not None else self._task_state.policy
        instruction_notice = instruction_not_read_notice(driven)
        return {
            "status": "success" if self._task_state.status == TaskStatus.COMPLETED else "error",
            "content": [
                {
                    "text": f"Task: '{instruction}' - {self._task_state.status.value}\n"
                    f"Robot: {self.tool_name_str} ({self.robot})\n"
                    f"Policy: {policy_desc}\n"
                    f"Duration: {self._task_state.duration:.1f}s\n"
                    f"Steps: {self._task_state.step_count}"
                    + (
                        f"\n{self._task_message_label}: {self._task_state.error_message}"
                        if self._task_state.error_message
                        else ""
                    )
                    + (f"\n{instruction_notice}" if instruction_notice else "")
                }
            ],
        }

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Start robot task asynchronously and return immediately.

        Args:
            instruction: Natural-language instruction passed to the policy.
            policy_port: Port of the policy server to query. Required when
                ``policy_provider`` dials one: this entry point takes no
                pre-built policy, so the port is the only thing a policy can be
                built from. A provider that builds in process declares no port,
                and supplying one anyway is refused rather than forwarded and
                dropped. Validated on the shared
                :func:`~strands_robots.utils.tcp_port_error` domain before the
                task is submitted, so a port no policy can be built from is
                reported here instead of as a started task that connects the
                arm and then fails on the executor thread.
            policy_host: Host of the policy server.
            policy_provider: Provider name used to build the policy.
            duration: Wall-clock budget in seconds. Must be positive and
                finite; it is validated before the task is submitted, so a
                budget the loop cannot honor is reported here instead of as a
                started task that commands nothing (or never ends).
            **policy_kwargs: Checkpoint/provider keywords (``model_path``,
                ``policy_type``, ``pretrained_name_or_path``,
                ``server_address``, ...) forwarded to ``create_policy`` via
                :meth:`_get_policy`. This is the vocabulary the mesh dispatch
                collects from the wire command; per-provider validity is
                ``create_policy``'s contract, so an unknown keyword is the
                provider's own refusal to make.

        Returns:
            Tool-shaped result confirming the task started, or an error naming
            the offending parameter. A second task is refused while another
            rollout is in flight - including during its connect/policy-build
            bring-up - because the arm has a single command bus. A robot that
            ``cleanup`` / ``stop`` has shut down is refused permanently: the
            executor and bridges are gone, so a rollout started now would
            command the arm zero times.
        """
        # Before the submit: the work happens on a background thread, so a
        # budget checked inside it would still report "Task started" to the
        # caller for a task that commands nothing (or never ends).
        # Checked before the budget and before the claim: once ``cleanup()`` has
        # run, the executor submit below raises ``RuntimeError`` from inside
        # concurrent.futures, which names an executor internal rather than the
        # robot. Refusing here reports it in the same tool shape as the other
        # two entry points.
        if err := self._shutdown_error("start_task"):
            return err
        if err := self._duration_error(duration, "start_task"):
            return err
        # Unconditional here, unlike ``_execute_task_sync``: this entry point
        # takes no ``policy_object``, so the port is always the only thing a
        # policy can be built from. Pre-fix every unusable value returned
        # "Task started" and failed on the executor thread after the arm was
        # already connected.
        # Unconditional here for the same reason the port check is: this
        # entry point takes no ``policy_object``, so the provider name is
        # always what the policy is resolved from.
        if err := self._policy_provider_error(policy_provider, "start_task"):
            return err
        if err := self._policy_port_error(policy_port, "start_task", policy_provider):
            return err
        if err := self._policy_requires_error(policy_provider, policy_kwargs, "start_task"):
            return err

        # Claim the bus here, not on the executor thread: this method returns
        # before its job begins, so a claim taken inside the job would report
        # "Task started" to a second caller and only then turn it away, with
        # nobody left to tell.
        if err := self._claim_task(instruction):
            return err

        # Start task in background. The claim is released by
        # ``_drive_claimed_task`` when the rollout ends; if the submit itself
        # fails (a shut-down executor) nothing would ever run to release it.
        try:
            self._task_state.task_future = self._executor.submit(
                functools.partial(self._drive_claimed_task, **policy_kwargs),
                instruction,
                policy_port,
                policy_host,
                policy_provider,
                duration,
            )
        except BaseException:
            self._release_task()
            raise

        # The policy is built on the executor thread after this returns, so
        # the class the registry maps the provider to is what can be asked
        # whether the instruction just echoed will be read at all.
        start_notice = self._pending_instruction_notice(policy_provider)
        return {
            "status": "success",
            "content": [
                {
                    "text": f"Task started: '{instruction}'\n"
                    f"Robot: {self.tool_name_str}\n"
                    + (f"{start_notice}\n" if start_notice else "")
                    + "Use action='status' to check progress\n"
                    "Use action='stop' to interrupt"
                }
            ],
        }

    @staticmethod
    def _pending_instruction_notice(policy_provider: str | None) -> str | None:
        """The instruction-not-read notice for a provider that has not been built yet."""
        return instruction_not_read_notice(provider_policy_class(policy_provider), pending=True)

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run a pre-built policy object on the real robot (blocking).

        Hardware counterpart of ``Simulation.run_policy(policy_object=...)``:
        drive a policy already constructed in-process (e.g. via
        ``create_policy(...)`` around a local checkpoint) without standing up
        a policy server on a port. Reuses the exact ``start_task`` control
        loop - connect (with half-open rollback), state-key initialization,
        the RTC control-frequency / observed-delay contract, and
        ``resolve_chunk_length`` chunk consumption - so a pre-built object
        and a server-backed provider behave identically on the wire.

        Blocking: returns when ``duration`` elapses, after ``n_steps``
        applied actions, or when ``stop_task()`` is called from another
        thread. For the server-backed provider path (and for fire-and-forget
        execution) use ``start_task``.

        Exclusive: the arm has a single command bus, so this is refused while
        another rollout is in flight - including while that rollout is still
        connecting or building its policy, before it reports ``RUNNING``.
        Two rollouts admitted at once would interleave ``send_action`` writes
        on one half-duplex bus and share the single task-state slot.

        Terminal after shutdown: ``cleanup`` / ``stop`` release the task
        executor, the mesh and the ROS bridges and cannot be undone, so a
        rollout requested afterwards is refused rather than admitted to command
        the arm zero times. A rollout already running when the shutdown lands is
        reported ``STOPPED`` - a shutdown truncates a task exactly as
        ``stop_task`` does, so its step count is a partial one.

        Args:
            policy_object: A constructed ``Policy`` instance. The object's
                own device / embodiment / chunking configuration is honored;
                the loop only injects the robot state keys and the RTC
                control rate, exactly as it does for server-backed policies.
                Its per-episode state is cleared via ``Policy.reset`` at the
                start of every task, so one object can be reused across tasks
                without the previous task's cached action chunk being driven.
            instruction: Natural-language instruction passed to the policy on
                every ``get_actions`` call.
            duration: Wall-clock budget in seconds (same default as
                ``start_task``). Must be positive and finite; the loop bounds
                the rollout by it even when ``n_steps`` is given, so a value it
                cannot honor is refused rather than reported as a rollout that
                commanded nothing.
            n_steps: Optional cap on applied actions (mirrors the sim
                ``run_policy`` parameter); the loop stops at whichever of
                ``duration`` / ``n_steps`` comes first. ``None`` (the
                default) requests no cap and leaves ``duration`` the sole
                horizon. Any other value must be a positive integer, the
                domain the sim horizon and this loop's ``action_horizon``
                already share: the loop bounds the rollout by
                ``step_count < n_steps``, so a cap it cannot count against
                is refused rather than spent on the arm.

        Returns:
            Tool-shaped result: a text summary plus a ``{"json": ...}`` block
            carrying ``status`` / ``steps`` / ``duration_s`` / ``instruction``
            / ``policy`` (and ``error`` when one occurred), or an error naming
            the rollout already in flight.
        """
        if policy_object is None:
            return {
                "status": "error",
                "content": [{"text": "policy_object is required (for the provider+port path use start_task)"}],
            }
        if err := self._shutdown_error("run_policy"):
            return err
        if err := self._duration_error(duration, "run_policy"):
            return err
        if err := self._n_steps_error(n_steps, "run_policy"):
            return err
        if err := self._claim_task(instruction):
            return err

        self._drive_claimed_task(instruction, duration=duration, policy_object=policy_object, n_steps=n_steps)

        succeeded = self._task_state.status == TaskStatus.COMPLETED
        summary = (
            f"Policy rollout {self._task_state.status.value}: "
            f"{self._task_state.step_count} steps in {self._task_state.duration:.1f}s "
            f"({type(policy_object).__name__} on {self.tool_name_str})"
        )
        payload: dict[str, Any] = {
            "status": self._task_state.status.value,
            "steps": self._task_state.step_count,
            "duration_s": round(self._task_state.duration, 3),
            "instruction": instruction,
            "policy": type(policy_object).__name__,
        }
        if self._task_state.error_message:
            payload["error"] = self._task_state.error_message
            summary += f"\nError: {self._task_state.error_message}"
        return {
            "status": "success" if succeeded else "error",
            "content": [{"text": summary}, {"json": payload}],
        }

    # Verbs an agent reaches for on a real arm that this tool's enum does not
    # publish, mapped to where each one lives. The quickstart once asked this
    # tool for three of them; the generic "Unknown action" left the agent to
    # guess whether the verb was gone, renamed, or in another tool.
    #
    # Two of the three destinations are elsewhere - the recording pair are the
    # simulation tool's actions, teleoperation is the lerobot_teleoperate tool -
    # and the third is this tool itself: run_policy/start_policy/stop_policy are
    # the simulation tool's spellings for execute/start/stop, so those refusals
    # send the caller back here rather than away.
    _ELSEWHERE_ACTIONS: dict[str, str] = {
        "teleoperate": "teleoperation",
        "start_teleop": "teleoperation",
        "stop_teleoperate": "teleoperation",
        "record": "recording",
        "start_recording": "recording",
        "stop_recording": "recording",
        "record_episode": "recording",
        "run_policy": "policy",
        "start_policy": "policy",
        "stop_policy": "policy",
    }

    def _unknown_action_text(self, action: Any) -> str:
        """The refusal for an action this tool does not have.

        Names every verb the real robot tool does have - the observe ones as
        well as the motion ones, read from the same tuple the ``action`` enum
        is built from - and for the
        spellings an agent is known to reach for it also names where that verb
        lives: the ``lerobot_teleoperate`` tool for teleoperation, the
        simulation tool for dataset recording, and this tool's own
        ``execute``/``start``/``stop`` for the simulation tool's policy verbs.
        So an agent's next call is the right verb rather than another spelling
        of the wrong one.

        ``action`` is rendered through :func:`~strands_robots.utils.refusal_str`
        rather than interpolated: a Python caller of :meth:`stream` supplies it,
        and a value whose ``__str__`` raises would make building this refusal
        raise instead of answering. ``str`` rather than ``repr`` keeps the text
        an agent reads unquoted, as it has always been.

        Args:
            action: The action the caller sent.

        Returns:
            One paragraph: the refusal, the valid actions, and the remedy when
            the verb is a known one that lives elsewhere - or here under
            another name.
        """
        text = f"Unknown action: {refusal_str(action)}. Valid actions: {', '.join(_PUBLISHED_ACTIONS)}"
        kind = self._ELSEWHERE_ACTIONS.get(action) if isinstance(action, str) else None
        if kind == "teleoperation":
            text += (
                f". {self.tool_name_str} drives policies; it does not teleoperate from an agent. "
                "Leader-arm teleoperation of a real arm is the lerobot_teleoperate tool "
                "(action='start', robot_port=..., teleop_port=...), or from Python "
                "robot.attach_teleop('so101_leader', port=...).teleoperate(duration=...)"
            )
        elif kind == "recording":
            text += (
                f". {self.tool_name_str} drives policies; it does not record datasets. "
                "Recording a real arm under teleoperation is the lerobot_teleoperate tool "
                "(action='start' with dataset_repo_id=..., dataset_root=..., dataset_single_task=...); "
                "start_recording/stop_recording are the simulation tool's actions"
            )
        elif kind == "policy":
            text += (
                f". {self.tool_name_str} does drive policies, under its own verbs: action='execute' "
                "runs one to completion, action='start' runs one in the background and action='stop' "
                "halts it (instruction=..., policy_provider=..., duration=... carry the rollout). "
                "run_policy/start_policy/stop_policy are the simulation tool's spellings; from Python, "
                "robot.run_policy(policy_object=...) drives a policy already built in-process"
            )
        return text

    def _device_facts(self) -> dict[str, Any]:
        """Measured facts about the device, readable before it is connected.

        The device is connected lazily, on the first task, so for most of a
        robot's life an agent asks about it while nothing is open. What can be
        read then: the port it was built for and whether that path exists on
        this host, the cameras it was configured with, and the ``is_connected``
        flag. What cannot: ``is_calibrated`` - lerobot reads it through the
        bus and raises ``DeviceNotConnectedError`` before ``connect()`` - so
        it is ``None`` until the device is connected, rather than a raise that
        turned the whole :meth:`get_status` probe into its error shape for
        every idle arm.

        Every fact here is three-state for the same reason: ``True``/``False``
        is a reading, ``None`` is "this was not readable" - including
        ``is_connected``, which is a probe and not a stored flag. Every shipped
        lerobot arm folds its cameras into it (``self.bus.is_connected and
        all(cam.is_connected for cam in self.cameras.values())``, 8 drivers in
        lerobot 0.6.2), so the camera whose probe raises - the one this method
        already tolerates per-camera below - arrives through that aggregate
        first. Read unguarded it took the whole probe down with it: the port, a
        filesystem read that cannot raise, was discarded too, and
        :meth:`get_status` answered ``is_connected: False`` for an arm that was
        connected and driving.

        A port is not always a device path - lerobot's network drivers carry a
        TCP port (``Reachy2RobotConfig.port`` is ``50065``, reached at its
        ``ip_address``), and there is nothing on this host to stat for one, so
        ``port_present`` stays ``None`` and must not be reported as either
        answer.

        Returns:
            ``port`` (``None`` when the driver has no port), ``port_present``
            (``True``/``False`` from the filesystem for a path port; ``None``
            when the port is not a path, so presence was never read),
            ``address`` (the host a network port is reached at, ``None`` when
            the config names none), ``is_connected`` (``None`` when the
            driver's probe raised, so it was not read), ``is_calibrated``
            (``None`` until connected), ``cameras`` (the configured names) and
            ``cameras_connected`` (per live camera, best-effort - a camera
            whose probe raises is omitted).
        """
        robot = self.robot
        try:
            is_connected: bool | None = bool(getattr(robot, "is_connected", False))
        except Exception as e:  # noqa: BLE001 - an unreadable flag is None, not a probe that fails
            logger.debug("%s could not read is_connected: %s", self.tool_name_str, e)
            is_connected = None
        config = getattr(robot, "config", None)
        port = getattr(config, "port", None)
        port_present: bool | None = None
        if isinstance(port, str) and port.startswith("/"):
            port_present = os.path.exists(port)
        # A port that is not a device path is a network port, and the number
        # alone does not say where. lerobot spells the host differently per
        # driver family, so the first one the config carries wins.
        address: str | None = None
        for field in _ADDRESS_FIELDS:
            value = getattr(config, field, None)
            if isinstance(value, str) and value:
                address = value
                break
        is_calibrated: bool | None = None
        if is_connected:
            is_calibrated = bool(getattr(robot, "is_calibrated", True))
        cameras: list[str] = []
        configured = getattr(config, "cameras", None)
        if isinstance(configured, dict):
            cameras = list(configured.keys())
        cameras_connected: dict[str, bool] = {}
        live_cameras = getattr(robot, "cameras", None)
        if isinstance(live_cameras, dict):
            for name, camera in live_cameras.items():
                try:
                    cameras_connected[name] = bool(camera.is_connected)
                except Exception as e:  # noqa: BLE001 - a camera that cannot answer is omitted, not fatal
                    logger.debug("camera %r on %s could not report is_connected: %s", name, self.tool_name_str, e)
        return {
            "port": port,
            "port_present": port_present,
            "address": address,
            "is_connected": is_connected,
            "is_calibrated": is_calibrated,
            "cameras": cameras,
            "cameras_connected": cameras_connected,
        }

    @staticmethod
    def _device_lines(facts: dict[str, Any]) -> str:
        """The device facts as the lines ``status`` prints under the task state.

        Args:
            facts: The mapping :meth:`_device_facts` returns.

        Returns:
            Newline-terminated lines: the connection, the port and the
            cameras. Each reports what was read - and an unreadable fact says
            so rather than borrowing the negative answer: a connection that
            could not be probed is not "not connected", and a camera missing
            from ``cameras_connected`` is not "not connected" either. The port
            line reports present, absent (with the remedy), or, for a network
            port, that presence on this host was never a fact to read.
        """
        if facts["is_connected"]:
            calibrated = "calibrated" if facts["is_calibrated"] else "NOT calibrated"
            device = f"Device: connected on {facts['port']} ({calibrated})"
        elif facts["is_connected"] is None:
            device = (
                "Device: whether the bus is open could not be read - the driver's is_connected "
                "probe raised, so neither answer would be a reading. A lerobot arm reads it as the "
                "bus AND every camera, so one camera that cannot answer lands here; the cameras "
                "line names which one."
            )
        else:
            device = "Device: not connected (the bus is opened by the first task)"
        lines = [device]
        if facts["port_present"] is False:
            lines.append(
                f"Port: {facts['port']} is not present on this host - no serial device answers to that path, "
                "so the first task will fail to connect. Check the cable and power; scan_serial_devices() "
                "lists what this host does see."
            )
        elif facts["port"] is not None and not facts["is_connected"]:
            if facts["port_present"]:
                lines.append(f"Port: {facts['port']} is present on this host")
            else:
                # port_present is None: the port is not a device path, so
                # nothing on this host was stat-ed. Reporting it as present
                # would be the defect this method exists to fix - a sentence
                # an agent cannot tell from a reading.
                where = f", reached at {facts['address']}" if facts["address"] else ""
                lines.append(
                    f"Port: {facts['port']} is a network port{where}, not a device path - this host has no "
                    "such path to check, so neither answer about it would be a reading."
                )
        if facts["cameras"]:
            states = facts["cameras_connected"]
            named = ", ".join(
                f"{n} ({'connected' if states[n] else 'not connected'})" if n in states else f"{n} (could not be read)"
                for n in facts["cameras"]
            )
            lines.append(f"Cameras: {named}")
        else:
            lines.append("Cameras: none configured")
        return "\n".join(lines) + "\n"

    def get_task_status(self) -> dict[str, Any]:
        """Report the task state, then the device it would drive.

        The task machine used to be the whole answer: an arm whose port does
        not exist on this host read ``Robot Status: IDLE``, byte-identical to
        a connected, healthy arm at rest, and the difference only surfaced as
        a connect failure inside the first task. The measured facts were
        already gathered by :meth:`get_status` for Python callers; the tool
        action now prints the same facts under the task state and carries
        them as a ``json`` block.

        Returns:
            ``status=success`` with the task state on the first line (as
            before), the device lines from :meth:`_device_facts` after it,
            and a second content block holding those facts as JSON.
        """
        # Update duration for running tasks
        if self._task_state.status == TaskStatus.RUNNING:
            self._settle_task_duration()

        status_text = f"Robot Status: {self._task_state.status.value.upper()}\n"

        if self._task_state.instruction:
            status_text += f"Task: {self._task_state.instruction}\n"

        if self._task_state.status == TaskStatus.RUNNING:
            status_text += f"Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"Steps: {self._task_state.step_count}\n"
        elif self._task_state.status in [TaskStatus.COMPLETED, TaskStatus.STOPPED, TaskStatus.ERROR]:
            status_text += f"Total Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"Total Steps: {self._task_state.step_count}\n"

        if self._task_state.error_message:
            status_text += f"{self._task_message_label}: {self._task_state.error_message}\n"

        # Same sentence the ``execute`` envelope carries, in the present tense
        # while the task runs: a RUNNING / 18 steps report on a policy that
        # never reads the instruction it is filed under otherwise says the task
        # is being performed. Above the device lines, with the task state it
        # qualifies - and so on the path that cannot read the device too.
        if self._task_state.status not in (TaskStatus.IDLE, TaskStatus.CONNECTING):
            notice = instruction_not_read_notice(
                self._task_state.policy, pending=self._task_state.status == TaskStatus.RUNNING
            )
            if notice:
                status_text += f"{notice}\n"

        try:
            facts = self._device_facts()
        except Exception as e:  # noqa: BLE001 - the task state must still be reported
            logger.debug("%s device facts unavailable: %s", self.tool_name_str, e)
            return {"status": "success", "content": [{"text": status_text}]}
        status_text += self._device_lines(facts)
        return {
            "status": "success",
            "content": [{"text": status_text}, {"json": facts}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Stop the current task, including one that is still connecting.

        This is the interrupt an operator (or the fleet ``{"action": "stop"}``
        dispatch, via :class:`~strands_robots.mesh.core.Mesh`) reaches for, so
        it has to hold for a
        task in ANY stage that can still command the arm - not only the one
        stage whose status happens to be ``RUNNING``.

        ``_execute_task_async`` sits in ``CONNECTING`` for the whole hardware
        bring-up: a motors-bus handshake plus ``warmup_s`` per camera, seconds
        on a real arm and longer on a multi-camera rig, followed by the policy
        build. A stop pressed in that window used to be answered with
        ``status="success"`` and ``"No task running to stop"``, and the arm then
        moved anyway once the bring-up finished - the operator was told the
        interrupt was handled while the rollout it was meant to cancel was
        still pending.

        A status write cannot express the request on its own, because
        ``_execute_task_async`` writes ``RUNNING`` once bring-up completes and
        that overwrites any ``STOPPED`` recorded before it. So the request is
        latched in an event that is set here FIRST, before the status is even
        read, and cleared only when a new task starts. The rollout honors the
        latch at each stage boundary and in its loop condition.

        Returns:
            A tool-shaped result confirming the stop, or - for a robot that is
            genuinely idle or already in a terminal state - reporting that
            there was nothing to stop. Both are ``status="success"``: asking an
            idle robot to stop is satisfied, not an error.
        """
        # Latched before the status is read: a stop that lands in the gap
        # between the rollout's last stage check and its ``RUNNING`` write must
        # not be lost, and the latch is the only record that survives it.
        self._stop_requested.set()

        stoppable = (TaskStatus.RUNNING, TaskStatus.CONNECTING)
        if self._task_state.status not in stoppable:
            return {
                "status": "success",
                "content": [{"text": f"No task running to stop (current: {self._task_state.status.value})"}],
            }

        was_connecting = self._task_state.status == TaskStatus.CONNECTING

        # Signal task to stop
        self._task_state.status = TaskStatus.STOPPED
        self._settle_task_duration()

        # Cancel future if it exists
        if self._task_state.task_future:
            self._task_state.task_future.cancel()

        logger.info(f"Task stopped: {self._task_state.instruction}")

        stage = " (during connect)" if was_connecting else ""
        return {
            "status": "success",
            "content": [
                {
                    "text": f"Task stopped{stage}: '{self._task_state.instruction}'\n"
                    f"Duration: {self._task_state.duration:.1f}s\n"
                    f"Steps completed: {self._task_state.step_count}"
                    + (
                        f"\n{stop_notice}"
                        if (stop_notice := instruction_not_read_notice(self._task_state.policy))
                        else ""
                    )
                }
            ],
        }

    @property
    def tool_name(self) -> str:
        """The Strands agent-tool name this robot registers itself under."""
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        """The Strands tool category for this device (always ``"robot"``)."""
        return "robot"

    @property
    def tool_spec(self) -> ToolSpec:
        """Get tool specification with async actions."""
        # The first sentence is the first thing an agent does with the tool, so
        # it leads with what can be learned for free. Before the observe actions
        # existed the only verbs were the motion ones, and an agent asked to
        # "read the joint positions, do not move" requested a ten-second mock
        # policy rollout on the real arm to do the reading.
        port = getattr(getattr(self.robot, "bus", None), "port", None)
        where = f" on {port}" if port else ""
        return {
            "name": self.tool_name_str,
            "description": (
                f"Drive the real robot {self.robot}{where} with a policy, or read it for free. "
                "Observe (no approval, writes nothing): get_state (alias get_robot_state) = joint "
                "degrees, ticks, torque, voltage; list_cameras; render = a camera frame. "
                "Motion: execute (blocking), start (async), status, stop. execute/start pause for "
                "operator approval before the arm moves (a headless script pre-approves with "
                f"{COMMAND_ALLOW_ENV}=execute,start) and run for at most duration seconds (default 30). "
                "They need instruction; the default provider groot also needs policy_port, while "
                "mock and lerobot_local build in process with no server - mock ignores the instruction, "
                "and lerobot_local needs pretrained_name_or_path. No set_joint_positions/move_to "
                "here: to read the arm call get_state, never execute."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": (
                                "get_state | get_robot_state | list_cameras | render (observe, ungated); "
                                "execute | start (motion, operator approval); status | stop"
                            ),
                            "enum": list(_PUBLISHED_ACTIONS),
                            "default": "get_state",
                        },
                        "camera_name": {
                            "type": "string",
                            "description": "render: which configured camera (see list_cameras). Optional when there is exactly one.",
                        },
                        "output_path": {
                            "type": "string",
                            "description": (
                                "render: PNG destination inside the render sandbox (~/.strands_robots/renders or "
                                "STRANDS_ROBOTS_RENDER_ROOT); default <robot>-<camera>-<ms>.png."
                            ),
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction (required for execute/start actions)",
                        },
                        "policy_port": {
                            "type": "integer",
                            "description": "Policy service port. Required by groot and moveit2, read by the other server-dialing providers, refused for providers that build in process (mock, lerobot_local).",
                        },
                        "policy_host": {
                            "type": "string",
                            "description": "Policy service host (default: localhost)",
                            "default": "localhost",
                        },
                        "policy_provider": {
                            "type": "string",
                            "description": (
                                "Which policy backend runs: one of cosmos3, curobo, groot, kimodo, "
                                "lerobot_async, lerobot_local, microduck, mock, moveit2, protomotions, "
                                "remote, rl, wbc, wbc_gait. "
                                "groot (default, needs policy_port) and moveit2 dial a server; "
                                "lerobot_local runs a local checkpoint in process and needs "
                                "pretrained_name_or_path; mock is a test motion on every joint that "
                                "ignores the instruction. The model itself is named in policy_config. "
                                "Not an action - an unknown name is refused listing the current registry."
                            ),
                            "default": "groot",
                        },
                        "duration": {
                            "type": "number",
                            "description": (
                                "Maximum execution time in seconds. Must be a positive finite "
                                "number: the loop compares elapsed time against it, so 0 or a "
                                "negative value commands nothing and infinity never ends."
                            ),
                            "default": 30.0,
                        },
                    },
                    "required": ["action"],
                }
            },
        }

    @staticmethod
    def _make_tool_result(tool_use_id: str, result: dict[str, Any]) -> ToolResult:
        """Create a ToolResult dict with the given tool_use_id merged into result."""
        return cast(ToolResult, {"toolUseId": tool_use_id, **result})

    def _dashboard_grant(self, tool_input: Mapping[str, Any]) -> bool:
        """Spend a grant the dashboard's motion hook deposited for this exact call.

        The dashboard registers :class:`~strands_robots.dashboard.agent_hitl.MotionInterruptHook`
        on its agent, which asks the operator before the tool runs and records a
        one-shot grant keyed on what they were shown. Asking again here would be
        the same question twice, so a grant is consumed and the call proceeds.
        The dashboard extra may be absent, and a missing module must read as
        "no grant", never as a crash: the gate below then asks the operator.

        Args:
            tool_input: The call as the hook saw it - the tool's own input dict.

        Returns:
            True when a grant for this exact call existed and was spent.
        """
        try:
            from strands_robots.dashboard import agent_hitl
        except ImportError:
            return False
        return bool(agent_hitl.consume_grant(self.tool_name_str, tool_input))

    def _pre_gate_error(
        self, action: str, policy_port: Any, policy_provider: str, duration: Any
    ) -> dict[str, Any] | None:
        """The input checks that decide a command's fate with no operator and no hardware.

        ``execute``/``start`` ask the operator before dispatch, and the
        dispatcher (:meth:`execute_task` / :meth:`start_task`) then checks
        the inputs. Every check here is a pure function of the call and of
        this object - a shut-down robot, a duration that is not a positive
        number, a ``policy_port`` outside 1-65535 or missing for a provider
        that needs one - so a call that fails one was never going to move
        the arm. Asking first would spend an approval on nothing and leave
        the operator reading an error under the "y" they just typed, with
        the agent's corrected retry costing a second round. Run them before
        the gate; the dispatcher runs them again, which is defence in depth,
        not a second answer.

        Args:
            action: ``"execute"`` or ``"start"``; names the dispatcher the
                refusal speaks for.
            policy_port: As supplied by the tool input.
            policy_provider: As supplied by the tool input.
            duration: As supplied by the tool input.

        Returns:
            The dispatcher's error envelope, or None when the call reaches
            the operator.
        """
        method = "execute_task" if action == "execute" else "start_task"
        if err := self._shutdown_error(method):
            return err
        if err := self._duration_error(duration, method):
            return err
        # Before the port check, which cannot judge a port for a provider it
        # cannot resolve - see _policy_provider_error.
        if err := self._policy_provider_error(policy_provider, method):
            return err
        return self._policy_port_error(policy_port, method, policy_provider)

    def _gate_motion(
        self, action: str, tool_input: Mapping[str, Any], tool_use: ToolUse, invocation_state: Mapping[str, Any]
    ) -> str | None:
        """Operator approval for one ``execute``/``start``, before it is dispatched.

        The warning is the only description of the motion an operator reads
        before a real arm moves, so it says how long the arm may move and, when
        the policy about to be built never reads the instruction, that the words
        will not shape the motion. It read ``drives the real robot 'so101' with
        'Wave the arm'`` with no budget and nothing else: with ``mock`` the arm
        does not wave - every joint follows a sinusoid whatever the task says -
        so the operator approved a motion they were not told about, for an
        unstated length of time.

        An ``AgentTool`` receives no ``tool_context`` argument; the SDK builds
        one from the invoking agent for decorated tools, and this builds the
        same object from the same two inputs so the shared gate can raise the
        same interrupt. With no agent in ``invocation_state`` (a direct call,
        a headless script) there is no operator to ask and the gate refuses.

        Args:
            action: ``"execute"`` or ``"start"``.
            tool_input: The tool's input dict, shown to the operator and used to
                match a dashboard grant.
            tool_use: The tool-use request carrying ``toolUseId``.
            invocation_state: The agent runtime's kwargs; ``"agent"`` when the
                call came through an :class:`strands.Agent`.

        Returns:
            A refusal message, or None to let the dispatch proceed.

        Raises:
            InterruptException: When the operator has not answered yet; the
                caller turns it into a ``ToolInterruptEvent`` exactly as the
                SDK does for a decorated tool.
        """
        if self._dashboard_grant(tool_input):
            return None
        agent = invocation_state.get("agent")
        tool_context: ToolContext | None = None
        if agent is not None:
            tool_context = ToolContext(tool_use=tool_use, agent=agent, invocation_state=dict(invocation_state))
        instruction = str(tool_input.get("instruction", ""))
        provider = tool_input.get("policy_provider", "groot")
        host = tool_input.get("policy_host", "localhost")
        port = tool_input.get("policy_port")
        duration = tool_input.get("duration", 30.0)
        # How long the arm may move, and whether the words the operator is
        # reading will shape that motion at all. The budget is the horizon the
        # control loop compares against (see :meth:`_duration_error`, which has
        # already refused anything but a positive finite number by the time the
        # gate runs); the notice is the one ``start`` and a RUNNING ``status``
        # carry, in the same words, so what the operator approves is what the
        # envelopes will report.
        budget = (
            f" for up to {duration:g}s" if isinstance(duration, int | float) and not isinstance(duration, bool) else ""
        )
        notice = self._pending_instruction_notice(provider)
        # ``tool`` is the fixed word "robot" so the interrupt id and the audit
        # source read the same for every robot; the target names which one.
        return gate_motion(
            "robot",
            action,
            self.tool_name_str,
            f"{action!r} drives the real robot {self.tool_name_str!r}{budget} with {instruction!r} "
            f"({self._policy_description(provider, host, port)}); "
            "it needs operator approval before it is dispatched." + (f" {notice}" if notice else ""),
            tool_context,
            allow_env=COMMAND_ALLOW_ENV,
            allow_match=lambda allowed: "*" in allowed or action in allowed,
        )

    def _observe(self, action: str, tool_input: Mapping[str, Any]) -> dict[str, Any]:
        """Answer one observe action: a read of the arm, never a write to it.

        ``get_state``/``get_robot_state`` open the motor bus if it is closed
        (the bus only - not the robot's ``connect()``, whose ``configure()``
        writes servo registers) and read positions, torque and voltage; on an
        arm with no calibration the degrees are the encoder estimate and the
        text says so. ``list_cameras`` opens nothing. ``render`` opens the
        named camera on first use, saves one PNG in the render sandbox and returns
        the frame as an ``image`` block so the model sees it, as the simulation's
        ``render`` does.

        A failure names the port and the remedy rather than the SDK's
        traceback: a read that cannot open the port is the one message a
        developer reads first on a new machine.
        """
        try:
            if action in ("get_state", "get_robot_state"):
                # The ledger is written by the open itself, not by this success
                # path: a read that opens the bus and then raises must still
                # leave "bus" on record, or no connect will hand it back.
                state = hardware_observe.read_joint_state(self.robot, on_open=self._observe_ledger().add)
                state["robot"] = self.tool_name_str
                state["task_status"] = self._task_state.status.value
                return {
                    "status": "success",
                    "content": [
                        {"text": hardware_observe.format_joint_state(self.tool_name_str, state)},
                        {"json": state},
                    ],
                }
            if action == "list_cameras":
                cams = hardware_observe.list_cameras(self.robot)
                return {
                    "status": "success",
                    "content": [
                        {"text": hardware_observe.format_cameras(self.tool_name_str, cams)},
                        {"json": {"robot": self.tool_name_str, "cameras": cams}},
                    ],
                }
            # render
            camera_name = tool_input.get("camera_name")
            output_path = tool_input.get("output_path")
            frame = hardware_observe.capture_frame(
                self.robot,
                None if camera_name is None else str(camera_name),
                None if output_path is None else str(output_path),
                tool_name=self.tool_name_str,
                on_open=self._observe_ledger().add,
            )
            png = frame.pop("png")
            text = (
                f"Saved one frame from camera {frame['camera']!r} to {frame['path']} "
                f"({frame['width']}x{frame['height']}, {frame['channels']} channels, read {frame['read_ms']} ms)."
            )
            # The image block is what lets the model SEE the frame, as the simulation's
            # ``render`` does; raw bytes, not base64 (Bedrock encodes on the wire).
            return {
                "status": "success",
                "content": [
                    {"text": text},
                    {"image": {"format": "png", "source": {"bytes": png}}},
                    {"json": frame},
                ],
            }
        except ValueError as exc:
            # A refusal this module composed: unknown camera, unsafe path, no cameras.
            return {"status": "error", "content": [{"text": f"{self.tool_name_str}: {exc}"}]}
        except Exception as exc:  # noqa: BLE001 - every hardware failure becomes a tool error that names the port
            port = getattr(getattr(self.robot, "bus", None), "port", None)
            lines = str(exc).strip().splitlines()
            reason = lines[-1] if lines else type(exc).__name__
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"{self.tool_name_str}: {action} could not read the arm on {port!r}: {reason} "
                            "Check the arm is powered and the port is right (`lerobot-find-port`, or "
                            "strands_robots._serial_discovery.scan_serial_devices()), and that no other "
                            "process holds the port (`lsof <port>`)."
                        )
                    }
                ],
            }

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent | ToolInterruptEvent, None]:
        """Stream robot task execution with async actions."""
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})

            action = input_data.get("action", "get_state")

            # Handle different actions
            if action in hardware_observe.OBSERVE_ACTIONS:
                # Reads. Serial I/O blocks, so each runs off the event loop;
                # none of them writes a servo register, so none is gated.
                result = await asyncio.to_thread(self._observe, action, input_data)
                yield ToolResultEvent(self._make_tool_result(tool_use_id, result))

            elif action == "execute":
                # Blocking execution (legacy behavior)
                instruction = input_data.get("instruction", "")
                policy_port = input_data.get("policy_port")
                policy_host = input_data.get("policy_host", "localhost")
                policy_provider = input_data.get("policy_provider", "groot")
                duration = input_data.get("duration", 30.0)

                # Only ``instruction`` is judged here. Whether a ``policy_port``
                # is missing, unusable or unread is the named provider's call
                # (``mock`` and ``lerobot_local`` build without one), and the
                # dispatcher below asks :meth:`_policy_port_error` that.
                if not instruction:
                    yield ToolResultEvent(
                        self._make_tool_result(
                            tool_use_id,
                            {
                                "status": "error",
                                "content": [{"text": "instruction is required for execute action"}],
                            },
                        )
                    )
                    return

                # A call the dispatcher would refuse on its inputs alone is
                # refused here, before the operator is asked to approve it.
                if err := self._pre_gate_error(action, policy_port, policy_provider, duration):
                    yield ToolResultEvent(self._make_tool_result(tool_use_id, err))
                    return

                # Ask the operator before anything is dispatched: a refused or
                # unanswered call is exactly as inert as one that never happened.
                try:
                    refusal = self._gate_motion(action, input_data, tool_use, invocation_state)
                except InterruptException as exc:
                    yield ToolInterruptEvent(tool_use, [exc.interrupt])
                    return
                if refusal is not None:
                    yield ToolResultEvent(
                        self._make_tool_result(
                            tool_use_id,
                            {"status": "error", "content": [{"text": f"{self.tool_name_str}: {refusal}"}]},
                        )
                    )
                    return

                # Execute task synchronously
                task_result = self._execute_task_sync(instruction, policy_port, policy_host, policy_provider, duration)
                yield ToolResultEvent(self._make_tool_result(tool_use_id, task_result))

            elif action == "start":
                # Asynchronous execution start
                instruction = input_data.get("instruction", "")
                policy_port = input_data.get("policy_port")
                policy_host = input_data.get("policy_host", "localhost")
                policy_provider = input_data.get("policy_provider", "groot")
                duration = input_data.get("duration", 30.0)

                # Only ``instruction`` is judged here. Whether a ``policy_port``
                # is missing, unusable or unread is the named provider's call
                # (``mock`` and ``lerobot_local`` build without one), and the
                # dispatcher below asks :meth:`_policy_port_error` that.
                if not instruction:
                    yield ToolResultEvent(
                        self._make_tool_result(
                            tool_use_id,
                            {
                                "status": "error",
                                "content": [{"text": "instruction is required for start action"}],
                            },
                        )
                    )
                    return

                if err := self._pre_gate_error(action, policy_port, policy_provider, duration):
                    yield ToolResultEvent(self._make_tool_result(tool_use_id, err))
                    return

                try:
                    refusal = self._gate_motion(action, input_data, tool_use, invocation_state)
                except InterruptException as exc:
                    yield ToolInterruptEvent(tool_use, [exc.interrupt])
                    return
                if refusal is not None:
                    yield ToolResultEvent(
                        self._make_tool_result(
                            tool_use_id,
                            {"status": "error", "content": [{"text": f"{self.tool_name_str}: {refusal}"}]},
                        )
                    )
                    return

                # Start task asynchronously
                start_result = self.start_task(instruction, policy_port, policy_host, policy_provider, duration)
                yield ToolResultEvent(self._make_tool_result(tool_use_id, start_result))

            elif action == "status":
                # Get current task status
                status_result = self.get_task_status()
                yield ToolResultEvent(self._make_tool_result(tool_use_id, status_result))

            elif action == "stop":
                # Stop current task
                stop_result = self.stop_task()
                yield ToolResultEvent(self._make_tool_result(tool_use_id, stop_result))

            else:
                yield ToolResultEvent(
                    self._make_tool_result(
                        tool_use_id,
                        {"status": "error", "content": [{"text": self._unknown_action_text(action)}]},
                    )
                )

        except Exception as e:
            logger.error(f"{self.tool_name_str} error: {e}")
            yield ToolResultEvent(
                self._make_tool_result(
                    tool_use_id,
                    {
                        "status": "error",
                        "content": [{"text": f"{self.tool_name_str} error: {str(e)}"}],
                    },
                )
            )

    def cleanup(self) -> None:
        """Cleanup resources and stop any running tasks.

        Terminal: this latches a shutdown, releases the task executor, tears
        down the mesh and ROS bridges, and disconnects the robot -- the motors
        bus and every camera -- so no device node stays held. The one exception
        is a teleop loop that did not join: the devices are left open rather
        than closed under a live writer, and the reason is recorded at ERROR
        with the remedy. There is no
        ``restart``, so
        ``run_policy`` / ``start_task`` / the ``execute`` action refuse
        permanently afterwards rather than admit a rollout that would command
        the arm zero times, and a rollout still in flight when this runs is
        abandoned at its next stage check -- rather than finishing a bring-up
        this teardown has already made pointless -- and reported ``STOPPED``
        rather than ``COMPLETED``. Construct a new
        ``Robot`` to run another task.
        """
        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Stop any local teleoperation loop + disconnect attached devices
            # (TeleopMixin). Best-effort: a teleop teardown failure must not
            # block the rest of hardware cleanup.
            # ``stop_teleoperate`` reports whether the loop actually joined, and
            # the devices must not be closed under one that did not - see the
            # ``_disconnect_devices`` call below. A raise leaves the outcome
            # unknown and keeps the pre-existing warn-and-continue contract:
            # only a positive report of a live loop defers the close.
            teleop_stopped = True
            if getattr(self, "_teleop_running", False) or getattr(self, "_teleops", None):
                try:
                    teleop_stopped = _stop_reported_stopped(self.stop_teleoperate())
                except Exception as teleop_exc:  # noqa: BLE001
                    logger.warning(
                        "%s: stop_teleoperate() raised during cleanup: %s",
                        self.tool_name_str,
                        teleop_exc,
                    )

            # Stop any running task
            if self._task_state.status == TaskStatus.RUNNING:
                self.stop_task()

            # Shutdown executor
            self._executor.shutdown(wait=True)

            # Tear down the Zenoh mesh component if one was attached.
            # ``self.mesh`` is any object exposing ``.stop()``; falsy values
            # (None - the construction-time default and what a hardware robot
            # gets when ``mesh=False``) are skipped silently.
            if self.mesh:
                try:
                    self.mesh.stop()
                except Exception as mesh_exc:  # noqa: BLE001
                    # Mesh teardown should never block hardware cleanup.
                    logger.warning(
                        "%s: mesh.stop() raised during cleanup: %s",
                        self.tool_name_str,
                        mesh_exc,
                    )

            # Tear down the ROS 2 telemetry bridge if one was created.
            # Guarded like the mesh and teleop steps above, and for a stronger
            # reason: this is the last step before the devices close, so a
            # ``destroy_node()`` on a context another component already shut
            # down would reach the handler at the bottom of this method and
            # skip the disconnect entirely -- leaving the serial port held and
            # the arm energised at its last commanded position, with nothing
            # left that would close either. The sim engine already suppresses
            # this same call; a software resource that will not release must
            # not decide whether the physical ones do.
            try:
                self._shutdown_ros_bridge()
            except Exception as ros_exc:  # noqa: BLE001
                logger.warning(
                    "%s: ROS 2 bridge shutdown raised during cleanup: %s",
                    self.tool_name_str,
                    ros_exc,
                )

            # Close the devices last, once every source of commands is down.
            # ``send_action`` re-opens the robot lazily on a command that finds
            # it disconnected, so a port closed before the teleop loop, the
            # task executor, the mesh and the ROS bridge have stopped can be
            # re-opened behind this teardown -- and would then stay open for
            # the life of the process, since nothing runs after ``cleanup()``.
            #
            # The teleop loop is the one command source whose shutdown is not
            # guaranteed by the ordering: its leader's ``get_action()`` can
            # block past the join budget, which is why ``stop_teleoperate``
            # reports the outcome instead of assuming it. Closing under a live
            # one is worse than deferring on both counts -- the loop's next
            # write re-opens the port through ``send_action``'s lazy connect, so
            # the release does not hold, and that write lands *after* the
            # driver's ``disconnect()`` disabled torque, leaving the arm
            # energised at a fresh command. ``G1Driver.cleanup`` declines the
            # same teardown for the same reason.
            if teleop_stopped:
                self._disconnect_devices()
            else:
                logger.error(
                    "%s: devices left open because the teleop loop did not stop; "
                    "call stop_teleoperate() to re-join it, then cleanup() again",
                    self.tool_name_str,
                )

            logger.info(f"{self.tool_name_str} cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error for {self.tool_name_str}: {e}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        if not hasattr(self, "_shutdown_event"):
            # ``__init__`` refused a kwarg (``action_horizon``,
            # ``control_frequency``) before the executor and this latch were
            # created, so the instance holds nothing to release. ``cleanup()``
            # would raise on the first attribute it never reached and log that
            # name as a cleanup failure beside the ValueError the caller was
            # owed. A bring-up that fails after this point still cleans up.
            return
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors in destructor

    async def get_status(self) -> dict[str, Any]:
        """Report the device's measured state plus this robot's task state.

        Two fields answer different questions about cameras, and the pair is
        what makes an unhealthy device attributable:

        * ``cameras`` enumerates the camera names the device is *configured*
          for - which image streams exist at all.
        * ``cameras_connected`` maps each live camera to its own
          ``is_connected`` reading - which of those streams are actually up.

        ``is_connected`` on a lerobot arm is ``bus and all(cameras)``, so a
        single dropped camera pulls it to ``False`` without naming which of the
        N+1 facts fell. Reported beside the per-camera readings, a ``False``
        becomes attributable: a camera reading ``False`` is the culprit, and
        every camera reading ``True`` leaves the motor bus as the one by
        elimination.

        Best-effort per camera, mirroring the observation path: a camera whose
        ``is_connected`` probe raises is omitted from ``cameras_connected``
        rather than failing the whole probe, so a name present in ``cameras``
        and absent from ``cameras_connected`` is one whose state could not be
        read. That same raise also reaches the arm's aggregate ``is_connected``,
        which reports ``None`` for it rather than degrading this probe to its
        error shape - so the attribution survives the failure it exists to
        attribute. A device exposing no live camera objects reports an empty map.

        Returns:
            Status dict of measured facts. An unexpected failure anywhere in
            the probe degrades to ``{"robot_name", "error", "is_connected":
            False, "task_status": "error"}`` rather than propagating, so a
            supervising agent reads a verdict instead of taking an exception.
        """
        try:
            status_data = {
                "robot_name": self.tool_name_str,
                "robot_type": getattr(self.robot, "robot_type", self.robot.name),
                "robot_info": str(self.robot),
                "data_config": self.data_config,
                **self._device_facts(),
                "ros2_bridge": bool(getattr(self, "_ros_bridge", None) is not None),
                "ros2_transport": getattr(self, "_ros2_transport", "rclpy")
                if getattr(self, "_ros_bridge", None) is not None
                else None,
                "task_status": self._task_state.status.value,
                "current_instruction": self._task_state.instruction,
                "task_duration": self._task_state.duration,
                "task_steps": self._task_state.step_count,
            }

            # Add error info if present
            if self._task_state.error_message:
                status_data["task_error"] = self._task_state.error_message

            return status_data

        except Exception as e:
            logger.error(f"Error getting status for {self.tool_name_str}: {e}")
            return {
                "robot_name": self.tool_name_str,
                "error": str(e),
                "is_connected": False,
                "task_status": "error",
            }

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,  # noqa: ARG002 - single hardware robot; arg is for TeleopMixin parity with sim
    ) -> dict[str, Any]:
        """Apply a single action to the hardware robot (TeleopMixin contract).

        Synchronous so it can be driven from the :class:`TeleopMixin` teleop
        loop thread. Ensures the underlying lerobot robot is connected, then
        delegates through :func:`strands_robots.bus_access.write_action`, so a
        teleop or ROS 2 command shares the bus with the mesh's readers instead of
        racing them. ``robot_name`` is accepted
        for parity with the multi-robot simulation host but ignored here - a
        hardware ``Robot`` wraps exactly one device.

        Args:
            action: Flat ``{motor.pos: float}`` action dict (lerobot shape).
            robot_name: Ignored (single robot). Present for mixin parity.

        Returns:
            Status dict (``success``/``error``) so the teleop loop can count
            errors without exceptions tearing down the hot loop.
        """
        try:
            # Lazy connect on first action, through the same locked unit as the
            # policy-run path: a device an observe action opened goes back
            # first - on a camera-less arm the open bus reads as connected, and
            # a write would then reach servos the driver's ``configure()`` never
            # set up - and no observe action can reopen it before the connect.
            # calibrate=False: a teleop session assumes the follower is already
            # calibrated (same contract as the policy-run path).
            try:
                self._bring_up_robot()
            except Exception:
                self._close_open_devices()
                raise
            write_action(self.robot, action)
            return {"status": "success", "content": [{"text": "ok"}]}
        except Exception as e:  # noqa: BLE001 - surface as status, never kill the loop
            logger.error("%s send_action failed: %s", self.tool_name_str, e)
            return {
                "status": "error",
                "content": [{"text": f"{self.tool_name_str} send_action error: {e}"}],
            }

    async def stop(self) -> None:
        """Stop the robot and release everything it holds.

        Terminal, exactly as :meth:`cleanup` is -- this is the async spelling
        of it, and it delegates every step rather than performing any itself.

        The disconnect in particular belongs to that cleanup rather than ahead
        of it. lerobot gates ``Robot.disconnect()`` on ``is_connected``
        (``bus.is_connected and all(cam.is_connected ...)``) and raises
        ``DeviceNotConnectedError`` when it is false, so disconnecting here
        first meant that stopping a robot which was never connected -- or one
        left half-open by a failed bring-up -- raised before :meth:`cleanup`
        was reached. Every terminal guarantee was then silently skipped: no
        shutdown latch, a task executor still accepting work, and any device a
        half-open connect had opened still held, with no entry point left that
        would close it.

        Ordering matters as much as reachability. :meth:`cleanup` closes the
        devices *last*, once the teleop loop, the task executor, the mesh and
        the ROS bridge are all down, because :meth:`send_action` re-opens the
        robot lazily on a command that finds it disconnected. Disconnecting
        here put that close ahead of every one of those command sources.

        Runs off the event loop: :meth:`cleanup` joins the task executor and
        closes a serial port, both of which block.
        """
        try:
            await asyncio.to_thread(self.cleanup)

            logger.info(f"{self.tool_name_str} stopped and disconnected")

        except Exception as e:
            logger.error(f"Error stopping robot: {e}")

    # ------------------------------------------------------------------
    # Teleoperation over mesh - input publishing and receiving
    # ------------------------------------------------------------------

    def start_teleop_publish(
        self,
        teleoperator: Any,
        device_name: str = "leader",
        method: str = "arm",
        hz: float = 50.0,
    ) -> dict[str, Any]:
        """Start publishing teleoperator actions to the mesh.

        This makes the robot a *teleop source*: another peer on the mesh
        can call ``start_teleop_receive(source_peer_id=self.peer_id)`` to
        have its hardware follow along.

        Args:
            teleoperator: Any object with a callable ``get_action() -> dict``.
                Typically a lerobot Teleoperator (SOLeader, GamepadTeleop,
                KeyboardTeleop, Phone). The publish loop polls it every tick, so
                a device that does not satisfy that contract is refused here
                rather than on the loop thread - the same domain
                :meth:`attach_teleop` grades a locally attached device against.
            device_name: Name for this input stream (e.g. "leader", "gamepad").
            method: Input method label ("arm", "gamepad", "keyboard", "phone").
            hz: Publishing frequency in Hz. Must be a positive finite number;
                the publish loop's period is ``1 / hz``.

        Returns:
            Status dict with topic and peer_id for the receiver to use, or an
            error dict when the mesh is inactive, ``teleoperator`` cannot be
            polled, ``device_name`` is not a valid mesh identifier, or ``hz`` is
            not a rate the publish loop can honor.
        """
        if not self.mesh or not self.mesh.alive:
            return {"status": "error", "content": [{"text": "Mesh not active. Cannot publish input."}]}

        from strands_robots.mesh.security import ValidationError, validate_mesh_identifier

        # All three arguments are validated up front, before the teardown of any
        # publisher already registered under this device name: a rejected call
        # must not stop a live stream. ``device_name`` becomes a segment of the
        # published key expression and a key in ``_input_publishers``, ``hz``
        # sets the publish loop's ``1 / hz`` period, and ``teleoperator`` is what
        # that loop polls for every frame - a device with no callable
        # ``get_action`` cannot produce one, so it replaces a working stream with
        # a session that reports running and publishes nothing. Report through
        # the tool envelope rather than raising.
        try:
            validate_mesh_identifier(device_name, "start_teleop_publish.device_name")
        except ValidationError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}

        error = positive_finite_number_error(hz, "hz", "start_teleop_publish")
        if error:
            return {"status": "error", "content": [{"text": error}]}

        error = teleoperator_contract_error(teleoperator, "teleoperator", "start_teleop_publish")
        if error:
            return {"status": "error", "content": [{"text": error}]}

        from strands_robots.mesh import InputPublisher

        # Store publisher on the robot instance
        if not hasattr(self, "_input_publishers"):
            self._input_publishers: dict[str, InputPublisher] = {}

        if device_name in self._input_publishers:
            # Stop existing publisher for this device
            self._input_publishers[device_name].stop()

        publisher = InputPublisher(
            mesh=self.mesh,
            teleoperator=teleoperator,
            device_name=device_name,
            method=method,
            hz=hz,
        )
        publisher.start()
        self._input_publishers[device_name] = publisher

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Input publisher started: {device_name} ({method} @ {hz}Hz)\n"
                    f"Topic: {publisher.topic}\n"
                    f"Peer ID: {self.peer_id}\n"
                    f"Remote peers can receive with: start_teleop_receive(source_peer_id='{self.peer_id}')"
                }
            ],
        }

    def stop_teleop(self, device_name: str | None = None) -> dict[str, Any]:
        """Stop all or a specific teleop publisher/receiver.

        Args:
            device_name: If provided, stop only the named publisher/receiver.
                If None, stop all.

        Returns:
            Stats from stopped sessions.
        """
        results = []

        # Stop publishers
        if hasattr(self, "_input_publishers"):
            if device_name:
                pub = self._input_publishers.pop(device_name, None)
                if pub:
                    results.append(pub.stop())
            else:
                for name, pub in list(self._input_publishers.items()):
                    results.append(pub.stop())
                self._input_publishers.clear()

        # Stop receivers
        if hasattr(self, "_input_receivers"):
            if device_name:
                # Match by device name suffix
                to_remove = [k for k in self._input_receivers if k.endswith(f"/{device_name}")]
                for k in to_remove:
                    results.append(self._input_receivers.pop(k).stop())
            else:
                for key, rcv in list(self._input_receivers.items()):
                    results.append(rcv.stop())
                self._input_receivers.clear()

        if not results:
            return {"status": "success", "content": [{"text": "No active teleop sessions."}]}

        stats_text = "\n".join(
            f"  {r.get('device', r.get('source', '?'))}: "
            f"{r.get('frames', r.get('frames_received', 0))} frames, "
            f"{r.get('hz_actual', 0):.1f} Hz"
            for r in results
        )
        return {
            "status": "success",
            "content": [{"text": f"Teleop stopped:\n{stats_text}"}],
        }

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
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import importlib
import logging
import pkgutil
import shutil
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolResult, ToolSpec, ToolUse

from strands_robots.teleop_mixin import TeleopMixin
from strands_robots.utils import positive_finite_number_error, require_optional

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.robot import Robot as LeRobotRobot

    from .policies import Policy

logger = logging.getLogger(__name__)


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


# Forwarded kwargs whose lerobot consumer dispatches on the value's EXACT type,
# so an int where a float is declared is not "close enough" - it takes a
# ``raise TypeError`` branch at the first use. Mapped to the coercion applied at
# the forwarding seam.
#
# ``max_relative_target`` is the motivating case and a safety knob:
# ``lerobot.robots.utils.ensure_safe_goal_position`` does
#     if isinstance(max_relative_target, float): ...
#     elif isinstance(max_relative_target, dict): ...
#     else: raise TypeError(max_relative_target)
# and Python's ``int`` is not a subclass of ``float``, so ``Robot('so101',
# max_relative_target=10)`` made EVERY ``send_action`` raise ``TypeError: 10``
# before a single servo write - the arm did not move at all. The repo's own
# docs recommend exactly that int literal, so the documented safety
# configuration bricked the arm. ``SOFollowerConfig`` annotates the field
# ``float | dict[str, float] | None``, so honouring the declared type is this
# seam's job: it already validates and rejects unknown kwargs here.
_NUMERIC_FORWARDED_KWARGS: frozenset[str] = frozenset(
    {
        "max_relative_target",
        "control_dt",
    }
)


def _coerce_forwarded_kwarg(key: str, value: Any) -> Any:
    """Coerce a forwarded kwarg to the numeric type its lerobot consumer requires.

    Only the kwargs in :data:`_NUMERIC_FORWARDED_KWARGS` are touched; every other
    value is forwarded verbatim so this cannot change the meaning of an unrelated
    field. ``bool`` is rejected rather than coerced: ``True`` would silently
    become ``1.0`` (a 1-degree clamp that looks like a frozen arm), and it is far
    more likely to be a caller mistake than an intended magnitude.

    A per-joint ``dict`` is coerced value-wise, matching
    ``ensure_safe_goal_position``'s dict branch.

    Args:
        key: The kwarg name being forwarded.
        value: The caller-supplied value.

    Returns:
        The value to place on the lerobot config dataclass.

    Raises:
        ValueError: When ``value`` is a bool, or a type that cannot be a numeric
            clamp, with the accepted shapes named.
    """
    if key not in _NUMERIC_FORWARDED_KWARGS or value is None:
        return value
    if isinstance(value, bool):
        raise ValueError(
            f"{key}={value!r} is a bool; expected a number (e.g. {key}=10.0) or a "
            f"per-joint mapping (e.g. {key}={{'shoulder_pan': 10.0}}). A bool would "
            f"silently become {float(value)}, which reads as a frozen arm rather than a "
            f"configuration error."
        )
    if isinstance(value, int):
        # The whole point: int -> float so lerobot's isinstance(..., float)
        # dispatch takes the clamp branch instead of raising.
        return float(value)
    if isinstance(value, float):
        return value
    if isinstance(value, dict):
        coerced: dict[str, float] = {}
        for joint, limit in value.items():
            if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                raise ValueError(
                    f"{key}[{joint!r}]={limit!r} is not a number. Every value in a per-joint "
                    f"{key} mapping must be numeric (e.g. {{'shoulder_pan': 10.0}})."
                )
            coerced[str(joint)] = float(limit)
        return coerced
    raise ValueError(
        f"{key}={value!r} has unsupported type {type(value).__name__}; expected a number "
        f"(e.g. {key}=10.0) or a per-joint mapping of joint name to number "
        f"(e.g. {key}={{'shoulder_pan': 10.0}})."
    )


#: Camera ``type`` selector -> the lerobot config dataclass implementing it, as
#: ``(module path, class name)`` so the import stays lazy. Adding a backend is a
#: one-line entry; previously only ``"opencv"`` was reachable and every other
#: value raised "Unsupported camera type", even though lerobot ships
#: ``RealSenseCameraConfig`` (with its own ``use_depth`` / ``use_rgb`` fields).
_CAMERA_CONFIG_CLASSES: dict[str, tuple[str, str]] = {
    "opencv": ("lerobot.cameras.opencv.configuration_opencv", "OpenCVCameraConfig"),
    "realsense": ("lerobot.cameras.realsense.configuration_realsense", "RealSenseCameraConfig"),
}

#: Camera-spec keys that are strands-level metadata, not dataclass fields.
_CAMERA_META_KEYS = frozenset({"type"})

#: Defaults strands applies when the caller omits them. lerobot's own dataclass
#: defaults for these three are ``None``, but ``RobotConfig.__post_init__``
#: REJECTS a camera with any of them unset ("Specifying 'width' is required for
#: the camera to be used in a robot"), so a spec of just
#: ``{"index_or_path": 0}`` cannot construct a robot without them. Applied only
#: for fields the resolved dataclass actually declares.
_CAMERA_FIELD_DEFAULTS: dict[str, Any] = {"fps": 30, "width": 640, "height": 480}


def _lerobot_driver_dir_name(config: Any) -> str | None:
    """The calibration DIRECTORY name lerobot will use for ``config``.

    lerobot resolves ``calibration_dir = HF_LEROBOT_CALIBRATION / ROBOTS /
    self.name``, where ``name`` is the DRIVER class's attribute - so
    ``so101_follower`` and ``so100_follower`` both live in ``so_follower/``. The
    config class does not carry it, and the type-to-driver mapping lives in
    ``lerobot.robots.utils.make_robot_from_config``'s if/elif chain, so the driver
    is instantiated to ask it. That is cheap (``Robot.__init__`` only resolves
    paths and reads any existing calibration; it opens no port).

    Args:
        config: A constructed lerobot ``RobotConfig``.

    Returns:
        The directory name, or ``None`` when the driver cannot be resolved (a
        diagnostic helper must never raise).
    """
    try:
        from lerobot.robots.utils import make_robot_from_config

        driver_name = getattr(make_robot_from_config(config), "name", None)
        if isinstance(driver_name, str) and driver_name:
            return driver_name
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never break construction
        logger.debug("could not resolve the lerobot driver dir for %r: %s", config, exc)
    return None


def _build_camera_config(name: str, spec: dict[str, Any]) -> Any:
    """Build one lerobot camera config from a strands camera spec.

    Driven by the target dataclass's own fields rather than a hand-picked key
    list. The previous hand-picked version forwarded 6 keys (later 8) and
    silently discarded everything else, so ``four_cc`` (a typo for ``fourcc``),
    ``backend``, ``use_depth`` / ``use_rgb`` and any nonsense key vanished with no
    error - the exact opposite of the loud unknown-kwarg rejection this same
    method applies to ROBOT kwargs a few lines later, and a defect the AGENTS.md
    rule "Reject silently-dropped kwargs" names directly. A dropped ``fourcc``
    silently caps a UVC camera at ~5fps, which then presents as a policy
    mysteriously starved of frames.

    Being fields-driven also means a new lerobot camera field works with no
    change here, matching the robot-kwarg contract.

    Args:
        name: Camera name, used only in error messages.
        spec: The caller's camera dict, e.g.
            ``{"type": "opencv", "index_or_path": "/dev/video0", "fourcc": "MJPG"}``.

    Returns:
        The constructed lerobot camera config dataclass instance.

    Raises:
        ValueError: If ``type`` is unknown, a key is not a field of the resolved
            dataclass, a required field is missing, or the dataclass rejects a
            value.
    """
    cam_type = str(spec.get("type", "opencv"))
    target = _CAMERA_CONFIG_CLASSES.get(cam_type)
    if target is None:
        raise ValueError(
            f"Unsupported camera type {cam_type!r} for camera {name!r}. "
            f"Supported types: {sorted(_CAMERA_CONFIG_CLASSES)}."
        )
    module_path, class_name = target
    try:
        ConfigClass = getattr(importlib.import_module(module_path), class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Camera type {cam_type!r} for camera {name!r} is not available in this lerobot "
            f"install ({class_name} could not be imported: {e})."
        ) from e

    valid_fields = {f.name for f in dataclasses.fields(ConfigClass)}
    unknown = set(spec) - valid_fields - _CAMERA_META_KEYS
    if unknown:
        raise ValueError(
            f"Unknown camera key(s) for camera {name!r} (type={cam_type!r}): {sorted(unknown)}. "
            f"{class_name} accepts: {sorted(valid_fields)} (plus 'type'). "
            f"(If this is a typo, fix it.)"
        )

    config_data = {k: v for k, v in spec.items() if k in valid_fields}
    for field, default in _CAMERA_FIELD_DEFAULTS.items():
        if field in valid_fields:
            config_data.setdefault(field, default)
    try:
        return ConfigClass(**config_data)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to construct {class_name} for camera {name!r}: {e}. Config: {config_data}") from e


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

    # Pick up third-party plugins (``lerobot_robot_*`` distributions) via
    # lerobot's own loader if available -- lets external robot vendors
    # expose drivers without any strands_robots involvement.
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
    start_time: float = 0.0
    duration: float = 0.0
    step_count: int = 0
    error_message: str = ""
    task_future: Future | None = None
    #: Control rate the loop actually sustained, in Hz (actions applied divided
    #: by elapsed wall time). Distinct from the configured
    #: ``control_frequency``: observation reads and policy inference happen
    #: between action batches, so the achieved rate is always lower. Reported so
    #: an operator can see the real rate instead of assuming the declared one -
    #: RTC chunk blending is computed against the declared rate, so a large gap
    #: means the blend is using a timebase the loop never ran at.
    achieved_hz: float = 0.0


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
            action_horizon: Actions per inference step
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
            ros2_commands: When True (default), the bridge also subscribes to
                ``/<robot>/joint_command`` and forwards inbound messages to
                ``send_action`` so an external ROS 2 stack can drive the real
                arm (full duplex). Set False for a read-only telemetry bridge.
                Ignored unless ``ros2_bridge=True``.
            ros2_transport: Which ROS 2 backend the bridge uses:
                ``"rclpy"`` (default) - full ``sensor_msgs`` fidelity, needs a
                sourced ROS 2 distro; ``"rtps"`` - pure cyclonedds (a single
                pip wheel, no rclpy / no sourced distro), type coverage bounded
                by the local IDL bundle (joint_states + image_raw). Both emit
                byte-identical topics. Ignored unless ``ros2_bridge=True``.
            joint_limits: Optional ``{motor: (min, max)}`` clamp ranges threaded
                into the ROS 2 bridge. When set, an inbound ``joint_command``
                whose ANY joint is outside its declared range is rejected whole
                (no partial application). Requires ``ros2_bridge=True``.
            dds_security_config: Optional DDS Security credentials
                (``identity_ca``, ``certificate``, ``private_key``,
                ``governance``, ``permissions``; ``permissions_ca`` optional)
                for the pure-RTPS bridge. When ``ros2_commands=True`` on the
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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{tool_name}_executor")
        self._shutdown_event = threading.Event()
        # Set by stop_task() regardless of task state, and cleared at the start of
        # each rollout. Distinct from _shutdown_event (process teardown): this is
        # a per-rollout stop latch, and it exists so a stop issued during the
        # CONNECTING window is not lost. See stop_task.
        self._stop_requested = threading.Event()
        # Real mutual exclusion for the control loop. The old guard compared
        # _task_state.status to RUNNING, but _execute_task_async sits in
        # CONNECTING for the whole 2-3s hardware bring-up, so a second call in
        # that window passed the guard and BOTH loops then interleaved
        # sync_read/sync_write on one FeetechMotorsBus - which is not
        # thread-safe, and whose only interlock is the port handler's is_using
        # flag. A lock is the right primitive because the status flip and the
        # guard read are not atomic; a widened status check would still race.
        self._task_lock = threading.Lock()
        # Transient-observation tolerance (see _read_observation_resiliently).
        # Defaults chosen from the live rig: 3 retries matches what the user's own
        # scripts hand-rolled, and 20 consecutive reused frames is ~1s at 20Hz -
        # long enough to ride out a USB2 bandwidth hiccup, short enough that the
        # policy is never driven open-loop for a whole manoeuvre.
        self._obs_retries = 3
        self._obs_retry_backoff_s = 0.05
        self._max_consecutive_obs_failures = 20
        self._consecutive_obs_failures = 0
        self._last_good_observation: dict[str, Any] | None = None
        self._last_obs_error = ""
        # Action keys the driver refused to command (see
        # _account_for_dropped_action). Per-rollout, reset in _execute_task_async.
        self._dropped_action_steps = 0
        self._dropped_action_keys: list[str] = []

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
        # lerobot ImportError first and masks the documented
        # "pip install 'strands-robots[ros2]'" hint that the operator who set
        # ros2_bridge=True actually needs to see. require_optional caches the
        # module, so the real bridge construction in _init_ros_bridge pays nothing.
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
        missing ``rclpy`` / ``cyclonedds`` surfaces its documented
        ``pip install 'strands-robots[ros2]'`` error immediately - rather than
        being masked by the lerobot ImportError ``_initialize_robot`` raises
        first in an environment without the ``[lerobot]`` extra.
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
            require_optional("rclpy", extra="ros2", purpose="the ROS 2 telemetry bridge (ros2_bridge=True)")

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
            ros2_commands: When True (default), also subscribe to
                ``joint_command`` and drive the arm; False for read-only.
            ros2_transport: ``"rclpy"`` or ``"rtps"`` (see above).
            joint_limits: Optional ``{motor: (min, max)}`` clamp ranges threaded
                into whichever bridge is built (both enforce them on inbound
                commands via the shared base).
            dds_security_config: Optional DDS Security credentials, consumed
                only by the ``"rtps"`` bridge. Passing it with the ``"rclpy"``
                transport raises (rclpy DDS Security is configured at the RMW
                layer, not by a config dict).

        Raises:
            ValueError: If ``ros2_transport`` is not ``"rclpy"`` or ``"rtps"``;
                if ``joint_limits`` / ``dds_security_config`` are supplied with
                ``ros2_bridge=False``; or if ``dds_security_config`` is supplied
                with ``ros2_transport="rclpy"``.
        """
        self._ros2_bridge_enabled = bool(ros2_bridge)
        self._ros2_domain = int(ros2_domain)
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
        observation = self.robot.get_observation()
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
        """Initialize LeRobot robot instance using native lerobot patterns."""
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
        """
        # ``lerobot`` is already a hard dep at this point (``_initialize_robot``
        # imports it eagerly). Importing the config module here is cheap and the
        # only reason it isn't at module top is that some downstream packagers
        # tree-shake unused submodules. Camera configs are built by
        # ``_build_camera_config``, which imports its own dataclasses lazily.
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
                config_data[key] = _coerce_forwarded_kwarg(key, kwargs[key])
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
                config_data[key] = _coerce_forwarded_kwarg(key, kwargs[key])

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

        try:
            config = ConfigClass(**config_data)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to construct {ConfigClass.__name__} for robot type {robot_type!r}: {e}. Config: {config_data}"
            ) from e

        if "id" not in kwargs and "id" in valid_fields:
            # A diagnostic must never be able to fail a construction that already
            # succeeded, whatever goes wrong inside it.
            try:
                self._warn_shared_calibration_id(robot_type, config, kwargs)
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                logger.debug("shared-calibration-id check failed: %s", exc)
        return config

    async def _get_policy(
        self, policy_port: int | None = None, policy_host: str = "localhost", policy_provider: str = "groot"
    ) -> Policy:
        """Create policy on-the-fly from invocation parameters."""
        from .policies import create_policy

        if not policy_port:
            raise ValueError("policy_port is required for robot operation")

        policy_config = {"port": policy_port, "host": policy_host}

        if self.data_config:
            policy_config["data_config"] = self.data_config

        return create_policy(policy_provider, **policy_config)

    def _rollback_half_open_connect(self) -> None:
        """Close every subdevice a failed connect can leave half-open.

        lerobot's ``MotorsBus.connect()`` opens the serial port *before* the
        motor handshake, so a failed handshake (e.g. an unpowered bus) leaves
        ``is_connected`` True while fire-and-forget writes to the dead bus
        keep "succeeding". Closing the port makes the next attempt retry the
        connect and keep surfacing the failure. ``disable_torque=False``: the
        handshake already failed, so the default disconnect's torque write
        would only raise again (before ``closePort``) and leave the port open.

        The CAMERAS need the same treatment, and used not to get it.
        ``SOFollower.connect()`` opens them in dict order, so when the second
        camera fails the first is left open. ``OpenCVCamera.connect`` is
        ``@check_if_already_connected``, so the next attempt raises
        ``DeviceAlreadyConnectedError`` on that stale-open camera - which the
        handler in :meth:`_connect_robot` read as benign - and the loop never
        reached the failing camera again. Measured across three attempts with a
        camera that fails once::

            attempt 1: bus.connect wrist.connect front.connect bus.disconnect
            attempt 2: bus.connect wrist.connect        <- dies on stale wrist
            attempt 3: bus.connect wrist.connect
            front.connect() called exactly ONCE -> healthy hardware never retried

        So the robot could never be connected again, even after the operator
        replugged the cable, until the process restarted. The recovery route was
        closed too: ``SOFollower.disconnect`` is ``@check_if_not_connected`` and
        raises in exactly this half-open state, which is why the live scripts
        bypass this path with ``robot.robot.connect(calibrate=False)``.
        """
        bus = getattr(self.robot, "bus", None)
        try:
            if bus is not None and getattr(bus, "is_connected", False):
                bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.debug("%s post-connect-failure port close failed", self.tool_name_str)
        # Per camera, not via robot.disconnect(): that is decorated
        # @check_if_not_connected and raises on a half-open robot, so it cannot
        # be the cleanup path for the state it refuses to describe.
        for name, camera in (getattr(self.robot, "cameras", None) or {}).items():
            try:
                if getattr(camera, "is_connected", False):
                    camera.disconnect()
                    logger.debug("%s rolled back half-open camera %s", self.tool_name_str, name)
            except Exception as exc:  # noqa: BLE001 - one bad camera must not block the rest
                logger.warning(
                    "%s could not close half-open camera %s during connect rollback: %s. "
                    "The next connect attempt may report it as already connected.",
                    self.tool_name_str,
                    name,
                    exc,
                )

    def _raise_unless_fully_connected(self, exc: Exception) -> None:
        """Accept an "already connected" error only if the device truly is.

        ``SOFollower.is_connected`` is ``bus.is_connected and all(cam.is_connected
        ...)`` - all-or-nothing by contract. So an "already connected" error
        raised while that is False means some SUBDEVICE is stale-open and the
        connect never completed; reporting success there is how a half-open
        robot became permanently unconnectable.

        Args:
            exc: The "already connected" error that was caught.

        Raises:
            Exception: ``exc`` itself, when the robot is not fully connected.
                The caller's handler then rolls back, so the next attempt starts
                from a fully-closed device.
        """
        if getattr(self.robot, "is_connected", False):
            logger.info("%s was already connected", self.robot)
            return
        stale = [
            name
            for name, camera in (getattr(self.robot, "cameras", None) or {}).items()
            if getattr(camera, "is_connected", False)
        ]
        logger.error(
            "%s reported 'already connected' but is NOT fully connected (stale-open cameras: %s). "
            "Rolling back so the next attempt retries every subdevice.",
            self.tool_name_str,
            stale or "none",
        )
        raise exc

    async def _connect_robot(self) -> tuple[bool, str]:
        """Connect to robot hardware with proper error handling.

        Returns:
            tuple[bool, str]: (success, error_message) - error_message is empty on success
        """
        try:
            # Import lerobot exceptions
            from lerobot.utils.errors import DeviceAlreadyConnectedError

            # Check if already connected
            if self.robot.is_connected:
                logger.info(f"{self.robot} already connected")
                return True, ""

            logger.info(f"Connecting to {self.robot}...")

            # Handle robot connection using lerobot's error handling patterns
            try:
                if not self.robot.is_connected:
                    await asyncio.to_thread(self.robot.connect, False)  # calibrate=False

            except DeviceAlreadyConnectedError as e:
                # Benign ONLY if the whole device really is connected. When a
                # SUBCOMPONENT is stale-open (the usual case: camera 1 opened,
                # camera 2 failed, so connect re-raises on camera 1 before ever
                # retrying camera 2) treating this as success hid a device that
                # can never finish connecting. Roll back and re-raise so the
                # caller sees the real failure and the NEXT attempt starts clean.
                self._raise_unless_fully_connected(e)

            except Exception as e:
                # Check if it's the string version of "already connected" error
                error_str = str(e).lower()
                if "already connected" in error_str or "is already connected" in error_str:
                    self._raise_unless_fully_connected(e)
                else:
                    # Re-raise if it's a different error
                    raise e

            # Final connection check
            if not self.robot.is_connected:
                error_msg = f"Failed to connect to {self.robot}"
                logger.error(f"{error_msg}")
                return False, error_msg

            # Check robot calibration
            if hasattr(self.robot, "is_calibrated") and not self.robot.is_calibrated:
                error_msg = (
                    f"Robot {self.robot} is not calibrated. Please calibrate the robot manually"
                    " first using LeRobot's calibration process (lerobot-calibrate)"
                )
                logger.error(f"{error_msg}")
                return False, error_msg

            logger.info(f"{self.robot} connected and ready")
            return True, ""

        except Exception as e:
            error_msg = f"Robot connection failed: {e}. Ensure robot is calibrated and accessible on the specified port"
            logger.error(f"{error_msg}")
            # Same rollback as the lazy teleop connect: without it a half-open
            # port makes the NEXT _connect_robot short-circuit on
            # "already connected" and report success against a dead bus.
            self._rollback_half_open_connect()
            return False, error_msg

    async def _initialize_policy(self, policy: Policy) -> bool:
        """Initialize policy with robot state keys.

        The keys bound here are the index basis for the policy's action output
        (``_tensor_to_action_dicts`` maps tensor column i to key i), so anything
        that lands in this list becomes a joint the policy believes it commands.
        See :meth:`_derive_robot_state_keys` for how the list is chosen.
        """
        try:
            # Get robot state keys from observation
            test_obs = await asyncio.to_thread(self.robot.get_observation)
            policy.set_robot_state_keys(self._derive_robot_state_keys(test_obs))
            return True

        except Exception as e:
            logger.error(f"Failed to initialize policy: {e}")
            return False

    def _shutdown_guard(self, entry_point: str) -> dict[str, Any] | None:
        """Refuse to start a rollout on a Robot that has already been shut down.

        ``cleanup()`` / ``stop()`` set ``_shutdown_event`` and never clear it, and
        the control loop's condition includes ``not _shutdown_event.is_set()`` - so
        a later rollout exited before its FIRST iteration while the terminal block
        still marked it ``COMPLETED``, which ``run_policy`` maps to
        ``status="success"``. Measured: "Policy rollout completed: 0 steps in 0.0s"
        with zero servo writes. ``start_task`` was worse, raising a bare
        ``RuntimeError: cannot schedule new futures after shutdown`` from the dead
        executor.

        A shut-down Robot is not recoverable by design (its executor is gone and
        the hardware is disconnected), so the honest answer is an explicit error
        naming that, not a no-op reported as a success.

        Args:
            entry_point: Name of the calling API, for the message.

        Returns:
            An error dict when the robot is shut down, else ``None``.
        """
        if not getattr(self, "_shutdown_event", None) or not self._shutdown_event.is_set():
            return None
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"{entry_point}: this Robot has been shut down (cleanup()/stop() was called), "
                        f"so its executor is gone and the hardware is disconnected. It cannot run "
                        f"another rollout - construct a new Robot(...) instead."
                    )
                }
            ],
        }

    def _account_for_dropped_action(self, requested: dict[str, Any], sent: Any) -> None:
        """Detect keys the driver silently refused to command.

        lerobot drivers return the action they ACTUALLY sent - ``SOFollower``'s
        docstring says so explicitly ("this function always returns the action
        actually sent") and its body filters ``if key.endswith(".pos")``. That
        return value was discarded and ``step_count`` incremented regardless, so
        an action naming some motors correctly and some incorrectly reported
        ``status="success"`` with a full step count while only the matching subset
        moved - half the arm frozen, silently.

        A TOTAL mismatch is already loud (``MotorsBus.sync_write({})`` raises), and
        ``_derive_robot_state_keys`` now binds from ``action_features``, so this is
        the residual PARTIAL case: it needs the driver's own answer to detect,
        which is exactly what the return value is for.

        Counts rather than raising mid-rollout: the arm is already moving, and
        aborting on the first dropped key would leave it wherever it happens to
        be. The count is escalated to an error by the caller once the rollout ends
        (see :meth:`_dropped_action_summary`).

        Args:
            requested: The action dict handed to ``send_action``.
            sent: Whatever ``send_action`` returned. Non-dict returns (older
                drivers, fakes) are ignored - absence of evidence is not evidence
                of a drop.
        """
        if not isinstance(sent, dict) or not requested:
            return
        missing = [key for key in requested if key not in sent]
        if not missing:
            return
        self._dropped_action_steps += 1
        if not self._dropped_action_keys:
            self._dropped_action_keys = sorted(missing)
            declared = sorted(self._declared_action_keys()) or "<undeclared>"
            logger.warning(
                "%s: the driver did not command %d of %d action key(s): %s. It accepts %s. Those "
                "joints are NOT moving; the rest of the arm is. Check the policy's action keys "
                "against the driver's action_features.",
                self.tool_name_str,
                len(missing),
                len(requested),
                sorted(missing),
                declared,
            )

    def _dropped_action_summary(self) -> str:
        """A one-line report of dropped action keys, or ``""`` when none.

        Reads its counters with ``getattr`` defaults because ``get_task_status``
        is reachable on an instance whose ``__init__`` did not complete and before
        any rollout has run - the same half-built contract ``stop_task`` honours.
        """
        steps_with_drops = getattr(self, "_dropped_action_steps", 0)
        if not steps_with_drops:
            return ""
        return (
            f"{steps_with_drops} of {self._task_state.step_count} step(s) had action key(s) "
            f"the driver refused to command (first seen: {getattr(self, '_dropped_action_keys', [])})"
        )

    async def _read_observation_resiliently(self) -> dict[str, Any] | None:
        """Read one observation, tolerating transient camera/bus failures.

        The read used to sit bare inside the loop's single top-level ``try``, so
        ONE dropped frame jumped to the terminal ``except``, set ``ERROR`` and
        returned mid-manoeuvre with the arm still torqued at its last commanded
        pose. Measured: a single ``TimeoutError`` on the 3rd read ended a
        40-step rollout at 8 steps.

        Those exceptions are expected, not exceptional. The installed lerobot
        ``OpenCVCamera.read_latest`` raises ``TimeoutError`` whenever the newest
        frame is older than ``max_age_ms`` (500 ms default) and ``RuntimeError``
        when its background read thread has died - both routine under USB2
        bandwidth contention, which is exactly what two MJPG streams on one
        controller produce. The user's own scripts hand-roll this same retry +
        last-good-frame reuse, which is the clearest evidence it belongs here.

        Strategy: retry within the step (``obs_retries``), then fall back to the
        last good observation, bounded by ``max_consecutive_obs_failures`` so the
        loop can never replay a stale frame open-loop forever. A successful read
        resets the budget.

        Returns:
            The observation, or ``None`` when the failure budget is exhausted and
            the caller must stop.
        """
        last_error: BaseException | None = None
        for attempt in range(1, self._obs_retries + 1):
            try:
                observation = await asyncio.to_thread(self.robot.get_observation)
            except (TimeoutError, RuntimeError, OSError) as exc:
                last_error = exc
                logger.warning(
                    "%s: observation read failed (attempt %d/%d): %s",
                    self.tool_name_str,
                    attempt,
                    self._obs_retries,
                    exc,
                )
                if attempt < self._obs_retries:
                    await asyncio.sleep(self._obs_retry_backoff_s)
                continue
            else:
                self._consecutive_obs_failures = 0
                self._last_good_observation = observation
                return cast("dict[str, Any]", observation)

        # Every retry failed. Reuse the last good frame for a bounded number of
        # consecutive steps: a brief USB hiccup should not abort a manoeuvre, but
        # driving a policy on an indefinitely stale frame is open-loop motion.
        self._last_obs_error = str(last_error) if last_error is not None else "unknown"
        self._consecutive_obs_failures += 1
        if self._last_good_observation is None:
            logger.error(
                "%s: first observation read failed and there is no previous frame to reuse: %s",
                self.tool_name_str,
                self._last_obs_error,
            )
            return None
        if self._consecutive_obs_failures > self._max_consecutive_obs_failures:
            return None
        logger.warning(
            "%s: reusing the last good observation (%d/%d consecutive failures); the policy is "
            "running on a stale frame",
            self.tool_name_str,
            self._consecutive_obs_failures,
            self._max_consecutive_obs_failures,
        )
        return self._last_good_observation

    def _warn_shared_calibration_id(self, robot_type: str, config: Any, kwargs: dict[str, Any]) -> None:
        """Warn when a defaulted ``id`` will reuse an existing calibration file.

        lerobot's whole purpose for ``RobotConfig.id`` is per-instance calibration
        namespacing: ``calibration_fpath = calibration_dir / f"{id}.json"``. But
        the default here is the robot TYPE, so two physical arms of the same model
        on different ports both resolve to ``so101.json``. The second one is not
        silently driven with the first one's numbers - ``_connect_robot`` gates on
        ``is_calibrated`` and lerobot refuses a motor-id/range mismatch - but the
        operator gets a confusing "not calibrated" refusal (or, if the two arms
        happen to be close enough to pass, one arm's offsets on the other's
        motors: the calibrations on this machine differ by up to 82 degrees on
        shoulder_lift).

        Warning rather than changing the default: a port-derived default would
        orphan every existing ``<type>.json`` on disk. This tells the operator the
        file is shared and how to namespace it, which is what the ids already
        present on this host (left_arm, right_arm, orange_arm, ...) show they end
        up doing by hand anyway.

        Best-effort: any failure to resolve the path is a diagnostic miss, never a
        construction failure.

        Args:
            robot_type: The lerobot robot type, for the message.
            config: The constructed lerobot config (its ``id`` is the default).
            kwargs: Construction kwargs, used to name the port in the message.
        """
        try:
            from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

            explicit_dir = kwargs.get("calibration_dir")
            if explicit_dir is not None:
                base = Path(explicit_dir)
            else:
                # The directory is the driver CLASS name (``so_follower``), not
                # the robot type (``so101_follower``); resolve it the way lerobot
                # does, from the registered driver class.
                driver_name = _lerobot_driver_dir_name(config)
                if driver_name is None:
                    return
                base = Path(HF_LEROBOT_CALIBRATION) / ROBOTS / driver_name
            path = base / f"{config.id}.json"
        except (ImportError, AttributeError, OSError, TypeError) as exc:
            logger.debug("could not resolve the calibration path for the shared-id check: %s", exc)
            return

        if not path.is_file():
            return
        port = kwargs.get("port") or kwargs.get("robot_ip") or "<unset>"
        logger.warning(
            "%s: no id= was given, so calibration defaults to the robot TYPE and this instance will "
            "use the EXISTING file %s (port=%s). lerobot uses id to namespace calibration per "
            "PHYSICAL arm, so a second arm of the same model will either be refused as "
            "'not calibrated' or run on the first arm's offsets. Pass a distinct id= per arm "
            "(e.g. id='left_arm' / id='right_arm', or the controller serial) and calibrate each: "
            "lerobot-calibrate --robot.type=%s --robot.port=%s --robot.id=<your_id>",
            self.tool_name_str,
            path,
            port,
            robot_type,
            port,
        )

    def _derive_robot_state_keys(self, observation: dict[str, Any]) -> list[str]:
        """Choose the ordered joint keys a policy's action columns bind to.

        Prefers the driver's OWN declared ``action_features`` (intersected with
        what the observation actually reports, in the driver's declared order)
        because that is the exact set ``send_action`` accepts. lerobot's
        ``SOFollower.send_action`` keeps only ``"<motor>.pos"`` entries; a key it
        does not accept is a wasted action column that shifts every subsequent
        column onto the wrong joint.

        Falls back to "observation keys minus declared cameras" only when the
        driver declares no usable action features. That fallback was the sole
        behaviour, and it over-collects: a driver's observation carries entries
        beyond joints and its declared camera names, and none of them are
        filtered because they are not in ``config.cameras``. Verified against
        the installed lerobot 0.6.0:

        * ``<cam>_depth`` - an ndarray depth frame, emitted by every
          depth-capable driver (``so_follower``, ``koch_follower``,
          ``omx_follower``, ``openarm_follower``, ``rebot_b601_follower``,
          ``hope_jr``). The key is ``f"{cam_key}_depth"``, NOT ``cam_key``, so
          the camera filter misses it and an image array enters the state
          vector.
        * ``<motor>.vel`` / ``<motor>.torque`` - extra per-motor channels
          emitted by ``openarm_follower``, which triple-count every joint.

        Non-scalar values are dropped from the fallback for the same reason, so
        a depth frame can never occupy an action column.

        Args:
            observation: A live observation read from the driver.

        Returns:
            Ordered keys to bind to the policy's action columns.
        """
        declared = self._declared_action_keys()
        if declared:
            ordered = [
                k
                for k in declared
                if k in observation
                and isinstance(observation[k], (int, float))
                and not isinstance(observation[k], bool)
            ]
            # A driver may DECLARE more channels than send_action consumes.
            # lerobot's openarm_follower with use_velocity_and_torque=True has
            # action_features == _motors_ft, i.e. '<motor>.pos' AND '.vel' AND
            # '.torque' - 21 entries for a 7-DOF arm - while its send_action does
            # `if key.endswith(".pos")` and discards the rest. Since binding is
            # positional (tensor column i -> key i), a 7-D checkpoint's columns
            # would land on pos/vel/torque triples: joint_1 <- model[0],
            # joint_2 <- model[3], joint_3 <- model[6], joints 4..7 <- 0.0. Keep
            # only the commandable channel when the declared set mixes families.
            positional = [k for k in ordered if k.endswith(".pos")]
            if positional and len(positional) < len(ordered):
                logger.info(
                    "%s: driver declares %d action_features across mixed channels; bound the %d "
                    "'.pos' key(s) only and dropped %s, which send_action does not consume "
                    "(a positional binding would otherwise scatter action columns across them).",
                    self.tool_name_str,
                    len(ordered),
                    len(positional),
                    sorted(set(ordered) - set(positional)),
                )
                ordered = positional
            if ordered:
                dropped = [k for k in observation if k not in set(ordered)]
                logger.info(
                    "%s: bound %d policy action key(s) from the driver's declared action_features: %s%s",
                    self.tool_name_str,
                    len(ordered),
                    ordered,
                    f" (observation also carried {dropped}, not commandable)" if dropped else "",
                )
                return ordered
            logger.warning(
                "%s: the driver declares action_features %s but the observation reports none of "
                "them (it reports %s). Falling back to the observation's own scalar keys; verify "
                "the policy's action columns land on the joints you expect.",
                self.tool_name_str,
                sorted(declared)[:8],
                sorted(observation)[:8],
            )

        camera_keys: set[str] = set()
        if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
            camera_keys = set(self.robot.config.cameras.keys())
        # Keep scalars only: a depth frame or an RGB array is never a joint
        # command, and letting one through shifts every later action column.
        return [
            k
            for k, v in observation.items()
            if k not in camera_keys and isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

    def _declared_action_keys(self) -> list[str]:
        """Ordered action keys the underlying lerobot driver declares it accepts.

        Reads ``robot.action_features`` (a lerobot ``Robot`` contract: a mapping
        of action key to type/shape). Returns ``[]`` when the driver does not
        expose it or the read fails, so the caller falls back rather than
        breaking on a driver that predates the property.
        """
        try:
            features = getattr(self.robot, "action_features", None)
        except Exception as exc:  # noqa: BLE001 - a driver property must not break policy setup
            logger.debug("%s: action_features unavailable: %s", self.tool_name_str, exc)
            return []
        if not isinstance(features, dict):
            return []
        return [k for k in features if isinstance(k, str)]

    async def _execute_task_async(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        policy_object: Policy | None = None,
        n_steps: int | None = None,
    ) -> None:
        """Execute robot task in background thread (internal method).

        ``policy_object`` (when given) is driven as-is and the provider/port
        arguments are ignored - the ``run_policy`` path. ``n_steps`` caps the
        number of applied actions; the loop stops at whichever of
        ``duration`` / ``n_steps`` comes first.
        """
        # resolve_chunk_length lives in the light policies.base module (no torch);
        # imported lazily to match this file's policy-import convention.
        from .policies.base import resolve_chunk_length

        try:
            # A stop latched by a PREVIOUS rollout must not kill this one.
            # Created on demand for the same reason stop_task does it: the loop
            # must run on an instance whose __init__ did not complete.
            if getattr(self, "_stop_requested", None) is None:
                self._stop_requested = threading.Event()
            self._stop_requested.clear()

            # Per-rollout observation-resilience state. Defaults are applied here
            # too so the loop runs on an instance whose __init__ did not complete
            # (the same contract stop_task honours).
            for _attr, _default in (
                ("_obs_retries", 3),
                ("_obs_retry_backoff_s", 0.05),
                ("_max_consecutive_obs_failures", 20),
            ):
                if getattr(self, _attr, None) is None:
                    setattr(self, _attr, _default)
            self._consecutive_obs_failures = 0
            self._last_good_observation = None
            self._last_obs_error = ""
            self._dropped_action_steps = 0
            self._dropped_action_keys = []

            # Update task state
            self._task_state.status = TaskStatus.CONNECTING
            self._task_state.instruction = instruction
            self._task_state.start_time = time.time()
            self._task_state.step_count = 0
            self._task_state.error_message = ""

            # Connect to robot
            connected, connect_error = await self._connect_robot()
            if not connected:
                self._task_state.status = TaskStatus.ERROR
                self._task_state.error_message = connect_error or f"Failed to connect to {self.tool_name_str}"
                return

            # Honour a stop pressed DURING connect (a 2-3s window on a real
            # SO-101: motor-bus handshake + warmup_s per camera). Checked here,
            # before policy construction and before the first send_action, so the
            # arm never moves for a rollout the operator already cancelled.
            if self._stop_requested.is_set():
                self._task_state.status = TaskStatus.STOPPED
                logger.info("%s: stop requested during connect; aborting before any motion", self.tool_name_str)
                return

            # Get policy instance: a caller-supplied pre-built object wins;
            # otherwise build the server-backed one from provider + port.
            if policy_object is not None:
                policy_instance = policy_object
            else:
                policy_instance = await self._get_policy(policy_port, policy_host, policy_provider)

            # Initialize policy with robot state keys
            if not await self._initialize_policy(policy_instance):
                self._task_state.status = TaskStatus.ERROR
                self._task_state.error_message = "Failed to initialize policy"
                return

            # Clear the policy's per-episode state before the first inference.
            # A task boundary IS an episode boundary on hardware, but nothing
            # here used to say so: a second run_policy/start_task on the same
            # Robot with the same pre-built policy object began by REPLAYING the
            # previous task's leftover action chunk - measured
            # task1 [200.0, 201.0] then task2 [202.0, 203.0] with ZERO
            # inferences, i.e. motion generated for the previous scene applied
            # to a physical arm. LerobotLocalPolicy.reset clears the action
            # queue, the RTC previous-chunk/latency history and the observed
            # delay, and re-arms the zero-action and action-dim monitors; every
            # one of those otherwise survived the boundary. The sim runner does
            # the same per episode. Fail-soft (matching PolicyRunner): a reset
            # that raises must not abort a rollout the caller can still run.
            try:
                policy_instance.reset()
            except Exception as exc:  # noqa: BLE001 - reset is best-effort
                logger.warning(
                    "policy.reset() raised %s; continuing with possibly stale per-episode state "
                    "(a leftover action chunk from the previous task may be replayed)",
                    exc,
                )

            logger.info(f"Starting task: '{instruction}' on {self.tool_name_str}")
            if policy_object is not None:
                logger.info(f"Using pre-built policy object: {type(policy_instance).__name__}")
            else:
                logger.info(f"Using policy: {policy_provider} on {policy_host}:{policy_port}")

            # Real-Time Chunking contract (mirror PolicyRunner._run_policy_rollout
            # in strands_robots/simulation/policy_runner.py): tell the policy the
            # control rate ONCE before the rollout so RTC-capable providers
            # (pi0/pi0.5/SmolVLA/MolmoAct2) convert their inference latency into a
            # correct count of action steps and blend chunk seams identically to
            # sim. Without it a wrong assumed rate corrupts RTC blending at every
            # frequency except the assumed one. No-op for non-RTC policies.
            policy_instance.set_control_frequency(self.control_frequency)

            self._task_state.status = TaskStatus.RUNNING
            start_time = time.time()
            self._task_state.achieved_hz = 0.0
            # Deadline for the NEXT action, advanced by exactly one control
            # period per applied action. Sleeping to a running deadline instead
            # of sleeping a fixed period after each send_action is what keeps the
            # loop at control_frequency: the fixed sleep made observation and
            # inference time ADDITIVE to every period, so a 50Hz-configured loop
            # with a 33ms observation read and 120ms inference actually ran at
            # ~23Hz. That matters beyond throughput - line 1043 tells the policy
            # the declared rate, and RTC providers convert their inference
            # latency into a step count against it, so a rate the loop never
            # achieves corrupts every chunk-seam blend. Missed deadlines are
            # absorbed (max(0, ...)) rather than accumulating debt the loop would
            # try to repay with a burst of unsleeping servo writes.
            next_action_deadline = time.perf_counter()

            while (
                time.time() - start_time < duration
                and (n_steps is None or self._task_state.step_count < n_steps)
                and self._task_state.status == TaskStatus.RUNNING
                and not self._shutdown_event.is_set()
                and not self._stop_requested.is_set()
            ):
                # Get observation from robot, tolerating the transients a USB
                # camera bus produces under load. Returns None once the failure
                # budget is spent, which ends the rollout cleanly.
                observation = await self._read_observation_resiliently()
                if observation is None:
                    self._task_state.status = TaskStatus.ERROR
                    self._task_state.error_message = (
                        f"observation unavailable for {self._max_consecutive_obs_failures} consecutive "
                        f"control steps (last error: {self._last_obs_error}); stopping rather than "
                        f"driving the arm on a stale frame"
                    )
                    logger.error("%s: %s", self.tool_name_str, self._task_state.error_message)
                    break

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
                    if self._task_state.status != TaskStatus.RUNNING or self._stop_requested.is_set():
                        break
                    if n_steps is not None and self._task_state.step_count >= n_steps:
                        break
                    sent = await asyncio.to_thread(self.robot.send_action, action_dict)
                    self._task_state.step_count += 1
                    self._account_for_dropped_action(action_dict, sent)
                    # Hold the configured control period between actions, timed
                    # from the previous deadline rather than from now, so the
                    # send_action duration is absorbed by the period instead of
                    # being added to it.
                    next_action_deadline += self.action_sleep_time
                    await asyncio.sleep(max(0.0, next_action_deadline - time.perf_counter()))

            # Update final state
            elapsed = time.time() - start_time
            self._task_state.duration = elapsed
            steps = self._task_state.step_count
            if elapsed > 0:
                self._task_state.achieved_hz = steps / elapsed

            if self._task_state.status == TaskStatus.RUNNING:
                dropped = self._dropped_action_summary()
                if steps == 0:
                    # A rollout that applied NO action never drove the robot, so
                    # COMPLETED (which run_policy maps to status="success") would
                    # report motion that did not happen. The loop can exit at zero
                    # steps for several reasons - a latched shutdown, a duration
                    # that expired during connect, n_steps=0 - and none of them are
                    # a completed task.
                    self._task_state.status = TaskStatus.ERROR
                    self._task_state.error_message = (
                        "rollout applied 0 actions: the control loop exited before its first step "
                        "(check duration/n_steps, and whether this Robot was already shut down)"
                    )
                    logger.error("Task '%s' applied no actions: %s", instruction, self._task_state.error_message)
                elif dropped and self._dropped_action_steps >= steps > 0:
                    # EVERY step had refused keys: the policy and the driver do not
                    # agree on names at all, so reporting COMPLETED would present a
                    # rollout in which part of the arm never moved as a success.
                    self._task_state.status = TaskStatus.ERROR
                    self._task_state.error_message = dropped
                    logger.error("Task '%s' completed its steps but %s", instruction, dropped)
                else:
                    self._task_state.status = TaskStatus.COMPLETED
                    logger.info(
                        "Task completed: '%s' in %.1fs (%d steps, %.1fHz achieved of %.1fHz configured)%s",
                        instruction,
                        elapsed,
                        steps,
                        self._task_state.achieved_hz,
                        self.control_frequency,
                        f" -- WARNING: {dropped}" if dropped else "",
                    )

            # A loop that cannot keep up is not merely slow: the policy was told
            # control_frequency (RTC blends against it), so a large shortfall
            # means the blend used a timebase that never existed. Report it
            # rather than leaving the operator to assume the declared rate.
            self._warn_on_rate_shortfall(elapsed, steps)

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            self._task_state.status = TaskStatus.ERROR
            self._task_state.error_message = str(e)

    #: Fraction of the configured control_frequency the loop must sustain before
    #: the achieved rate is reported as a problem. Observation reads and
    #: inference always cost something, so a small shortfall is normal; below
    #: this the declared rate is misleading enough that RTC seam blending (which
    #: is computed against the declared rate) is materially wrong.
    _RATE_SHORTFALL_WARN_RATIO = 0.8

    def _warn_on_rate_shortfall(self, elapsed: float, steps: int) -> None:
        """Warn when the achieved control rate falls well short of the configured one.

        Args:
            elapsed: Wall-clock duration of the rollout in seconds.
            steps: Number of actions actually applied.
        """
        if elapsed <= 0 or steps <= 1:
            return
        achieved = steps / elapsed
        if achieved >= self.control_frequency * self._RATE_SHORTFALL_WARN_RATIO:
            return
        logger.warning(
            "%s ran at %.1fHz but is configured for %.1fHz (%.0f%% of target, %d actions in %.1fs). "
            "Observation reads and policy inference cost more than the control period, so the loop "
            "cannot keep up. The policy was told %.1fHz, and RTC-capable policies blend chunk seams "
            "against that rate, so the blending is using a rate that was never achieved. Lower "
            "control_frequency to the achievable rate, reduce camera count/resolution, or use a "
            "faster policy.",
            self.tool_name_str,
            achieved,
            self.control_frequency,
            100.0 * achieved / self.control_frequency,
            steps,
            elapsed,
            self.control_frequency,
        )

    def _acquire_task_slot(self) -> bool:
        """Take exclusive ownership of the control loop, without blocking.

        The motors bus is a single half-duplex RS-485 port and
        ``FeetechMotorsBus`` is not thread-safe, so exactly one loop may drive
        it. Callers that get ``False`` must reject rather than wait: blocking
        here would queue a second rollout that runs with a stale instruction
        the moment the first one ends.

        Returns:
            ``True`` when the caller now owns the loop and must call
            :meth:`_release_task_slot` in a ``finally``.
        """
        lock = getattr(self, "_task_lock", None)
        if lock is None:
            lock = self._task_lock = threading.Lock()
        return lock.acquire(blocking=False)

    def _release_task_slot(self) -> None:
        """Release the control-loop lock taken by :meth:`_acquire_task_slot`."""
        lock = getattr(self, "_task_lock", None)
        if lock is not None and lock.locked():
            lock.release()

    def _task_busy_error(self, caller: str) -> dict[str, Any] | None:
        """Reject ``caller`` when another control loop already owns the bus.

        Checked at every public entry point INSTEAD of comparing
        ``_task_state.status`` to ``RUNNING``: the loop spends the whole
        multi-second hardware bring-up in ``CONNECTING``, so the status check
        let a second caller through and both loops then interleaved
        ``sync_read``/``sync_write`` on one port.

        Args:
            caller: Name of the entry point, used in the log line only.

        Returns:
            A tool-shaped error when the loop is busy, else ``None``.
        """
        # A submitted-but-not-yet-started job holds no lock yet: start_task
        # returns as soon as it hands the job to the single-worker executor, so
        # for a moment nothing is running and nothing is locked. Accepting then
        # would QUEUE a second rollout that later runs unattended with its own
        # instruction. An outstanding future is therefore just as busy.
        pending = getattr(self._task_state, "task_future", None)
        if pending is not None and not pending.done():
            queued = self._task_state.instruction or "(starting)"
            logger.warning(
                "%s rejected on %s: a background task is already submitted (%s).",
                caller,
                self.tool_name_str,
                queued,
            )
            return {
                "status": "error",
                "content": [{"text": f"Task already running: {queued}"}],
            }
        # Belt and braces: the lock is the interlock, but a status that already
        # claims the loop is live means SOMETHING believes it owns the bus.
        # Refuse rather than reason about how the two disagreed. CONNECTING is
        # included because that is exactly the window the old check missed.
        if self._task_state.status in (TaskStatus.CONNECTING, TaskStatus.RUNNING):
            logger.warning(
                "%s rejected on %s: task state is %s.",
                caller,
                self.tool_name_str,
                self._task_state.status.value,
            )
            return {
                "status": "error",
                "content": [{"text": f"Task already running: {self._task_state.instruction or '(connecting)'}"}],
            }
        if self._acquire_task_slot():
            self._release_task_slot()
            return None
        logger.warning(
            "%s rejected on %s: a control loop is already driving the motors bus (status=%s). "
            "The bus is a single half-duplex port, so only one loop may own it; stop the running "
            "task first.",
            caller,
            self.tool_name_str,
            self._task_state.status.value,
        )
        running = self._task_state.instruction or "(connecting)"
        return {
            "status": "error",
            "content": [{"text": f"Task already running: {running}"}],
        }

    def _execute_task_sync(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        policy_object: Policy | None = None,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Execute task synchronously in thread - no new event loop."""

        # Import here to avoid conflicts
        import asyncio

        # The real interlock. Every path into the control loop funnels through
        # here (start_task's executor job, run_policy, and the "execute" tool
        # action), so holding the lock across the whole rollout is what
        # actually keeps two loops off one serial bus. The entry-point checks
        # only exist to return a clean error instead of racing to here.
        if not self._acquire_task_slot():
            busy = self._task_busy_error("_execute_task_sync")
            assert busy is not None  # the acquire above just failed
            return busy

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
            )

        # Use asyncio.run only if no loop is running, otherwise run in existing loop
        try:
            try:
                # Try to get the current event loop
                asyncio.get_running_loop()
                # If we're already in an event loop, we need to run in a thread
                with ThreadPoolExecutor() as exec:
                    future = exec.submit(lambda: asyncio.run(task_runner()))
                    future.result()  # Wait for completion
            except RuntimeError:
                # No event loop running - safe to create one
                asyncio.run(task_runner())
        finally:
            # Released even when the rollout raises, or a second rollout could
            # never start again.
            self._release_task_slot()

        # Return final status
        policy_desc = (
            f"{type(policy_object).__name__} (pre-built object)"
            if policy_object is not None
            else f"{policy_provider} on {policy_host}:{policy_port}"
        )
        return {
            "status": "success" if self._task_state.status == TaskStatus.COMPLETED else "error",
            "content": [
                {
                    "text": f"Task: '{instruction}' - {self._task_state.status.value}\n"
                    f"Robot: {self.tool_name_str} ({self.robot})\n"
                    f"Policy: {policy_desc}\n"
                    f"Duration: {self._task_state.duration:.1f}s\n"
                    f"Steps: {self._task_state.step_count}"
                    + (f"\nError: {self._task_state.error_message}" if self._task_state.error_message else "")
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
    ) -> dict[str, Any]:
        """Start robot task asynchronously and return immediately."""

        # Check if a control loop already owns the motors bus. Not a status
        # comparison: the loop is in CONNECTING for the whole bring-up, and
        # accepting here would queue a second rollout on the single-worker
        # executor that then runs unattended after this one finishes.
        if busy_error := self._task_busy_error("start_task"):
            return busy_error
        if shutdown_error := self._shutdown_guard("start_task"):
            return shutdown_error

        # Start task in background
        self._task_state.task_future = self._executor.submit(
            self._execute_task_sync, instruction, policy_port, policy_host, policy_provider, duration
        )

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Task started: '{instruction}'\n"
                    f"Robot: {self.tool_name_str}\n"
                    f"Use action='status' to check progress\n"
                    f"Use action='stop' to interrupt"
                }
            ],
        }

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

        Args:
            policy_object: A constructed ``Policy`` instance. The object's
                own device / embodiment / chunking configuration is honored;
                the loop only injects the robot state keys and the RTC
                control rate, exactly as it does for server-backed policies.
            instruction: Natural-language instruction passed to the policy on
                every ``get_actions`` call.
            duration: Wall-clock budget in seconds (same default as
                ``start_task``).
            n_steps: Optional cap on applied actions (mirrors the sim
                ``run_policy`` parameter); the loop stops at whichever of
                ``duration`` / ``n_steps`` comes first.

        Returns:
            Tool-shaped result: a text summary plus a ``{"json": ...}`` block
            carrying ``status`` / ``steps`` / ``duration_s`` / ``instruction``
            / ``policy`` (and ``error`` when one occurred).
        """
        if policy_object is None:
            return {
                "status": "error",
                "content": [{"text": "policy_object is required (for the provider+port path use start_task)"}],
            }
        if busy_error := self._task_busy_error("run_policy"):
            return busy_error
        if shutdown_error := self._shutdown_guard("run_policy"):
            return shutdown_error

        loop_result = self._execute_task_sync(
            instruction, duration=duration, policy_object=policy_object, n_steps=n_steps
        )
        # The guard above and the lock inside are separate operations, so a
        # caller can still lose the race between them. When that happens the
        # loop never ran, and _task_state belongs to the OTHER rollout - report
        # its own rejection rather than the other loop's step count.
        if loop_result.get("status") == "error" and "Task already running" in str(loop_result.get("content", "")):
            return loop_result

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

    def get_task_status(self) -> dict[str, Any]:
        """Get current task execution status."""

        # Update duration for running tasks
        if self._task_state.status == TaskStatus.RUNNING:
            self._task_state.duration = time.time() - self._task_state.start_time

        status_text = f"Robot Status: {self._task_state.status.value.upper()}\n"

        # Connection state matters to an agent deciding what to do next, and
        # the tool dispatch exposes no other action that reports it.
        status_text += f"Connected: {getattr(self.robot, 'is_connected', False)}\n"

        # Connection state matters to an agent deciding what to do next, and
        # the tool dispatch exposes no other action that reports it.
        status_text += f"Connected: {getattr(self.robot, 'is_connected', False)}\n"

        if self._task_state.instruction:
            status_text += f"Task: {self._task_state.instruction}\n"

        if self._task_state.status == TaskStatus.RUNNING:
            status_text += f"Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"Steps: {self._task_state.step_count}\n"
        elif self._task_state.status in [TaskStatus.COMPLETED, TaskStatus.STOPPED, TaskStatus.ERROR]:
            status_text += f"Total Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"Total Steps: {self._task_state.step_count}\n"
            # Report the rate the loop actually sustained next to the configured
            # one. They differ by the per-observation cost (camera reads +
            # inference), and the gap is what RTC seam blending is wrong by.
            if self._task_state.achieved_hz > 0:
                status_text += (
                    f"Control Rate: {self._task_state.achieved_hz:.1f}Hz achieved "
                    f"of {self.control_frequency:.1f}Hz configured\n"
                )
            # A partial key mismatch means part of the arm never moved; the step
            # count alone cannot show that, so report it alongside.
            dropped_summary = self._dropped_action_summary()
            if dropped_summary:
                status_text += f"Dropped Actions: {dropped_summary}\n"

        if self._task_state.error_message:
            status_text += f"Error: {self._task_state.error_message}\n"

        return {
            "status": "success",
            "content": [{"text": status_text}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Stop the running task, including one still connecting.

        Sets :attr:`_stop_requested` UNCONDITIONALLY and BEFORE any status check.
        The status guard used to reject everything that was not ``RUNNING``, but
        ``_execute_task_async`` sits in ``CONNECTING`` for the whole hardware
        bring-up - a FeetechMotorsBus handshake plus ``warmup_s`` per camera,
        2-3 s on a real SO-101 and longer on a multi-camera rig. Every stop
        pressed in that window returned "No task running to stop" with
        ``status="success"`` and did nothing; the arm then started moving anyway
        (measured: 3 stop presses ignored, then 104 servo writes and a
        ``COMPLETED`` status). A latch that is set regardless of state means the
        rollout aborts before its first ``send_action`` no matter when the stop
        lands.

        ``mesh/core.py``'s fleet ``{"action": "stop"}`` dispatch routes straight
        here, so the mesh e-stop inherits the same fix.

        Returns:
            Status dict. ``success`` when a task was stopped or a stop was
            latched; the text distinguishes the two so an operator can tell a
            real stop from a no-op on an idle robot.
        """
        # Latch first, unconditionally: a stop must never be lost to a status
        # the guard below does not happen to recognise. Created on demand because
        # a stop is the one call that must work even on a Robot whose __init__
        # did not complete (a half-built instance is exactly when an operator
        # reaches for the stop).
        stop_latch = getattr(self, "_stop_requested", None)
        if stop_latch is None:
            stop_latch = self._stop_requested = threading.Event()
        stop_latch.set()

        if self._task_state.status not in (TaskStatus.RUNNING, TaskStatus.CONNECTING):
            return {
                "status": "success",
                "content": [{"text": f"No task running to stop (current: {self._task_state.status.value})"}],
            }

        # Signal task to stop
        self._task_state.status = TaskStatus.STOPPED

        # Cancel future if it exists
        if self._task_state.task_future:
            self._task_state.task_future.cancel()

        logger.info(f"Task stopped: {self._task_state.instruction}")

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Task stopped: '{self._task_state.instruction}'\n"
                    f"Duration: {self._task_state.duration:.1f}s\n"
                    f"Steps completed: {self._task_state.step_count}"
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
        return {
            "name": self.tool_name_str,
            "description": f"Universal robot control with async task execution ({self.robot}). "
            f"Actions: execute (blocking), start (async), status, stop. "
            f"For execute/start actions: instruction and policy_port are required. "
            f"For status/stop actions: no additional parameters needed.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform: execute (blocking), start (async), status, stop",
                            "enum": ["execute", "start", "status", "stop"],
                            "default": "execute",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction (required for execute/start actions)",
                        },
                        "policy_port": {
                            "type": "integer",
                            "description": "Policy service port (required for execute/start actions)",
                        },
                        "policy_host": {
                            "type": "string",
                            "description": "Policy service host (default: localhost)",
                            "default": "localhost",
                        },
                        "policy_provider": {
                            "type": "string",
                            "description": "Policy provider (groot, openai, etc.)",
                            "default": "groot",
                        },
                        "duration": {
                            "type": "number",
                            "description": "Maximum execution time in seconds",
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

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        """Stream robot task execution with async actions."""
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})

            action = input_data.get("action", "execute")

            # Handle different actions
            if action == "execute":
                # Blocking execution (legacy behavior)
                instruction = input_data.get("instruction", "")
                policy_port = input_data.get("policy_port")
                policy_host = input_data.get("policy_host", "localhost")
                policy_provider = input_data.get("policy_provider", "groot")
                duration = input_data.get("duration", 30.0)

                if not instruction or not policy_port:
                    yield ToolResultEvent(
                        self._make_tool_result(
                            tool_use_id,
                            {
                                "status": "error",
                                "content": [{"text": "instruction and policy_port are required for execute action"}],
                            },
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

                if not instruction or not policy_port:
                    yield ToolResultEvent(
                        self._make_tool_result(
                            tool_use_id,
                            {
                                "status": "error",
                                "content": [{"text": "instruction and policy_port are required for start action"}],
                            },
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
                        {
                            "status": "error",
                            "content": [
                                {"text": f"Unknown action: {action}. Valid actions: execute, start, status, stop"}
                            ],
                        },
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
        """Cleanup resources and stop any running tasks."""
        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Stop any local teleoperation loop + disconnect attached devices
            # (TeleopMixin). Best-effort: a teleop teardown failure must not
            # block the rest of hardware cleanup.
            if getattr(self, "_teleop_running", False) or getattr(self, "_teleops", None):
                try:
                    self.stop_teleoperate()
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

            # Disconnect the hardware (motor bus + cameras). Camera read
            # threads left alive at interpreter exit abort the whole process
            # on some platforms (glibc "FATAL: exception not rethrown" from
            # cv2's pthreads), so cleanup() must release the device even when
            # the caller forgot to call disconnect()/stop() first.
            robot = getattr(self, "robot", None)
            if robot is not None and getattr(robot, "is_connected", False):
                try:
                    robot.disconnect()
                except Exception as disconnect_exc:  # noqa: BLE001 - best-effort teardown
                    logger.warning(
                        "%s: robot.disconnect() raised during cleanup: %s",
                        self.tool_name_str,
                        disconnect_exc,
                    )

            # Disconnect the hardware (motor bus + cameras). Camera read
            # threads left alive at interpreter exit abort the whole process
            # on some platforms (glibc "FATAL: exception not rethrown" from
            # cv2's pthreads), so cleanup() must release the device even when
            # the caller forgot to call disconnect()/stop() first.
            robot = getattr(self, "robot", None)
            if robot is not None and getattr(robot, "is_connected", False):
                try:
                    robot.disconnect()
                except Exception as disconnect_exc:  # noqa: BLE001 - best-effort teardown
                    logger.warning(
                        "%s: robot.disconnect() raised during cleanup: %s",
                        self.tool_name_str,
                        disconnect_exc,
                    )

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
            self._shutdown_ros_bridge()

            logger.info(f"{self.tool_name_str} cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error for {self.tool_name_str}: {e}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors in destructor

    async def get_status(self) -> dict[str, Any]:
        """Get robot status including connection and task state."""
        try:
            # Get robot connection status
            is_connected = self.robot.is_connected if hasattr(self.robot, "is_connected") else False
            # lerobot's ``is_calibrated`` property reads the motor bus and
            # raises when disconnected (``hasattr`` only swallows
            # AttributeError, so it propagates). Report None ("unknown") until
            # connected instead of collapsing into the error dict below.
            is_calibrated: bool | None = None
            if is_connected:
                is_calibrated = self.robot.is_calibrated if hasattr(self.robot, "is_calibrated") else True

            # Get camera status
            camera_status = []
            if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
                for name in self.robot.config.cameras.keys():
                    camera_status.append(name)

            # Build status dict
            status_data = {
                "robot_name": self.tool_name_str,
                "robot_type": getattr(self.robot, "robot_type", self.robot.name),
                "robot_info": str(self.robot),
                "data_config": self.data_config,
                "is_connected": is_connected,
                "is_calibrated": is_calibrated,
                "cameras": camera_status,
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
        delegates to ``self.robot.send_action``. ``robot_name`` is accepted
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
            if not getattr(self.robot, "is_connected", False):
                # Lazy connect on first action. calibrate=False: a teleop
                # session assumes the follower is already calibrated (same
                # contract as the policy-run path).
                try:
                    self.robot.connect(False)
                except Exception:
                    self._rollback_half_open_connect()
                    raise
            self.robot.send_action(action)
            return {"status": "success", "content": [{"text": "ok"}]}
        except Exception as e:  # noqa: BLE001 - surface as status, never kill the loop
            logger.error("%s send_action failed: %s", self.tool_name_str, e)
            return {
                "status": "error",
                "content": [{"text": f"{self.tool_name_str} send_action error: {e}"}],
            }

    async def stop(self) -> None:
        """Stop robot and disconnect."""
        try:
            # Stop any running task first
            if self._task_state.status == TaskStatus.RUNNING:
                self.stop_task()

            # Disconnect robot hardware
            if hasattr(self.robot, "disconnect"):
                await asyncio.to_thread(self.robot.disconnect)

            # Cleanup resources
            self.cleanup()

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
            teleoperator: Any object with a ``get_action() -> dict`` method.
                Typically a lerobot Teleoperator (SOLeader, GamepadTeleop,
                KeyboardTeleop, Phone).
            device_name: Name for this input stream (e.g. "leader", "gamepad").
            method: Input method label ("arm", "gamepad", "keyboard", "phone").
            hz: Publishing frequency in Hz. Must be a positive finite number;
                the publish loop's period is ``1 / hz``.

        Returns:
            Status dict with topic and peer_id for the receiver to use, or an
            error dict when the mesh is inactive, ``device_name`` is not a
            valid mesh identifier, or ``hz`` is not a rate the publish loop
            can honor.
        """
        if not self.mesh or not self.mesh.alive:
            return {"status": "error", "content": [{"text": "Mesh not active. Cannot publish input."}]}

        from strands_robots.mesh.security import ValidationError, validate_mesh_identifier

        # Both arguments are validated up front, before the teardown of any
        # publisher already registered under this device name: a rejected call
        # must not stop a live stream. ``device_name`` becomes a segment of the
        # published key expression and a key in ``_input_publishers``, and
        # ``hz`` sets the publish loop's ``1 / hz`` period. Report through the
        # tool envelope rather than raising.
        try:
            validate_mesh_identifier(device_name, "start_teleop_publish.device_name")
        except ValidationError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}

        error = positive_finite_number_error(hz, "hz", "start_teleop_publish")
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

    def start_teleop_receive(
        self,
        source_peer_id: str,
        device_name: str = "leader",
        apply_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Start receiving teleoperator actions from a remote peer and applying to hardware.

        This makes the robot a *teleop follower*: it listens for input frames
        published by the source peer and applies them to its own hardware via
        ``self.robot.send_action(action)``.

        Args:
            source_peer_id: The peer ID of the publishing robot.
            device_name: Name of the input stream to subscribe to.
            apply_fn: Optional custom function ``(robot, action_dict) -> None``.
                Defaults to calling ``robot.send_action(action)``.

        Returns:
            Status dict, or an error dict when the mesh is inactive or
            ``source_peer_id`` / ``device_name`` is not a valid mesh
            identifier.
        """
        if not self.mesh or not self.mesh.alive:
            return {"status": "error", "content": [{"text": "Mesh not active. Cannot receive input."}]}

        from strands_robots.mesh.security import ValidationError, validate_mesh_identifier

        # Both identifiers become segments of the subscribed key expression, so
        # a Zenoh wildcard here would make this follower apply joint commands
        # from every peer instead of the named leader. Validate before stopping
        # any existing receiver for that key so a rejected call cannot tear
        # down a live stream, and report through the tool envelope rather than
        # raising past dispatch.
        try:
            validate_mesh_identifier(source_peer_id, "start_teleop_receive.source_peer_id")
            validate_mesh_identifier(device_name, "start_teleop_receive.device_name")
        except ValidationError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}

        from strands_robots.mesh import InputReceiver

        if not hasattr(self, "_input_receivers"):
            self._input_receivers: dict[str, InputReceiver] = {}

        key = f"{source_peer_id}/{device_name}"
        if key in self._input_receivers:
            self._input_receivers[key].stop()

        receiver = InputReceiver(
            mesh=self.mesh,
            robot=self.robot,
            source_peer_id=source_peer_id,
            device_name=device_name,
            apply_fn=apply_fn,
        )
        receiver.start()
        self._input_receivers[key] = receiver

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Input receiver started: listening to {source_peer_id}/{device_name}\n"
                    f"Topic: {receiver.topic}\n"
                    f"Actions will be applied to: {self.tool_name_str}"
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

    def get_teleop_status(self) -> dict[str, Any]:
        """Get status of all active teleop sessions."""
        publishers = {}
        receivers = {}

        if hasattr(self, "_input_publishers"):
            for name, pub in self._input_publishers.items():
                publishers[name] = pub.stats

        if hasattr(self, "_input_receivers"):
            for key, rcv in self._input_receivers.items():
                receivers[key] = rcv.stats

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Teleop status:\n"
                    f"  Publishers: {len(publishers)} active\n"
                    f"  Receivers: {len(receivers)} active\n"
                    + "".join(
                        f"  [pub] {n}: {s.get('frames', 0)} frames @ {s.get('hz_actual', 0):.1f}Hz\n"
                        for n, s in publishers.items()
                    )
                    + "".join(
                        f"  [rcv] {k}: {s.get('frames_received', 0)} frames @ {s.get('hz_actual', 0):.1f}Hz\n"
                        for k, s in receivers.items()
                    )
                },
                {"json": {"publishers": publishers, "receivers": receivers}},
            ],
        }

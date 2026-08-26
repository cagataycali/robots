"""Native CycloneDDS driver for the Unitree G1.

``Robot("g1", mode="real", driver="strands", port=<ip>, network_interface="eth0")``
builds one of these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver`, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, teleop rail and
agent tool surface consume it exactly like the lerobot driver they replace for
this robot.

What the driver actually does:

* Subscribes ``rt/lowstate``, ``rt/lf/bmsstate``, ``rt/utlidar/lidar_state``
  and ``rt/utlidar/cloud_livox_mid360`` on a background DDS thread. Each
  callback drops into an in-memory cache the mesh reads at its own cadence
  (:mod:`strands_robots.mesh.sensors` publishes ``_imu``, ``_battery``,
  ``_lidar_state`` and ``_lidar_summary`` from those caches).
* Gates writes on the FSM: :meth:`send_action` refuses when the low-state
  ``mode_machine`` is outside :data:`~strands_robots.tools.g1.HANDSHAKE_FSMS`
  or the battery is under the floor.
* Task and policy paths (``start_task``, ``run_policy``, ``stop_task``,
  ``get_task_status``) return a named "not wired yet" envelope. Locomotion
  and arm actions land here in issue #358; the driver's job in issue #354
  is the transport, and shipping empty stubs makes the omission surface at
  call time instead of at import time.

Nothing in this module imports ``unitree_sdk2py`` at module load. Every SDK
touch is inside a function body, so the module can be imported on Thor, on CI,
and in every unit test with a mocked bus.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from strands_robots.tools.g1 import HANDSHAKE_FSMS, decode_code
from strands_robots.tools.g1._dds_engine import DDSSubscriberSet

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants surfaced for tests and for issue #358 to share. The
# ``_LIDAR_MAX_POINTS`` cap is deliberate: :meth:`_summarise_cloud` runs on the
# DDS thread and a full Livox frame is ~30k points at 10 Hz. The point-cloud
# tile (issue #356) publishes its own downsampled ``lidar/cloud`` topic; here
# we only compute a summary.
# ---------------------------------------------------------------------------
_LIDAR_MAX_POINTS: int = 4000
_BATTERY_FLOOR_PCT: float = 15.0

# The topics the driver reads. Kept together so a reader sees the whole
# subscription set in one place.
_TOPIC_LOWSTATE = "rt/lowstate"
_TOPIC_BMS = "rt/lf/bmsstate"
_TOPIC_LIDAR_STATE = "rt/utlidar/lidar_state"
_TOPIC_LIDAR_CLOUD = "rt/utlidar/cloud_livox_mid360"


class G1Driver:
    """Native driver for the Unitree G1.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally
    - a Protocol - so no import from :mod:`strands_robots.drivers.base` is
    needed. The class-level surface check that
    :func:`~strands_robots.drivers.register_native_driver` runs at registration
    time is what pins the contract for this class.
    """

    def __init__(
        self,
        tool_name: str = "g1",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        network_interface: str = "eth0",
        battery_floor_pct: float = _BATTERY_FLOOR_PCT,
        lidar_max_points: int = _LIDAR_MAX_POINTS,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` does the DDS work.

        The three positional-ish arguments are the ones every native driver
        takes - see :mod:`strands_robots.drivers.base`'s constructor contract
        - so the factory can build any driver the same way.

        Args:
            tool_name: Name the agent invokes the driver by. Also the mesh
                peer id when the driver is wrapped by
                :class:`~strands_robots.mesh.Mesh`.
            cameras: Camera configuration accepted for parity with the
                lerobot driver; unused here because the G1's onboard cameras
                are addressed through the DDS bus, not v4l2.
            data_config: Data configuration name accepted for parity; unused.
            port: The robot's IP address. Recorded but not touched by this
                driver: CycloneDDS binds to a NIC, not an address. Kept for
                logging and future SSH-side helpers.
            network_interface: The interface CycloneDDS binds to. Passed to
                :func:`~strands_robots.tools.g1.ensure_dds`.
            battery_floor_pct: Percentage below which :meth:`send_action`
                refuses to write. The floor is separate from the FSM gate so
                a caller can see which check refused.
            lidar_max_points: Cap for the downsampled point count reported in
                ``_lidar_summary``. Full clouds go through the dashboard's
                own topic (issue #356), not this one.
            **kwargs: Ignored; accepted so the factory can forward extras
                without the driver knowing what they are.
        """
        del cameras, data_config  # accepted for parity; unused here
        if kwargs:
            logger.debug("G1Driver ignoring extra kwargs: %s", sorted(kwargs))
        self._tool_name = tool_name
        self._port = port
        self._network_interface = network_interface
        self._battery_floor_pct = float(battery_floor_pct)
        self._lidar_max_points = int(lidar_max_points)

        # Cached DDS decode results. Every one is optional per the mesh
        # contract, so a driver that has not connected yet is not broken.
        # Populated on the DDS thread; read by mesh loops. A lock guards the
        # tiny critical section around dict swap; readers snapshot.
        self._cache_lock = threading.Lock()
        self._imu: dict[str, Any] | None = None
        self._battery: dict[str, Any] | None = None
        self._lidar_state: dict[str, Any] | None = None
        self._lidar_summary: dict[str, Any] | None = None
        self._fsm_id: int | None = None

        # Populated by :meth:`connect_eagerly`. ``None`` on a machine that
        # never connected - a valid state for tests and imports.
        self._subs: DDSSubscriberSet | None = None
        self._connected: bool = False
        self._connect_error: str | None = None

    # ------------------------------------------------------------------ #
    # Agent tool surface (matches AgentTool's abstract members).         #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Always ``"robot"`` - mirrors the lerobot driver."""
        return "robot"

    @property
    def tool_spec(self) -> ToolSpec:
        """A minimal agent-facing spec.

        The rich verb set (``arm``, ``walk``, ``posture``, ``speak``,
        ``lidar_snapshot``) lands in issue #358 as vendored neon tools that
        the agent gets in addition to the driver-as-tool. Here we ship the
        universal ``status``/``stop`` verbs and a ``sensors`` read-out so a
        Strands agent can already introspect the robot the day the driver
        merges.
        """
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Unitree G1 native driver: reads the G1's CycloneDDS bus for "
                    "IMU, battery, lidar state and a lidar summary; motion and "
                    "policy paths land in the g1_tools bundle."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors: return the latest cached IMU/battery/lidar; "
                                    "status: report connection and FSM; "
                                    "stop: refuse further writes (a no-op today until #358 lands the motion verbs)"
                                ),
                                "enum": ["sensors", "status", "stop"],
                                "default": "sensors",
                            },
                        },
                        "required": ["action"],
                    }
                },
            },
        )

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Handle one agent invocation and yield exactly one tool result.

        The three verbs the spec declares are read-only or a no-op; each maps
        to a dict already computed by the DDS callbacks. Motion belongs to
        issue #358; this driver refuses ``stop`` in the same envelope shape it
        will use once the motion path is wired, so a caller writes the same
        error-checking code either way.
        """
        del kwargs  # forward-compat only
        del invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "sensors")
        if action == "sensors":
            envelope = {
                "status": "success",
                "content": [
                    {
                        "json": {
                            "imu": self._snapshot("_imu"),
                            "battery": self._snapshot("_battery"),
                            "lidar_state": self._snapshot("_lidar_state"),
                            "lidar_summary": self._snapshot("_lidar_summary"),
                        }
                    }
                ],
            }
        elif action == "status":
            envelope = {
                "status": "success",
                "content": [{"json": await self.get_status()}],
            }
        else:  # "stop"
            await self.stop()
            envelope = {
                "status": "success",
                "content": [{"text": "stop: no motion path wired yet (issue #358)"}],
            }
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Lifecycle and status.                                              #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Attach to the DDS bus and subscribe every sensor topic.

        Called by :func:`~strands_robots.robot.Robot` before returning the
        driver, so a real bring-up fails on the connect line rather than on
        the first :meth:`get_status` poll. The return type is a string so a
        caller who wants a soft failure (a headless smoke test that only
        wants the driver instance) can decide whether to raise.

        Returns:
            ``None`` on success. A named reason on failure - the driver is
            left disconnected but usable, so a mesh peer for a robot that is
            off can still be constructed for later use.
        """
        subs = DDSSubscriberSet(self._network_interface)
        err = subs.start()
        if err is not None:
            self._connect_error = err
            return err
        for topic, cls_path, decoder in self._subscription_plan():
            message_class = _resolve_message_class(cls_path)
            if isinstance(message_class, str):
                self._connect_error = message_class
                return message_class
            err = subs.subscribe(topic, message_class, decoder)
            if err is not None:
                self._connect_error = err
                return err
        self._subs = subs
        self._connected = True
        self._connect_error = None
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report the driver's connection and FSM state.

        The shape matches the lerobot driver's ``get_status`` envelope so the
        mesh publishes both peers identically. Not every field is populated
        until the DDS subscribers deliver messages - a driver that just
        connected reports what it has.
        """
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self._connected,
                        "connect_error": self._connect_error,
                        "port": self._port,
                        "network_interface": self._network_interface,
                        "fsm_id": self._fsm_id,
                        "battery_pct": (self._battery or {}).get("pct"),
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Refuse further writes; keep the DDS connection.

        The motion path lands in issue #358 - until then there is nothing to
        stop. The method exists because the driver contract requires it and
        the mesh calls it during shutdown; the no-op has the right shape for
        the day motion is wired.
        """
        logger.debug("%s.stop() no-op (no motion path wired yet)", self._tool_name)

    def cleanup(self) -> None:
        """Release every DDS subscriber. Idempotent."""
        if self._subs is not None:
            self._subs.close()
            self._subs = None
        self._connected = False

    # ------------------------------------------------------------------ #
    # Command path.                                                      #
    # ------------------------------------------------------------------ #

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Refuse writes today; the FSM and battery gates are already live.

        The motion writes to ``rt/armsdk`` and ``rt/lowcmd`` land in issue
        #358 - but the gates that a real ``send_action`` will consult are
        cheap to install now, and installing them here means the day the
        write lines land the gates already have coverage. The envelope
        matches what a rejected motion call will return.
        """
        del action, robot_name  # gates run before we would use them
        if not self._connected:
            return _refuse("not connected - call connect_eagerly() first")
        if self._fsm_id is None:
            return _refuse("FSM id unknown - lowstate has not delivered yet")
        if self._fsm_id not in HANDSHAKE_FSMS:
            return _refuse(f"FSM {self._fsm_id} refuses arm/loco writes; needs one of {sorted(HANDSHAKE_FSMS)}")
        battery_pct = (self._battery or {}).get("pct")
        if battery_pct is not None and battery_pct < self._battery_floor_pct:
            return _refuse(f"battery {battery_pct:.1f}% is under floor {self._battery_floor_pct:.1f}%")
        return _refuse("motion path not wired yet (issue #358)")

    # ------------------------------------------------------------------ #
    # Task and policy paths (stubs until #358).                          #
    # ------------------------------------------------------------------ #

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Report that policy tasks are not wired yet.

        The lerobot driver runs its ``start_task`` through the same policy
        provider registry the sim uses; wiring it here needs the g1_tools
        motion verbs (issue #358) so the loop has something to command. The
        stub returns the shape that lerobot returns, so a caller polls the
        same fields either way.
        """
        del instruction, policy_port, policy_host, policy_provider
        del duration, policy_kwargs
        return _refuse("start_task not wired yet (issue #358)")

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Report that policy rollouts are not wired yet - see :meth:`start_task`."""
        del policy_object, instruction, duration, n_steps
        return _refuse("run_policy not wired yet (issue #358)")

    def get_task_status(self) -> dict[str, Any]:
        """Report that no task is running (there is no way to start one yet)."""
        return {
            "status": "success",
            "content": [{"json": {"running": False, "reason": "no task path wired yet (issue #358)"}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """No-op: there is no running task to stop until :meth:`start_task` is wired."""
        return {
            "status": "success",
            "content": [{"text": "no task path wired yet (issue #358)"}],
        }

    # ------------------------------------------------------------------ #
    # DDS decoders. Each runs on the DDS thread; keep fast and pure.     #
    # ------------------------------------------------------------------ #

    def _subscription_plan(self) -> list[tuple[str, tuple[str, str], Any]]:
        """Return ``(topic, (idl_module, idl_class), decoder)`` for every topic.

        Kept as a method so a subclass can override, and so a test can call it
        to walk the plan without touching the DDS bus.
        """
        return [
            (
                _TOPIC_LOWSTATE,
                ("unitree_sdk2py.idl.unitree_hg.msg.dds_", "LowState_"),
                self._on_lowstate,
            ),
            (
                _TOPIC_BMS,
                ("unitree_sdk2py.idl.unitree_hg.msg.dds_", "BmsState_"),
                self._on_bms,
            ),
            (
                _TOPIC_LIDAR_STATE,
                ("unitree_sdk2py.idl.unitree_go.msg.dds_", "LidarState_"),
                self._on_lidar_state,
            ),
            (
                _TOPIC_LIDAR_CLOUD,
                ("unitree_sdk2py.idl.sensor_msgs.msg.dds_", "PointCloud2_"),
                self._on_lidar_cloud,
            ),
        ]

    def _on_lowstate(self, msg: Any) -> None:
        """Decode ``rt/lowstate`` into :attr:`_imu` and :attr:`_fsm_id`."""
        try:
            imu = getattr(msg, "imu_state", None)
            if imu is not None:
                self._imu = {
                    "rpy": [float(x) for x in getattr(imu, "rpy", [0.0, 0.0, 0.0])[:3]],
                    "gyroscope": [float(x) for x in getattr(imu, "gyroscope", [0.0, 0.0, 0.0])[:3]],
                    "accelerometer": [float(x) for x in getattr(imu, "accelerometer", [0.0, 0.0, 0.0])[:3]],
                    "quaternion": [float(x) for x in getattr(imu, "quaternion", [1.0, 0.0, 0.0, 0.0])[:4]],
                    "t": time.time(),
                }
            mode_machine = getattr(msg, "mode_machine", None)
            if mode_machine is not None:
                self._fsm_id = int(mode_machine)
        except Exception as exc:  # noqa: BLE001 - IDL message can be anything
            logger.debug("%s: lowstate decode failed: %s", self._tool_name, exc)

    def _on_bms(self, msg: Any) -> None:
        """Decode ``rt/lf/bmsstate`` into :attr:`_battery`."""
        try:
            soc = getattr(msg, "soc", None)
            self._battery = {
                "pct": float(soc) if soc is not None else None,
                "charging": bool(getattr(msg, "charge", 0)),
                "current": float(getattr(msg, "current", 0.0)),
                "cycle": int(getattr(msg, "cycle", 0)),
                "t": time.time(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s: bmsstate decode failed: %s", self._tool_name, exc)

    def _on_lidar_state(self, msg: Any) -> None:
        """Decode ``rt/utlidar/lidar_state`` into :attr:`_lidar_state`."""
        try:
            self._lidar_state = {
                "code": int(getattr(msg, "code", -1)),
                "code_text": decode_code(getattr(msg, "code", -1)),
                "freq": float(getattr(msg, "freq", 0.0)),
                "sys_rotation_speed": float(getattr(msg, "sys_rotation_speed", 0.0)),
                "t": time.time(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s: lidar_state decode failed: %s", self._tool_name, exc)

    def _on_lidar_cloud(self, msg: Any) -> None:
        """Compute a bounded summary of the Livox cloud into :attr:`_lidar_summary`.

        A full ``PointCloud2_`` from the MID-360 is ~30k points at 10 Hz - way
        too much to publish unpaced. The summary is what the mesh's health
        chip reads; the 3D tile (issue #356) subscribes the raw cloud itself
        through a paced publisher and does its own downsampling.
        """
        try:
            width = int(getattr(msg, "width", 0))
            height = int(getattr(msg, "height", 0))
            count = width * height
            self._lidar_summary = {
                "count": count,
                "capped_at": self._lidar_max_points,
                "width": width,
                "height": height,
                "point_step": int(getattr(msg, "point_step", 0)),
                "row_step": int(getattr(msg, "row_step", 0)),
                "t": time.time(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s: lidar_cloud summary failed: %s", self._tool_name, exc)

    # ------------------------------------------------------------------ #
    # Internal helpers.                                                  #
    # ------------------------------------------------------------------ #

    def _snapshot(self, attr: str) -> dict[str, Any] | None:
        """Return a copy of one cached sensor dict, or ``None``.

        Copy so a caller who mutates the result does not corrupt the cache
        the DDS thread writes into. Small dicts; cheap.
        """
        with self._cache_lock:
            value = getattr(self, attr, None)
            if value is None:
                return None
            if isinstance(value, dict):
                return dict(value)
            return value  # type: ignore[unreachable]


def _refuse(reason: str) -> dict[str, Any]:
    """Return the driver's error envelope with ``reason`` inside.

    Kept as a free function so every refusal path renders the same shape and
    a test can grep for the reason without unpacking the envelope by hand.
    """
    return {
        "status": "error",
        "content": [{"text": reason}],
    }


def _resolve_message_class(cls_path: tuple[str, str]) -> Any:
    """Return the IDL class for ``(module_path, class_name)`` or a reason string.

    Lazy import so the driver module stays importable without the SDK. Called
    from :meth:`G1Driver.connect_eagerly`, which turns a returned string into
    a named connect failure and leaves the driver in the "usable but not
    connected" state.
    """
    module_path, class_name = cls_path
    try:
        import importlib

        module = importlib.import_module(module_path)
    except ImportError as exc:
        return f"cannot import {module_path}: {exc}"
    if not hasattr(module, class_name):
        return f"{module_path} has no {class_name}"
    return getattr(module, class_name)

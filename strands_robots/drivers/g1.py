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
* Gates writes on the FSM: :meth:`send_action` refuses when the FSM state
  is outside :data:`~strands_robots.tools.g1.HANDSHAKE_FSMS` or the battery
  is under the floor.  The gate consults :attr:`_fsm_id` (the high-level
  FSM state from the motion-switcher API) rather than :attr:`_mode_machine`
  (the uint8 hardware-layout id from ``LowState``); those two fields have
  disjoint value ranges and must not be conflated.  Until the
  motion-switcher source is wired (harness#361 PR-C, #2765), the gate
  refuses honestly rather than silently rejecting every real frame.
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

from strands_robots.tools.g1 import HANDSHAKE_FSMS, WALK_FSMS, decode_code
from strands_robots.tools.g1._dds_engine import DDSPublisher, DDSSubscriberSet

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants surfaced for tests and for issue #358 to share.
# ---------------------------------------------------------------------------
_BATTERY_FLOOR_PCT: float = 15.0

# The topics the driver reads. Kept together so a reader sees the whole
# subscription set in one place.
_TOPIC_LOWSTATE = "rt/lowstate"
_TOPIC_BMS = "rt/lf/bmsstate"
_TOPIC_LIDAR_STATE = "rt/utlidar/lidar_state"
_TOPIC_LIDAR_CLOUD = "rt/utlidar/cloud_livox_mid360"

# The topic the driver writes.  ``rt/lowcmd`` carries a full ``LowCmd_`` shaped
# for the G1 wholebody actuator set - motion cannot go anywhere else without
# also crossing the FSM handshake, so the write path is a single-topic path.
_TOPIC_LOWCMD = "rt/lowcmd"

# The G1 wholebody motor slot count.  ``LowCmd_.motor_cmd`` is a fixed-length
# array of ``MotorCmd_`` (see :mod:`unitree_sdk2py.idl.unitree_hg.msg.dds_`);
# 35 is the wholebody layout the current firmware ships with, and slot 29
# (``kNotUsedJoint``) is the arm-SDK enable byte the ``LowCmd_`` protocol
# reserves - the driver leaves the enable byte at whatever the caller set.
# _G1_MOTOR_SLOTS was declared here and never read - the ``LowCmd_.motor_cmd``
# array is fixed by the IDL, so the constant is redundant.  Removed to keep
# the module surface honest and to close CodeQL's unused-global alert.

# The joint-name -> slot index mapping the driver accepts in
# :meth:`G1Driver.send_action`.  Kept as a module-level constant so a caller
# reading the driver's contract can grep the exact names it takes, and so a
# subclass that adds a joint does not have to rewrite the map.  Names match
# ``G1JointIndex`` in ``unitree_sdk2_python`` verbatim, but the driver does
# not import the SDK's class - a name-typo in a caller's action dict must
# surface here, not from an SDK-load failure on a machine that has no SDK.
_G1_JOINT_INDEX: dict[str, int] = {
    # Left leg
    "left_hip_pitch": 0,
    "left_hip_roll": 1,
    "left_hip_yaw": 2,
    "left_knee": 3,
    "left_ankle_pitch": 4,
    "left_ankle_roll": 5,
    # Right leg
    "right_hip_pitch": 6,
    "right_hip_roll": 7,
    "right_hip_yaw": 8,
    "right_knee": 9,
    "right_ankle_pitch": 10,
    "right_ankle_roll": 11,
    # Waist
    "waist_yaw": 12,
    "waist_roll": 13,
    "waist_pitch": 14,
    # Left arm
    "left_shoulder_pitch": 15,
    "left_shoulder_roll": 16,
    "left_shoulder_yaw": 17,
    "left_elbow": 18,
    "left_wrist_roll": 19,
    "left_wrist_pitch": 20,
    "left_wrist_yaw": 21,
    # Right arm
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}

# PD gain defaults used when a caller does not supply per-joint gains.  These
# are the gains the neon reference stack uses for a rested-arm hold; they are
# deliberately conservative and match what the FSM 501 (sitting) hold expects.
# A caller who cares supplies ``kp``/``kd`` in the action dict.
_DEFAULT_KP: float = 25.0
_DEFAULT_KD: float = 0.5


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

        # Cached DDS decode results. Every one is optional per the mesh
        # contract, so a driver that has not connected yet is not broken.
        # Populated on the DDS thread; read by mesh loops. A lock guards the
        # tiny critical section around dict swap; readers snapshot.
        self._cache_lock = threading.Lock()
        self._imu: dict[str, Any] | None = None
        self._battery: dict[str, Any] | None = None
        self._lidar_state: dict[str, Any] | None = None
        self._lidar_summary: dict[str, Any] | None = None
        # ``_mode_machine`` is the uint8 hardware layout id echoed on every
        # ``LowCmd_`` (``LowState_.mode_machine``, packed ``<2B`` alongside
        # ``mode_pr`` so the value is bounded to ``[0, 255]``).  ``_fsm_id`` is
        # the high-level FSM state the arm-SDK / locomotion gates test against
        # (:data:`HANDSHAKE_FSMS` = {500, 501, 801}, :data:`WALK_FSMS`).  These
        # are two different fields with two different value ranges - conflating
        # them was the source of the ``struct.error`` on ``mode_machine=500``
        # from PR #2767 review.  ``_fsm_id`` arrives from the motion-switcher
        # API rather than ``rt/lowstate``; until that source is wired the gate
        # refuses with a precise message rather than silently rejecting every
        # real frame.
        self._mode_machine: int | None = None
        self._fsm_id: int | None = None

        # Populated by :meth:`connect_eagerly`. ``None`` on a machine that
        # never connected - a valid state for tests and imports.
        self._subs: DDSSubscriberSet | None = None
        self._pubs: DDSPublisher | None = None
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
        """Attach to the DDS bus and subscribe every sensor topic. Idempotent.

        The factory only constructs the driver; it does not connect it. Whoever
        performs the bring-up calls this (see ``examples/robots/neon.py``) so a
        real bring-up fails here rather than on the first :meth:`get_status`
        poll. The return type is a string so a caller who wants a soft failure
        (a headless smoke test that only wants the driver instance) can decide
        whether to raise.

        A second call on a connected driver is a no-op success. Rebuilding the
        subscriber set instead would re-subscribe all four topics and drop the
        only reference to the previous one, leaking its subscribers on a bus
        whose bindings segfault under concurrent construction.

        Returns:
            ``None`` on success, and on a call against an already-connected
            driver. A named reason on failure - the driver is left
            disconnected but usable, so a mesh peer for a robot that is off
            can still be constructed for later use.
        """
        if self._connected:
            logger.debug("%s already connected; connect_eagerly() is a no-op", self._tool_name)
            return None
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
        # Start the publisher on the same interface.  The subscriber set has
        # already run ``ensure_dds`` once, so :meth:`DDSPublisher.start`
        # returns ``None`` without a second SDK init - the shared lock keeps
        # both halves on the same construction lane.
        pubs = DDSPublisher(self._network_interface)
        err = pubs.start()
        if err is not None:
            # The subscriber set is up; publisher failed.  Roll back so the
            # driver reports a single connect failure instead of a half-open
            # state, and so :meth:`cleanup` on the caller's error path drops
            # the subscribers cleanly.
            subs.close()
            self._connect_error = err
            return err
        self._pubs = pubs
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
                        "mode_machine": self._mode_machine,
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
        """Release every DDS subscriber and publisher. Idempotent."""
        if self._subs is not None:
            self._subs.close()
            self._subs = None
        if self._pubs is not None:
            self._pubs.close()
            self._pubs = None
        self._connected = False

    # ------------------------------------------------------------------ #
    # Command path.                                                      #
    # ------------------------------------------------------------------ #

    def _check_motion_gates(self, scope: str) -> dict[str, Any] | None:
        """Return a refusal envelope if a motion write is not safe, else ``None``.

        Two FSM sets are enforced separately because the G1 documents them
        separately: :data:`HANDSHAKE_FSMS` covers arm-SDK writes (``rt/armsdk``)
        and :data:`WALK_FSMS` is narrower - sitting (500) accepts arm gestures
        but not walking. ``scope`` is the caller's declared kind of write:

        * ``"arm"`` - test :data:`HANDSHAKE_FSMS` and phrase the refusal as
          "refuses arm writes".
        * ``"loco"`` - test :data:`WALK_FSMS` and phrase the refusal as
          "refuses locomotion writes".
        * ``"motion"`` - test the union (either arm or loco is enough) and
          phrase the refusal generically. Used by verbs like
          :meth:`start_task` that route to either kind at runtime.

        Battery-floor is checked after the FSM because a battery-under-floor
        refusal is the same no matter which write kind was requested, and the
        caller has already been told the FSM if the FSM is the reason.
        """
        if not self._connected:
            return _refuse("not connected - call connect_eagerly() first")
        if self._mode_machine is None:
            return _refuse("mode_machine unknown - lowstate has not delivered yet")
        if self._fsm_id is None:
            # ``_fsm_id`` arrives from the motion-switcher API rather than
            # ``rt/lowstate``; ``LowState_.mode_machine`` is uint8 and cannot
            # host {500, 501, 801}.  Refuse honestly rather than let a real
            # frame silently reach a gate whose intersection with the echo's
            # value range is empty.
            return _refuse(
                "FSM id unknown - motion-switcher source has not been wired "
                "(harness#361 PR-C); see #2765 for the wire-side decision"
            )
        if scope == "arm":
            allowed, kind = HANDSHAKE_FSMS, "arm writes"
        elif scope == "loco":
            allowed, kind = WALK_FSMS, "locomotion writes"
        else:  # "motion" - accept an FSM that would satisfy either scope
            allowed = HANDSHAKE_FSMS | WALK_FSMS
            kind = "motion writes"
        if self._fsm_id not in allowed:
            return _refuse(f"FSM {self._fsm_id} refuses {kind}; needs one of {sorted(allowed)}")
        battery_pct = (self._battery or {}).get("pct")
        if battery_pct is not None and battery_pct < self._battery_floor_pct:
            return _refuse(f"battery {battery_pct:.1f}% is under floor {self._battery_floor_pct:.1f}%")
        return None

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Publish one :class:`LowCmd_` on ``rt/lowcmd`` for the given joints.

        The action dict is keyed by joint name (see :data:`_G1_JOINT_INDEX`
        for the exact set).  A caller supplies either

        * ``{joint_name: target_position_radians}`` for the common case where
          every joint uses the driver's default :data:`_DEFAULT_KP` /
          :data:`_DEFAULT_KD` gains, or
        * ``{joint_name: {"q": ..., "kp": ..., "kd": ..., "dq": ..., "tau": ...}}``
          when a caller wants per-joint control.  Any missing key inside the
          inner dict falls back to the default gain (``kp``, ``kd``) or zero
          (``dq``, ``tau``); a missing ``q`` refuses the whole action so a
          silently-zeroed target cannot make it onto the wire.

        Two things this method is deliberately *not*:

        1. A control loop.  A caller who wants 500 Hz calls this on their own
           timer; the driver's job here is one wire frame, not a schedule.
           The loop lands in the follow-up PR that closes issue #361 in full.
        2. A safety filter.  The FSM and battery gates are the safety
           envelope; command-magnitude limits are the arm-SDK client's job.

        Scope is ``"arm"`` because ``send_action`` writes to ``rt/lowcmd`` for
        arm-SDK-shaped targets; base velocity is not a ``send_action`` verb.
        The scope classification and gate call are unchanged from the previous
        stub - so the tests that already pinned FSM and battery refusals stay
        valid.
        """
        del robot_name  # driver fronts one G1
        refusal = self._check_motion_gates("arm")
        if refusal is not None:
            return refusal
        if self._pubs is None:
            return _refuse("publisher not initialised - call connect_eagerly() first")
        cmd, err = _build_lowcmd_from_action(action, mode_machine=self._mode_machine)
        if err is not None:
            return _refuse(err)
        # Lazy import.  A missing SDK on the write path is the same failure
        # mode the subscriber set already covers; publisher returns a string.
        try:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        except ImportError as exc:  # pragma: no cover - exercised on hardware
            return _refuse(f"unitree_sdk2py is not installed: {exc}")
        pub_err = self._pubs.publish(_TOPIC_LOWCMD, LowCmd_, cmd)
        if pub_err is not None:
            return _refuse(pub_err)
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "topic": _TOPIC_LOWCMD,
                        "joints": sorted(action.keys()),
                        "fsm_id": self._fsm_id,
                        "mode_machine": self._mode_machine,
                    }
                }
            ],
        }

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
        """Refuse a task today; the FSM and battery gates are already live.

        The lerobot driver runs its ``start_task`` through the same policy
        provider registry the sim uses; wiring it here needs the g1_tools
        motion verbs (issue #358) so the loop has something to command. The
        stub returns the shape that lerobot returns, so a caller polls the
        same fields either way.

        Scope is ``"motion"`` because a task may issue either an arm or a
        locomotion write and the caller does not classify itself; the union
        gate matches what the tasks are able to reach. The day motion lands
        the loop will call :meth:`_check_motion_gates` per step with the
        correct narrower scope, so an unsafe FSM refuses the *step*, not the
        whole task.
        """
        del instruction, policy_port, policy_host, policy_provider
        del duration, policy_kwargs
        refusal = self._check_motion_gates("motion")
        if refusal is not None:
            return refusal
        return _refuse("start_task not wired yet (issue #358)")

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse a rollout today; the FSM and battery gates are already live.

        A policy rollout is a task by another name (see :meth:`start_task`),
        so it shares the same motion-scoped gate. The narrower per-step
        classification lands with the write lines in issue #358.
        """
        del policy_object, instruction, duration, n_steps
        refusal = self._check_motion_gates("motion")
        if refusal is not None:
            return refusal
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
        """Decode ``rt/lowstate`` into :attr:`_imu` and :attr:`_mode_machine`.

        ``LowState_.mode_machine`` is the uint8 hardware-layout id the firmware
        wants echoed on every ``LowCmd_``.  It is **not** the high-level FSM
        state the arm-SDK gates test against - that value lives behind the
        motion-switcher API and arrives on a different topic.  Writing this
        field to :attr:`_mode_machine` (rather than :attr:`_fsm_id`) keeps the
        two ranges separate: ``[0, 255]`` for the echo, arbitrary for the gate.
        """
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
                self._mode_machine = int(mode_machine)
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
        """Decode ``rt/utlidar/lidar_state`` into :attr:`_lidar_state`.

        The names read here are the ones ``LidarState_`` declares: the MID-360
        reports its fault code as ``error_state`` and its scan rate as
        ``cloud_frequency``. Reading a name the IDL does not define is
        indistinguishable from a healthy reading in this record, because
        ``getattr``'s default is what lands in it - so a unit whose lidar had
        faulted would publish ``code=-1`` and ``freq=0.0`` for as long as it
        ran, and the fleet card would read that as "no reading yet".

        ``error_state`` is read once and used for both the numeric code and its
        rendered text so the two cannot come to describe different fields.
        """
        try:
            error_state = getattr(msg, "error_state", -1)
            self._lidar_state = {
                "code": int(error_state),
                "code_text": decode_code(error_state),
                "freq": float(getattr(msg, "cloud_frequency", 0.0)),
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

        What bounds this record is that every field is read from the message
        *header* - ``width``, ``height``, ``point_step``, ``row_step`` - so its
        size is the same for a 200-point frame and a 30k-point one. No point is
        enumerated here, and nothing is downsampled, so there is no point count
        for a cap to apply to. ``count`` is therefore the cloud's true size: a
        MID-360 that drops from 24000 points to 3000 is reporting a fault, and
        clamping the number would hide exactly that.
        """
        try:
            width = int(getattr(msg, "width", 0))
            height = int(getattr(msg, "height", 0))
            count = width * height
            self._lidar_summary = {
                "count": count,
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
            value: dict[str, Any] | None = getattr(self, attr, None)
            if value is None:
                return None
            return dict(value)


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


def _build_lowcmd_from_action(
    action: dict[str, Any],
    mode_machine: int | None = None,
) -> tuple[Any, str | None]:
    """Build a ``LowCmd_`` populated from a caller's ``send_action`` dict.

    Returns ``(cmd, None)`` on success and ``(None, reason)`` when the action
    dict is unusable.  Kept as a free function so a test can walk the mapping
    without a driver instance, and so :meth:`G1Driver.send_action` reads as
    "gate, build, publish" - three verbs in three lines.

    The mapping is:

    * Every joint name in ``action`` must be a key of :data:`_G1_JOINT_INDEX`.
      An unknown name refuses the whole action - the alternative would be to
      silently drop a joint the caller thought was commanded, which is the
      single worst failure mode on a robot.
    * A scalar value is interpreted as the position target ``q``, with
      :data:`_DEFAULT_KP` / :data:`_DEFAULT_KD` gains and zero ``dq``/``tau``.
    * A dict value must contain ``"q"``; ``"kp"``, ``"kd"``, ``"dq"``, ``"tau"``
      are optional.  An unknown key inside the inner dict is refused for the
      same reason as an unknown joint name: silent drop is worse than a
      caller-facing error.

    Wire-frame contract (issue #361 review, [MUST FIX] from yinsong1986):

    * ``mode_pr = 0`` - PR mode, which is what the joint-name table this
      helper interprets is calibrated for.  AB mode (``mode_pr = 1``) would
      silently remap four ankle indices.
    * ``mode_machine`` is echoed from the live ``LowState`` (the driver
      caches it at :attr:`G1Driver._mode_machine`, a uint8 in ``[0, 255]``
      as packed by ``_CRC__packFmtHGLowCmd``).  Firmware drops a frame
      whose ``mode_machine`` does not match.  This is **not** the same
      value as :attr:`G1Driver._fsm_id`, which comes from the
      motion-switcher API and is the arm-SDK gate's admission value.
    * ``motor_cmd[i].mode = 1`` on every commanded slot - the enable byte.
      Unset (``0`` = Disable), a frame with a valid CRC still commands
      nothing.  Slots the caller did not touch stay at ``0``.
    * ``crc`` is computed by the SDK's own ``CRC().Crc(cmd)`` after every
      other field is populated.  Firmware silently drops a non-matching
      frame, so this is the last write before return.

    The SDK import is lazy so this helper is safe to call in a test without
    the SDK on the box; a missing SDK returns a reason string, matching the
    driver's other error paths.
    """
    if not isinstance(action, dict):
        return None, f"action must be a dict, got {type(action).__name__}"
    if not action:
        return None, "action is empty; nothing to command"
    try:
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as _default_lowcmd
        from unitree_sdk2py.utils.crc import CRC as _CRC
    except ImportError as exc:  # pragma: no cover - exercised on hardware
        return None, f"unitree_sdk2py is not installed: {exc}"
    cmd = _default_lowcmd()
    # Wire-frame contract: PR mode, echo mode_machine, enable the touched slots.
    cmd.mode_pr = 0
    if mode_machine is not None:
        cmd.mode_machine = int(mode_machine)
    known_inner = {"q", "kp", "kd", "dq", "tau"}
    for name, value in action.items():
        slot = _G1_JOINT_INDEX.get(name)
        if slot is None:
            allowed = ", ".join(sorted(_G1_JOINT_INDEX))
            return None, f"unknown joint name {name!r}; expected one of: {allowed}"
        if isinstance(value, dict):
            unknown_inner = set(value) - known_inner
            if unknown_inner:
                return None, (
                    f"unknown per-joint keys for {name!r}: "
                    f"{sorted(unknown_inner)}; expected a subset of {sorted(known_inner)}"
                )
            if "q" not in value:
                return None, f"per-joint dict for {name!r} is missing required key 'q'"
            q = value["q"]
            kp = value.get("kp", _DEFAULT_KP)
            kd = value.get("kd", _DEFAULT_KD)
            dq = value.get("dq", 0.0)
            tau = value.get("tau", 0.0)
        else:
            q, kp, kd, dq, tau = value, _DEFAULT_KP, _DEFAULT_KD, 0.0, 0.0
        try:
            q_f = float(q)
            kp_f = float(kp)
            kd_f = float(kd)
            dq_f = float(dq)
            tau_f = float(tau)
        except (TypeError, ValueError) as exc:
            return None, f"joint {name!r} carries a non-numeric target: {exc}"
        motor = cmd.motor_cmd[slot]
        motor.mode = 1  # Enable - a Disable slot commands nothing regardless of CRC.
        motor.q = q_f
        motor.dq = dq_f
        motor.tau = tau_f
        motor.kp = kp_f
        motor.kd = kd_f
    # CRC last - firmware drops a non-matching frame silently.
    cmd.crc = _CRC().Crc(cmd)
    return cmd, None


def _build_zero_torque_lowcmd(
    mode_machine: int | None = None,
) -> tuple[Any, str | None]:
    """Return a ``LowCmd_`` with every motor's gains and effort zeroed.

    A zero-kp/kd/tau motor holds no position and applies no torque - the
    softest wire frame the SDK protocol accepts.  Used by :meth:`G1Driver.stop`
    (issue #361 follow-up: the control loop uses the same helper on shutdown).

    Kept as a free function so a test can compare the produced envelope
    slot-by-slot without a driver instance, and so the ``stop`` and control
    loop paths share exactly one construction site.

    Wire-frame contract (parity with :func:`_build_lowcmd_from_action`):

    * ``mode_pr = 0`` - PR mode.  Firmware validates the same field on the
      stop frame as on any other; keep the value consistent.
    * ``mode_machine`` is echoed from the caller (typically the driver's
      cached :attr:`G1Driver._mode_machine`, uint8).  Firmware drops a stop
      frame whose ``mode_machine`` does not match, and a dropped stop is a
      fall.
    * ``motor_cmd[i].mode = 1`` (Enable) on every slot.  A Disable slot
      with zero gains lets the joint fall freely - the arm-SDK protocol
      treats Disable as "not controlled at all" regardless of gain.
      Enable + zero gains is the softest *controlled* state the protocol
      expresses; that is what a stop wants.
    * ``crc`` is stamped last so a later populate cannot invalidate it.
    """
    try:
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as _default_lowcmd
        from unitree_sdk2py.utils.crc import CRC as _CRC
    except ImportError as exc:  # pragma: no cover - exercised on hardware
        return None, f"unitree_sdk2py is not installed: {exc}"
    cmd = _default_lowcmd()
    # Wire-frame contract: PR mode, echo mode_machine, Enable every slot.
    cmd.mode_pr = 0
    if mode_machine is not None:
        cmd.mode_machine = int(mode_machine)
    for motor in cmd.motor_cmd:
        motor.mode = 1  # Enable - a Disable slot lets the joint fall freely.
        motor.q = 0.0
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
    # CRC last - firmware drops a non-matching frame silently.
    cmd.crc = _CRC().Crc(cmd)
    return cmd, None

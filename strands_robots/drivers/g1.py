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
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, cast

from strands_robots.mesh.pacing import Ticker
from strands_robots.tools.g1 import HANDSHAKE_FSMS, WALK_FSMS, decode_code
from strands_robots.tools.g1._dds_engine import DDSPublisher, DDSSubscriberSet
from strands_robots.utils import (
    finite_number_error,
    positive_count_error,
    positive_finite_number_error,
)

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants surfaced for tests and for issue #358 to share.
# ---------------------------------------------------------------------------
_BATTERY_FLOOR_PCT: float = 15.0

# Control-loop cadence.  500 Hz matches the SDK's own G1 low-level example
# (``example/g1/high_level/g1_ankle_swing_example.py`` sleeps 0.002 s between
# writes).  Firmware validates that consecutive frames arrive on-cadence to
# hold the last commanded posture; a slower loop lets the joints droop
# between frames.  Kept as a module constant so a test can override it
# without patching the sleep call directly.
_CONTROL_LOOP_HZ: float = 500.0
_CONTROL_LOOP_DT: float = 1.0 / _CONTROL_LOOP_HZ

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

# The named-joint count, derived from :data:`_G1_JOINT_INDEX` so a joint added
# later moves both builders together.  ``LowCmd_.motor_cmd`` is a 35-array; the
# G1 commands 29 joints and slots 29..34 are a reserved tail that neither
# builder names.  Bound the Enable-byte loop by this count rather than by
# ``len(cmd.motor_cmd)`` so the stop frame stays byte-identical to the SDK's
# own G1 reference (``example/g1/low_level/g1_low_level_example.py:135`` loops
# ``for i in range(G1_NUM_MOTOR)``, where ``G1_NUM_MOTOR = 29``).
_G1_NAMED_JOINTS: int = max(_G1_JOINT_INDEX.values()) + 1

# Per-joint PD gains applied when a caller does not supply ``kp``/``kd``,
# indexed by the same slot as :data:`_G1_JOINT_INDEX`.  Transcribed from the
# vendor's own ``rt/lowcmd`` reference for this robot -- the module-scope
# ``Kp``/``Kd`` lists in
# ``unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py`` -- which
# is the gain set the firmware's low-level position mode is tuned against.
#
# These cannot collapse to a single scalar: ``Kp`` takes three distinct values
# and ``Kd`` two, because the joints do not carry comparable loads.  Both knees
# (slots 3 and 9) are the stiffest entries at ``kp=100, kd=2`` and they are the
# joints that hold a standing biped up, so a scalar chosen anywhere in the
# table's range understiffens something: an arm-sized ``kp=40`` leaves each knee
# at 40% of its reference stiffness.  The firmware treats gains as advisory --
# it validates ``crc``, ``mode_machine`` and the Enable byte, not ``kp``/``kd``
# -- so a mis-gained frame is accepted and the publish reports success.  The
# closed-loop stiffness of the joint is then whatever was sent, and there is no
# status for a caller to read that says otherwise.
#
# A caller who wants different gains supplies ``kp``/``kd`` per joint in the
# action dict; a supplied value always wins over the table.
# The vendor groups its lists as two legs, a waist and two arms; naming the
# groups keeps that structure legible (the two legs are identical, as are the two
# arms) and makes the 29-entry width fall out of 6 + 6 + 3 + 7 + 7 rather than
# being asserted separately.
_LEG_KP: tuple[float, ...] = (60.0, 60.0, 60.0, 100.0, 40.0, 40.0)  # hip p/r/y, knee, ankle p/r
_WAIST_KP: tuple[float, ...] = (60.0, 40.0, 40.0)  # yaw, roll, pitch
_ARM_KP: tuple[float, ...] = (40.0,) * 7  # shoulder p/r/y, elbow, wrist r/p/y
_SDK_KP: tuple[float, ...] = _LEG_KP + _LEG_KP + _WAIST_KP + _ARM_KP + _ARM_KP

_LEG_KD: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 1.0)
_WAIST_KD: tuple[float, ...] = (1.0, 1.0, 1.0)
_ARM_KD: tuple[float, ...] = (1.0,) * 7
_SDK_KD: tuple[float, ...] = _LEG_KD + _LEG_KD + _WAIST_KD + _ARM_KD + _ARM_KD


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

        Raises:
            ValueError: If ``battery_floor_pct`` is not a finite number.
        """
        del cameras, data_config  # accepted for parity; unused here
        if kwargs:
            logger.debug("G1Driver ignoring extra kwargs: %s", sorted(kwargs))
        self._tool_name = tool_name
        self._port = port
        self._network_interface = network_interface
        # The floor is the only constructor value the safety gate compares a
        # reading against, so it is held to the same shared domain
        # ``run_policy`` holds ``duration`` to.  A bare ``float()`` accepted
        # ``nan``: every ``battery_pct < nan`` is False, so the driver stored a
        # floor it reported in :meth:`get_status` and enforced nowhere, and the
        # gate opened on a critically low pack.  ``finite_number_error`` also
        # rejects ``bool`` (``True`` would act as a silent 1.0%) and a numeric
        # string, which is how a config file spells ``nan``.
        if reason := finite_number_error(battery_floor_pct, "battery_floor_pct", "G1Driver"):
            raise ValueError(reason)
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
        # them raises ``struct.error`` on ``mode_machine=500`` because the CRC
        # layout packs ``mode_machine`` as ``<B``.  ``_fsm_id`` arrives from
        # the motion-switcher API rather than ``rt/lowstate``; until that source is wired the gate
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

        # ``_loop`` holds at most one :class:`_ControlLoop` at a time.  It is
        # ``None`` before :meth:`run_policy` is called and after the loop
        # thread joins (the loop clears the reference on exit under
        # ``_task_admission``).  ``run_policy`` refuses when the current
        # reference is running so a second concurrent rollout cannot silently
        # share the wire with the first.
        self._loop: _ControlLoop | None = None
        # ``_last_task_snapshot`` retains the loop's final snapshot after the
        # thread joins.  ``_run``'s ``finally`` clears ``_loop`` so a second
        # rollout starts fresh, but that clear made every self-terminating
        # exit reason (``n_steps``, ``duration``, ``gate``, ``policy``,
        # ``publish``) unobservable through :meth:`get_task_status`: five of
        # the six documented exit reasons round-tripped to the caller as
        # "no task has been started on this driver".  Stashing the snapshot
        # right before the clear keeps the vocabulary the loop names
        # reachable to a poller that missed the running window.
        self._last_task_snapshot: dict[str, Any] | None = None
        # ``_task_admission`` is held across every check-then-act sequence on
        # ``_loop`` (admission in :meth:`run_policy`, stop in
        # :meth:`stop_task` / :meth:`cleanup` / :meth:`stop`, clear on loop
        # exit).  Without it, two threads calling ``run_policy`` at once could
        # both pass the ``is_running`` check before either assigns
        # ``self._loop``, and an e-stop landing between the check and the
        # start would count this peer as stopped while the rollout starts a
        # moment later.  Mirrors ``HardwareRobot._task_admission`` -
        # single-command-bus invariant, one lock across the whole sequence.
        self._task_admission = threading.Lock()

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
        """Stop a running control loop and release further writes.

        The mesh calls this during shutdown.  If a rollout is running,
        signal the loop to exit, publish the zero-torque frame, and join
        its thread before returning - a controlled stop rather than
        abrupt frame cessation mid-policy on a robot whose FSM has just
        gone away.  Idempotent: no running task returns immediately.

        A loop that outlasts the join budget is reported rather than
        returned: this signature carries no envelope, and returning from
        shutdown while a thread still holds the wire is exactly what the
        log exists to say.  Unlike :meth:`cleanup` this leaves the
        publisher open either way, so that loop still publishes its
        zero-torque frame when the policy finally returns.
        """
        with self._task_admission:
            loop = self._loop
        if loop is not None and loop.is_running:
            logger.debug("%s.stop() halting control loop", self._tool_name)
            if not loop.stop("stop_task"):
                logger.error(
                    "%s.stop(): control loop did not join within the stop budget "
                    "and still holds the wire; its zero-torque frame publishes "
                    "when the policy returns",
                    self._tool_name,
                )

    def cleanup(self) -> None:
        """Release every DDS subscriber and publisher. Idempotent.

        Halts a running control loop first so the zero-torque shutdown
        frame goes out on a publisher that still exists.  Closing
        ``_pubs`` under a live 500 Hz thread would drop the loop into
        its ``publish`` branch and skip the zero-torque frame - the fall
        this whole path exists to prevent.

        Halting is not the same as having halted.
        :meth:`_ControlLoop.stop` reports whether the thread actually
        joined, and a caller-supplied policy that outlasts the join
        budget - a remote inference call is the ordinary case - leaves it
        running.  The publisher is therefore released only once the loop
        is provably gone: a loop that is still running keeps it, so the
        zero-torque frame that loop publishes from its own ``finally``
        reaches the wire instead of being dropped by
        :meth:`_ControlLoop._emit_zero_torque`'s ``pubs is None`` return.
        The subscribers close either way - the loop never reads them - and
        a second ``cleanup()`` once the loop has exited releases the rest.
        """
        with self._task_admission:
            loop = self._loop
        joined = True
        if loop is not None and loop.is_running:
            joined = loop.stop("stop_task")
        if self._subs is not None:
            self._subs.close()
            self._subs = None
        if joined:
            if self._pubs is not None:
                self._pubs.close()
                self._pubs = None
        else:
            logger.error(
                "%s.cleanup(): control loop did not join within the stop budget, "
                "so the publisher stays open for its zero-torque frame; call "
                "cleanup() again once the loop has exited",
                self._tool_name,
            )
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
          the joint takes its reference gains from :data:`_SDK_KP` /
          :data:`_SDK_KD`, or
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

    # ------------------------------------------------------------------ #
    # Task and policy paths.  The 500 Hz control loop lands here.        #
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
        """Start a policy-driven task on the control loop.

        The lerobot driver runs its ``start_task`` through a policy provider
        registry.  Providers live in :mod:`strands_robots.policies`; wiring a
        concrete inference client (Groot, ACT, Diffusion) here needs the
        ``g1_tools`` motion verbs (issue #358) so the loop has something to
        command with joint-name semantics.

        This driver's role for harness#361 PR-C is the **transport primitive**:
        a background thread that spins at 500 Hz, gates every step against the
        FSM and battery, publishes ``LowCmd_`` frames on ``rt/lowcmd``, and on
        stop or budget expiry publishes a zero-torque frame before exiting.
        A caller who supplies a callable-style policy via
        :meth:`run_policy` gets the whole loop today; :meth:`start_task` still
        refuses with a message naming issue #358 because the provider
        registry is not yet plumbed here (that decision moves with #358 to
        keep the two concerns separable).

        Scope is ``"motion"`` because a task may issue either an arm or a
        locomotion write and the caller does not classify itself; the union
        gate matches what the tasks are able to reach.  The per-step
        re-check inside the loop is scoped ``"motion"`` for the same reason.
        """
        del instruction, policy_port, policy_host, policy_provider
        del duration, policy_kwargs
        refusal = self._check_motion_gates("motion")
        if refusal is not None:
            return refusal
        return _refuse(
            "start_task: provider registry not wired yet (issue #358); "
            "use run_policy(policy_object=...) to drive the control loop today"
        )

    def run_policy(
        self,
        policy_object: Policy | Callable[[Any], dict[str, Any]],
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Roll out an already-built policy on the 500 Hz control loop.

        The loop runs on a dedicated thread so the caller returns
        immediately; poll :meth:`get_task_status` to observe progress.
        Every step re-gates through :meth:`_check_motion_gates` with
        scope ``"motion"``; a gate flip refuses the step and the loop
        publishes a zero-torque frame before exiting rather than freezing
        with the last commanded posture on a robot whose FSM just left the
        allowed set.

        ``policy_object`` is either a built :class:`~strands_robots.policies.Policy`
        or a bare callable - the admission check below accepts a ``.step()``
        attribute *or* a callable object, so the annotation admits the same
        set the refusal enforces.  It is called on each step with a snapshot
        of the cached observations (``mode_machine``, ``fsm_id``, ``imu``, etc.)
        and is expected to return a joint-name-keyed action dict of the
        same shape :meth:`send_action` accepts.  A policy that returns
        ``None`` or an unusable action is refused inside the loop; the
        refusal name and count surface through :meth:`get_task_status`.
        """
        del instruction  # policies own their own conditioning
        # Validate the continuous knobs on the shared domains before the
        # gate.  ``duration`` reaches ``deadline = started_at + duration``
        # inside the loop; ``nan`` poisons every comparison there so the
        # 500 Hz loop would actuate with no budget, ``inf`` collapses the
        # exit test to always-false, and a non-numeric string raises out of
        # a method that must return an envelope.  ``n_steps`` reaches
        # ``self._steps >= self._n_steps`` as ``range()``-shaped index; a
        # ``bool`` silently caps at 1, a fractional applies a cap the caller
        # never named, and ``0`` / negative exits instantly with
        # ``exit_reason="n_steps"`` in a success envelope for a rollout that
        # commanded nothing (see HardwareRobot._n_steps_error for the
        # incident that made positive_count_error load-bearing).
        if err := positive_finite_number_error(duration, "duration", "run_policy"):
            return _refuse(err)
        if n_steps is not None and (err := positive_count_error(n_steps, "n_steps", "run_policy")):
            return _refuse(err)
        refusal = self._check_motion_gates("motion")
        if refusal is not None:
            return refusal
        if policy_object is None:
            return _refuse("run_policy: policy_object is required")
        step_fn = getattr(policy_object, "step", None)
        if not callable(step_fn) and not callable(policy_object):
            return _refuse("run_policy: policy_object must be callable or expose a .step() method")
        # Admission held across the ``is_running`` check, the reference
        # assignment and ``start()`` so a second thread cannot pass the check
        # before either assigns ``self._loop`` (two rollouts on one wire),
        # and an e-stop landing between the check and the start cannot count
        # this peer as stopped while the rollout starts a moment later.
        loop = _ControlLoop(
            driver=self,
            policy=policy_object,
            duration=float(duration),
            n_steps=n_steps,
        )
        with self._task_admission:
            if self._loop is not None and self._loop.is_running:
                return _refuse("run_policy: a task is already running; call stop_task first")
            # Clear any stashed terminal snapshot from a previous rollout so
            # a poller between ``run_policy`` and the first published frame
            # sees the new loop's snapshot rather than the last one's exit
            # reason.
            self._last_task_snapshot = None
            self._loop = loop
            loop.start()
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "task_running": True,
                        "duration": float(duration),
                        "n_steps": n_steps,
                        "hz": _CONTROL_LOOP_HZ,
                    }
                }
            ],
        }

    def get_task_status(self) -> dict[str, Any]:
        """Report the running task's state.

        Returns a JSON envelope with ``running``, ``steps``, ``refusals``
        and (when finished) ``exit_reason``.  Safe to poll from any thread
        because the loop writes its snapshot under a lock.

        A poller that missed the running window still sees the loop's final
        snapshot: :attr:`_last_task_snapshot` is stashed under the admission
        lock right before the loop's ``finally`` clears ``self._loop``, so
        every self-terminating exit reason (``n_steps``, ``duration``,
        ``gate``, ``policy``, ``publish``) round-trips to the caller instead
        of collapsing to "no task has been started" once the thread joins.
        """
        with self._task_admission:
            loop = self._loop
            last = self._last_task_snapshot
        if loop is None:
            if last is not None:
                return {"status": "success", "content": [{"json": last}]}
            return {
                "status": "success",
                "content": [
                    {
                        "json": {
                            "running": False,
                            "reason": "no task has been started on this driver",
                        }
                    }
                ],
            }
        return {"status": "success", "content": [{"json": loop.snapshot()}]}

    def stop_task(self) -> dict[str, Any]:
        """Stop the running task and publish a zero-torque frame.

        Idempotent: no running task returns a success envelope naming the
        state.  A running task signals the loop to exit, joins its thread,
        and the loop publishes :func:`_build_zero_torque_lowcmd` on the way
        out - a soft *controlled* stop rather than a Disable that would let
        the named joints fall freely.

        The returned envelope reports the join outcome honestly.  A
        caller-supplied policy that outlasts the join budget (a remote
        inference call is the ordinary case) surfaces as
        ``status="error"`` naming the timeout and ``stopped=False`` in the
        payload, so the caller cannot read "success" while the payload's
        own ``running=True`` says the loop is still writing frames.
        """
        with self._task_admission:
            loop = self._loop
        if loop is None or not loop.is_running:
            return {
                "status": "success",
                "content": [{"text": "stop_task: no task is running"}],
            }
        joined = loop.stop("stop_task")
        snap = loop.snapshot()
        snap["stopped"] = joined
        if not joined:
            # The loop still holds the wire.  Report as an error so a
            # caller that reads only ``status`` cannot count the task as
            # stopped, and name the timeout in the payload for a caller
            # that reads it.
            return {
                "status": "error",
                "content": [
                    {
                        "json": {
                            **snap,
                            "reason": (
                                "stop_task: control loop did not join within timeout; "
                                "policy is likely blocking - the loop will publish the "
                                "zero-torque frame when it exits"
                            ),
                        }
                    }
                ],
            }
        return {
            "status": "success",
            "content": [{"json": snap}],
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
      the slot's :data:`_SDK_KP` / :data:`_SDK_KD` gains and zero ``dq``/``tau``.
    * A dict value must contain ``"q"``; ``"kp"``, ``"kd"``, ``"dq"``, ``"tau"``
      are optional.  An unknown key inside the inner dict is refused for the
      same reason as an unknown joint name: silent drop is worse than a
      caller-facing error.

    Wire-frame contract:

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
            kp = value.get("kp", _SDK_KP[slot])
            kd = value.get("kd", _SDK_KD[slot])
            dq = value.get("dq", 0.0)
            tau = value.get("tau", 0.0)
        else:
            q, kp, kd, dq, tau = value, _SDK_KP[slot], _SDK_KD[slot], 0.0, 0.0
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
    * ``motor_cmd[i].mode = 1`` (Enable) on every **named** slot (0..28).
      Slots 29..34 are a reserved tail: no name in :data:`_G1_JOINT_INDEX`
      maps to them, and the SDK's own G1 reference bounds its Enable loop by
      ``G1_NUM_MOTOR = 29`` rather than by the array width.  Enabling a
      reserved slot at zero gains would be a decision this driver does not
      have the information to make.  A Disable slot with zero gains lets a
      *named* joint fall freely - the arm-SDK protocol treats Disable as "not
      controlled at all" regardless of gain.  Enable + zero gains is the
      softest *controlled* state the protocol expresses on the joints this
      driver names; that is what a stop wants.
    * ``crc`` is stamped last so a later populate cannot invalidate it.

    ``mode_machine`` is accepted as an ``int | None`` for signature parity
    with :func:`_build_lowcmd_from_action`, and the caller is expected to
    pass the driver's cached uint8 (``[0, 255]``) rather than an FSM id.
    Values outside that range raise ``struct.error`` inside the SDK CRC
    packer as a property of the SDK's own ``<2B2x`` pack format, not of this
    helper; no production path in this driver can reach the raise because
    ``G1Driver._on_lowstate`` binds ``_mode_machine`` from a uint8 IDL field.

    This helper is defined but not yet wired: ``G1Driver.stop`` and
    ``stop_task`` currently return refusal envelopes rather than publishing
    a frame, and no other call site exists.  The 500 Hz control-loop PR
    (harness#361 PR-C) is where the wiring lands; the helper is defined
    here so the loop's shutdown path composes on a tested, CRC-correct
    frame rather than one hand-rolled next to it.
    """
    try:
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as _default_lowcmd
        from unitree_sdk2py.utils.crc import CRC as _CRC
    except ImportError as exc:  # pragma: no cover - exercised on hardware
        return None, f"unitree_sdk2py is not installed: {exc}"
    cmd = _default_lowcmd()
    # Wire-frame contract: PR mode, echo mode_machine, Enable every named slot.
    cmd.mode_pr = 0
    if mode_machine is not None:
        cmd.mode_machine = int(mode_machine)
    # ``motor_cmd`` is a 35-array; the G1 commands 29 joints, and slots
    # 29..34 are a reserved tail that no name in ``_G1_JOINT_INDEX`` maps to.
    # Assert the width so a future SDK change to a shorter array fails here
    # rather than by silent slice - this is the length check CodeQL 980
    # asked for, now that ``_G1_NAMED_JOINTS`` documents the distinction.
    assert len(cmd.motor_cmd) >= _G1_NAMED_JOINTS, (
        f"LowCmd_.motor_cmd is {len(cmd.motor_cmd)} slots; the driver names "
        f"{_G1_NAMED_JOINTS} joints and cannot address them all"
    )
    for i in range(_G1_NAMED_JOINTS):
        motor = cmd.motor_cmd[i]
        motor.mode = 1  # Enable - a Disable slot lets the named joint fall freely.
        motor.q = 0.0
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
    # Slots [_G1_NAMED_JOINTS, len(cmd.motor_cmd)) stay at SDK defaults
    # (mode=0, q=0, dq=0, tau=0, kp=0, kd=0) - byte-identical to the SDK
    # reference's zero-posture frame on those slots.
    # CRC last - firmware drops a non-matching frame silently.
    cmd.crc = _CRC().Crc(cmd)
    return cmd, None


class _ControlLoop:
    """Background 500 Hz control loop that a :class:`G1Driver` owns.

    Composition rather than inheritance: the driver holds one, ``run_policy``
    hands it a policy object, and it owns the thread, the FSM re-gate loop,
    and the zero-torque frame it emits on exit.  Kept in this module because
    it reaches into the driver's cached observations, the pure builders and
    the DDS publisher - three seams a sibling module would have to re-export.

    Exit reasons
    ------------
    Every terminal path names itself so :meth:`_ControlLoop.snapshot`
    reports why the
    rollout stopped:

    * ``"stop_task"`` - caller invoked :meth:`G1Driver.stop_task`.
    * ``"n_steps"`` - the step budget was met.
    * ``"duration"`` - the wall-clock budget was met.
    * ``"gate"`` - a per-step re-gate refused; the refusal reason is stored.
    * ``"policy"`` - the policy raised or returned an unusable action; the
      stored reason names which.
    * ``"publish"`` - the underlying DDS publish returned a reason string.

    A zero-torque frame is published on **every** terminal path except
    ``"publish"`` (where the wire is already reason-carrying and a second
    stamp would clobber the reason with a fresh error).  A soft controlled
    stop is what the biped wants; a Disable slot lets the joint fall.
    """

    def __init__(
        self,
        driver: G1Driver,
        policy: Any,
        duration: float,
        n_steps: int | None,
    ) -> None:
        self._driver = driver
        self._policy = policy
        self._duration = float(duration)
        self._n_steps = n_steps
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        # Snapshot fields, all under ``_lock``.
        self._steps: int = 0
        self._refusals: int = 0
        self._exit_reason: str | None = None
        self._exit_detail: str | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None

    @property
    def is_running(self) -> bool:
        """Whether the loop thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Spawn the loop thread.  Idempotent second calls are a bug."""
        if self._thread is not None:
            raise RuntimeError("_ControlLoop.start() called twice")
        self._started_at = time.monotonic()
        # ``daemon=True`` so a caller who forgets ``stop_task`` at process
        # exit does not hang the interpreter.  The zero-torque shutdown
        # runs at every normal exit path because :meth:`G1Driver.stop` and
        # :meth:`G1Driver.cleanup` join the loop before closing ``_pubs``.
        self._thread = threading.Thread(target=self._run, name=f"g1-control-{id(self):x}", daemon=True)
        self._thread.start()

    def stop(self, reason: str = "stop_task", timeout: float = 2.0) -> bool:
        """Signal the loop to exit and join its thread.

        Named reasons distinguish caller-driven stop (``"stop_task"``) from
        the loop's own terminal paths, which set ``_exit_reason`` directly.
        The signal wins over policy work: the loop re-reads
        ``_stop_event.is_set()`` at the top of every step, and once more
        after the policy returns and before the frame publishes.

        Returns:
            ``True`` when the thread joined within ``timeout``.  ``False``
            when the loop is still running - a caller-supplied policy that
            outlasts the join budget (a remote inference call is the
            ordinary case) needs the honest ``stopped=False`` in the
            :meth:`stop_task` envelope rather than a ``success`` claim the
            payload's ``running=True`` contradicts.
        """
        self._stop_event.set()
        thread = self._thread
        joined = True
        if thread is not None:
            thread.join(timeout=timeout)
            joined = not thread.is_alive()
        with self._lock:
            # The loop itself may have set an exit_reason (budget expiry
            # racing the caller's stop); if not, the caller wins.
            if self._exit_reason is None:
                self._exit_reason = reason
        return joined

    def snapshot(self) -> dict[str, Any]:
        """Return the loop's public state.  Safe from any thread."""
        with self._lock:
            elapsed: float | None
            if self._started_at is None:
                elapsed = None
            elif self._finished_at is None:
                elapsed = time.monotonic() - self._started_at
            else:
                elapsed = self._finished_at - self._started_at
            return {
                "running": self.is_running,
                "steps": self._steps,
                "refusals": self._refusals,
                "elapsed_s": elapsed,
                "duration_budget_s": self._duration,
                "n_steps_budget": self._n_steps,
                "exit_reason": self._exit_reason,
                "exit_detail": self._exit_detail,
                "hz": _CONTROL_LOOP_HZ,
            }

    # ------------------------------------------------------------------ #
    # Thread body.  Every branch names an exit reason before it returns. #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        assert self._started_at is not None
        deadline = self._started_at + self._duration
        publish_reason: str | None = None
        try:
            with Ticker(_CONTROL_LOOP_DT, self._stop_event) as ticker:
                while not self._stop_event.is_set():
                    now = time.monotonic()
                    if self._n_steps is not None and self._steps >= self._n_steps:
                        self._set_exit("n_steps", None)
                        break
                    if now >= deadline:
                        self._set_exit("duration", None)
                        break
                    # Per-step re-gate.  A gate flip refuses the *step* rather
                    # than the whole task, so an FSM transition out of the
                    # allowed set exits cleanly with a zero-torque frame.
                    refusal = self._driver._check_motion_gates("motion")
                    if refusal is not None:
                        detail = _refusal_text(refusal)
                        self._set_exit("gate", detail)
                        break
                    try:
                        action = self._call_policy()
                    except Exception as exc:  # policy is caller-supplied; catch broadly
                        self._set_exit("policy", f"raised {type(exc).__name__}: {exc}")
                        break
                    # Re-check the stop event *after* the policy returns and
                    # *before* the frame publishes.  Without this, a stop
                    # signal that arrives while the policy is still computing
                    # (the ordinary case for a remote inference call) is only
                    # noticed at the top of the next iteration - but by then
                    # the in-flight action has already reached the wire, so a
                    # fresh position command lands on ``rt/lowcmd`` after the
                    # caller was told the task stopped.  The zero-torque frame
                    # still goes out from ``finally``; the loop reports the
                    # exit as ``stop_task`` since the caller's ``stop()``
                    # already set the reason.
                    if self._stop_event.is_set():
                        break
                    if action is None:
                        with self._lock:
                            self._refusals += 1
                        self._set_exit("policy", "policy returned None")
                        break
                    cmd, err = _build_lowcmd_from_action(action, mode_machine=self._driver._mode_machine)
                    if err is not None:
                        with self._lock:
                            self._refusals += 1
                        self._set_exit("policy", f"policy returned an unusable action: {err}")
                        break
                    pubs = self._driver._pubs
                    if pubs is None:
                        self._set_exit("publish", "driver has no publisher; not connected")
                        publish_reason = "no publisher"
                        break
                    # The SDK's LowCmd_ class is the wire-format handshake with
                    # the DDS publisher: it identifies the topic type registered
                    # on the participant.  ``_build_lowcmd_from_action`` already
                    # returned an SDK-shaped ``cmd`` (or an err path we took
                    # above), so importing the class here cannot introduce a
                    # new failure mode - the same import already succeeded in
                    # the builder.  Tests stub ``unitree_sdk2py`` via
                    # ``monkeypatch.setitem(sys.modules, ...)`` so this same
                    # production lane runs on an SDK-less CI box.
                    try:
                        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
                    except ImportError as exc:  # pragma: no cover - hardware-only
                        self._set_exit("publish", f"unitree_sdk2py is not installed: {exc}")
                        publish_reason = "sdk missing"
                        break
                    pub_err = pubs.publish(_TOPIC_LOWCMD, LowCmd_, cmd)
                    if pub_err is not None:
                        self._set_exit("publish", str(pub_err))
                        publish_reason = str(pub_err)
                        break
                    with self._lock:
                        self._steps += 1
                    # Pace on a deadline, not a delay: the time spent in the
                    # step above is subtracted from the next period, so a
                    # policy that takes a few ms per step still holds 500Hz
                    # instead of dropping to work+2ms. ``Ticker.wait`` returns
                    # ``True`` promptly when the stop event fires, so the
                    # interruptible-stop property of the previous
                    # ``self._stop_event.wait(_CONTROL_LOOP_DT)`` is kept.
                    if ticker.wait():
                        break
        finally:
            # Publish a zero-torque frame on every exit path except the one
            # where the wire itself just refused - clobbering that reason
            # with a fresh publish error is worse than not stamping a stop
            # frame the wire cannot carry anyway.
            if publish_reason is None:
                self._emit_zero_torque()
            with self._lock:
                self._finished_at = time.monotonic()
            # Stash the terminal snapshot before dropping the reference so a
            # poller that missed the running window still sees the named
            # exit reason (n_steps / duration / gate / policy / publish);
            # without this stash, ``get_task_status`` collapsed five of the
            # six documented exit paths to "no task has been started on
            # this driver".  Held under the same admission lock as the
            # clear so a concurrent ``run_policy`` observes a coherent
            # (running loop | last snapshot) pair rather than a torn state.
            # ``snapshot()``'s ``running`` reads ``self._thread.is_alive()``
            # and this ``finally`` runs *inside* the thread, so it would
            # report ``True`` on a snapshot the caller reads *after* the
            # thread joins.  Override with a literal ``False`` - the loop
            # is terminating; the caller polling ``_last_task_snapshot`` has
            # already left the running window by construction.
            terminal = self.snapshot()
            terminal["running"] = False
            with self._driver._task_admission:
                self._driver._last_task_snapshot = terminal
                if self._driver._loop is self:
                    self._driver._loop = None

    def _call_policy(self) -> Any:
        """Invoke the policy with a snapshot of cached observations."""
        obs = {
            "mode_machine": self._driver._mode_machine,
            "fsm_id": self._driver._fsm_id,
            "battery": self._driver._battery,
            "imu": self._driver._imu,
        }
        step = getattr(self._policy, "step", None)
        if callable(step):
            return step(obs)
        return self._policy(obs)  # pragma: no cover - covered by direct-callable tests

    def _emit_zero_torque(self) -> None:
        """Publish one zero-torque frame.  Errors are logged, not raised."""
        pubs = self._driver._pubs
        if pubs is None:
            return
        cmd, err = _build_zero_torque_lowcmd(mode_machine=self._driver._mode_machine)
        if err is not None or cmd is None:
            logger.debug("g1 control loop: zero-torque build refused: %s", err)
            return
        try:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        except ImportError as exc:  # pragma: no cover - hardware-only
            logger.debug("g1 control loop: zero-torque sdk missing: %s", exc)
            return
        pub_err = pubs.publish(_TOPIC_LOWCMD, LowCmd_, cmd)
        if pub_err is not None:
            logger.debug("g1 control loop: zero-torque publish failed: %s", pub_err)

    def _set_exit(self, reason: str, detail: str | None) -> None:
        with self._lock:
            if self._exit_reason is None:
                self._exit_reason = reason
                self._exit_detail = detail


def _refusal_text(refusal: dict[str, Any]) -> str:
    """Extract the text reason from a ``_refuse()`` envelope."""
    for entry in refusal.get("content", []):
        text = entry.get("text")
        if isinstance(text, str):
            return text
    return "refused"
